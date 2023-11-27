import time
import random
import logging
from functools import partial

import numpy as np
import torch.optim
from tqdm import tqdm

from model import CARTON
from dataset import CSQADataset, collate_fn
from torch.utils.data import DataLoader, SequentialSampler, BatchSampler, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from utils import (NoamOpt, AverageMeter, SingleTaskLoss, MultiTaskLoss, save_checkpoint, init_weights)

from helpers import setup_logger
from constants import *
from args import get_parser

parser = get_parser()
args = parser.parse_args()

LOGDIR = ROOT_PATH.joinpath(args.snapshots).joinpath("logs")
LOGDIR.mkdir(exist_ok=True, parents=True)
# set LOGGER
LOGGER = setup_logger(__name__,
                      loglevel=logging.INFO,
                      handlers=[logging.FileHandler(LOGDIR.joinpath(f"{MODEL_NAME}_{args.name}_train_{args.task}.log"), 'w'),
                                logging.StreamHandler()])

# set a seed value
random.seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available() and not args.no_cuda:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    DEVICE = f"{DEVICE}:{args.cuda_device}"
else:
    DEVICE = "cpu"


def main():
    # load data
    dataset = CSQADataset(args)  # load all data from all splits to build full vocab from all splits
    data_dict, helper_dict = dataset.preprocess_data()
    vocabs = dataset.build_vocabs(args.stream_data)

    # load model
    model = CARTON(vocabs, DEVICE).to(DEVICE)

    # initialize model weights
    init_weights(model)

    LOGGER.info(f"Model: `{MODEL_NAME}`, Experiment: `{args.name}`")
    LOGGER.info(f'The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')

    # define loss function (criterion)
    criterion = {
        LOGICAL_FORM: SingleTaskLoss,
        NER: SingleTaskLoss,
        COREF: SingleTaskLoss,
        PREDICATE_POINTER: SingleTaskLoss,
        TYPE_POINTER: SingleTaskLoss,
        MULTITASK: MultiTaskLoss
    }[args.task](ignore_index=vocabs[LOGICAL_FORM].stoi[PAD_TOKEN], device=DEVICE)

    single_task_loss = SingleTaskLoss(ignore_index=vocabs[LOGICAL_FORM].stoi[PAD_TOKEN])

    # define optimizer
    optimizer = NoamOpt(torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    if args.resume:
        if os.path.isfile(args.resume):
            LOGGER.info(f"=> loading checkpoint '{args.resume}''")
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint[EPOCH]
            best_val = checkpoint[BEST_VAL]
            model.load_state_dict(checkpoint[STATE_DICT])
            optimizer.optimizer.load_state_dict(checkpoint[OPTIMIZER])
            LOGGER.info(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint[EPOCH]})")
        else:
            LOGGER.info(f"=> no checkpoint found at '{args.resume}'")
            best_val = float('inf')
    else:
        best_val = float('inf')

    # bs = CustomBatchSampler(data_dict['train'])
    # prepare training and validation loader
    train_loader = torch.utils.data.DataLoader(data_dict['train'],
                                               # batch_size=args.batch_size,
                                               # shuffle=True,
                                               pin_memory=True,
                                               collate_fn=partial(collate_fn, vocabs=vocabs, device=DEVICE),
                                               batch_sampler=BatchSampler(RandomSampler(data_dict['train']),
                                                                          batch_size=args.batch_size,
                                                                          drop_last=False),
                                               )

    val_loader = torch.utils.data.DataLoader(data_dict['val'],
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             collate_fn=partial(collate_fn, vocabs=vocabs, device=DEVICE))

    LOGGER.info('Loaders prepared.')
    LOGGER.info(f"Training data: {len(data_dict['train'])}")
    LOGGER.info(f"Validation data: {len(data_dict['val'])}")
    # LOGGER.info(f'Question example: {data_dict['train']}')
    # LOGGER.info(f'Logical form example: {data_dict['train'].examples[0].logical_form}')
    LOGGER.info(f"Unique tokens in input vocabulary: {len(vocabs[INPUT])}")
    LOGGER.info(f"Unique tokens in logical form vocabulary: {len(vocabs[LOGICAL_FORM])}")
    LOGGER.info(f"Unique tokens in ner vocabulary: {len(vocabs[NER])}")
    LOGGER.info(f"Unique tokens in coref vocabulary: {len(vocabs[COREF])}")
    LOGGER.info(f'Epochs: {args.epochs}')
    LOGGER.info(f'Batch size: {args.batch_size}')

    tb_writer = SummaryWriter(LOGDIR.joinpath("tb"))

    # run epochs
    for epoch in range(args.start_epoch, args.epochs):
        # evaluate on validation set
        if epoch % args.valfreq == 0:
            val_loss = validate(val_loader, model, vocabs, helper_dict['val'], criterion, single_task_loss)
            best_val = min(val_loss, best_val)  # log every validation step
            save_checkpoint({
                    EPOCH: epoch,
                    STATE_DICT: model.state_dict(),
                    BEST_VAL: best_val,
                    OPTIMIZER: optimizer.optimizer.state_dict(),
                    CURR_VAL: val_loss
                },
                experiment=args.name
            )
            LOGGER.info(f'* Val loss: {val_loss:.4f}')
            tb_writer.add_scalar('val loss', val_loss, epoch)

        # train for one epoch
        train_loss = train(train_loader, model, vocabs, helper_dict['train'], criterion, optimizer, epoch)
        tb_writer.add_scalar('training loss', train_loss, epoch+1)

    # Validate and save the final epoch
    val_loss = validate(val_loader, model, vocabs, helper_dict['val'], criterion, single_task_loss)
    best_val = min(val_loss, best_val)  # log every validation step
    save_checkpoint({
        EPOCH: args.epochs,
        STATE_DICT: model.state_dict(),
        BEST_VAL: best_val,
        OPTIMIZER: optimizer.optimizer.state_dict(),
        CURR_VAL: val_loss
    },
        experiment=args.name
    )
    LOGGER.info(f'* Val loss: {val_loss:.4f}')
    tb_writer.add_scalar('val loss', val_loss, args.epochs)


def train(train_loader, model, vocabs, helper_data, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    # total_batches = len(train_loader.dataset)//args.batch_size
    total_batches = (len(train_loader.dataset) + args.batch_size - 1) // args.batch_size
    # switch to train mode
    model.train()

    end = time.time()
    batch_progress_old = -1
    with tqdm(total=total_batches, desc=f'Epoch {epoch + 1}/{args.epochs}') as pbar:
        for i, batch in enumerate(train_loader):
            # get inputs
            input = batch.input
            logical_form = batch.logical_form
            ner = batch.ner
            coref = batch.coref
            predicate_t = batch.predicate_pointer
            type_t = batch.type_pointer

            # compute output
            output = model(input, logical_form[:, :-1])
            LOGGER.debug(f'output[NER] in train: ({output[NER].shape}) {output[NER]}')
            LOGGER.debug(f'output[COREF] in train: ({output[COREF].shape}) {output[COREF]}')

            # ner_out = output[NER].detach().argmax(1).tolist()
            # LOGGER.debug(f'ner_out in train: ({len(ner_out)}) {ner_out}')
            # ner_str = [vocabs[NER].itos[i] for i in ner_out][1:-1]
            # LOGGER.debug(f'ner_str in train: ({len(ner_str)}) {ner_str}')
            # ner_indices = {k: tag.split('-')[-1] for k, tag in enumerate(ner_str) if
            #                            tag.startswith(B) or tag.startswith(I)}  # idx: type_id
            # LOGGER.debug(f'ner_indices in train: ({len(ner_indices)}) {ner_indices}')

            # prepare targets
            target = {
                LOGICAL_FORM: logical_form[:, 1:].contiguous().view(-1),
                NER: ner.contiguous().view(-1),
                COREF: coref.contiguous().view(-1),
                PREDICATE_POINTER: predicate_t[:, 1:].contiguous().view(-1),
                TYPE_POINTER: type_t[:, 1:].contiguous().view(-1),
            }

            # compute loss
            loss = criterion(output, target) if args.task == MULTITASK else criterion(output[args.task], target[args.task])

            # record loss
            losses.update(loss.detach(), input.size(0))

            # compute gradient and do Adam step
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            pbar.set_postfix({'loss': losses.val, 'avg': losses.avg})
            pbar.update(1)

            # batch_progress = int(((i+1)/total_batches)*100)  # percentage
            # if batch_progress > batch_progress_old:
            #     LOGGER.info(f'{epoch}: Batch {batch_progress:02d}% - Train loss {losses.val:.4f} ({losses.avg:.4f})')
            # batch_progress_old = batch_progress

    LOGGER.info(f'{epoch}: Train loss: {losses.avg:.4f}')
    return losses.avg


def validate(val_loader, model, vocabs, helper_data, criterion, single_task_loss):
    losses = AverageMeter()

    # record individual losses
    losses_lf = AverageMeter()
    losses_ner = AverageMeter()
    losses_coref = AverageMeter()
    losses_pred = AverageMeter()
    losses_type = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for _, batch in tqdm(enumerate(val_loader), desc="\tvalidation"):
            # get inputs
            input = batch.input
            logical_form = batch.logical_form
            ner = batch.ner
            coref = batch.coref
            predicate_t = batch.predicate_pointer
            type_t = batch.type_pointer

            # compute output
            output = model(input, logical_form[:, :-1])

            # prepare targets
            target = {
                LOGICAL_FORM: logical_form[:, 1:].contiguous().view(-1),  # reshapes into one long 1d vector
                NER: ner.contiguous().view(-1),
                COREF: coref.contiguous().view(-1),
                PREDICATE_POINTER: predicate_t[:, 1:].contiguous().view(-1),
                TYPE_POINTER: type_t[:, 1:].contiguous().view(-1),
            }

            # compute loss
            loss = criterion(output, target) if args.task == MULTITASK else criterion(output[args.task], target[args.task])

            # compute individual losses
            loss_lf = single_task_loss(output[LOGICAL_FORM], target[LOGICAL_FORM])
            loss_ner = single_task_loss(output[NER], target[NER])
            loss_coref = single_task_loss(output[COREF], target[COREF])
            loss_pred = single_task_loss(output[PREDICATE_POINTER], target[PREDICATE_POINTER])
            loss_type = single_task_loss(output[TYPE_POINTER], target[TYPE_POINTER])

            # record loss
            losses.update(loss.detach(), input.size(0))

            # record individual losses
            losses_lf.update(loss_lf.detach(), input.size(0))
            losses_ner.update(loss_ner.detach(), input.size(0))
            losses_coref.update(loss_coref.detach(), input.size(0))
            losses_pred.update(loss_pred.detach(), input.size(0))
            losses_type.update(loss_type.detach(), input.size(0))

    LOGGER.info(f"Val losses:: LF: {losses_lf.avg} | NER: {losses_ner.avg} | COREF: {losses_coref.avg} | "
                f"PRED: {losses_pred.avg} | TYPE: {losses_type.avg}")
    return losses.avg


if __name__ == '__main__':
    main()
