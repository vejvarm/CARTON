import time
import random
import logging
from dataclasses import dataclass
from functools import partial

import numpy as np
import torch.optim
from tqdm import tqdm

from model import CARTON
from dataset import CSQADataset
from torchtext.data import BucketIterator
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, SequentialSampler, BatchSampler
from utils import (NoamOpt, AverageMeter,
                    SingleTaskLoss, MultiTaskLoss,
                    save_checkpoint, init_weights)

from helpers import setup_logger

from constants import *
from args import parse_and_get_args
args = parse_and_get_args()

# set LOGGER
LOGGER = setup_logger(__name__,
                      loglevel=logging.INFO,
                      handlers=[logging.FileHandler(f'{args.path_results}/train_{args.task}.log', 'w'),
                              logging.StreamHandler()])

# set a seed value
random.seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available():
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


# def collate_fn(batch):
#     # sort the list of examples by the length of the input in descending order
#     # print(batch)
#     batch.sort(key=lambda x: len(x.input), reverse=True)
#     # separate the inputs and targets, and pad the sequences
#     # inputs, targets = zip(*batch)
#     inputs = pad_sequence(batch, padding_value=PAD_TOKEN)
#     # targets = pad_sequence(targets, padding_value=PAD_TOKEN)
#     return inputs

text_transform = lambda x: [vocab['<BOS>']] + [vocab[token] for token in tokenizer(x)] + [vocab['<EOS>']]
label_transform = lambda x: 1 if x == 'pos' else 0


@dataclass
class DataBatch:
    """
        data[split]
        [0] ... ID
        [1] ... INPUT
        [2] ... LOGICAL_FORM
        [3] ... NER
        [4] ... COREF
        [5] ... PREDICATE_POINTER
        [6] ... TYPE_POINTER
        [7] ... ENTITY
    """
    id: torch.Tensor  # str
    input: torch.Tensor  # str
    logical_form: torch.Tensor  # list[str]
    ner: torch.Tensor  # list[str]
    coref: torch.Tensor  # list[str]
    predicate_pointer: torch.Tensor  # list[int]
    type_pointer = torch.Tensor  # list[int]
    entity_pointer = torch.Tensor  # list[int]

    def __init__(self, batch: list[list[any]], vocabs: dict, device: str):
        id = []
        inp = []
        lf = []
        ner = []
        coref = []
        predicate_pointer = []
        type_pointer = []
        entity_pointer = []
        for sample in batch:
            id.append(int(sample[0]))
            inp.append(self._tensor([vocabs[INPUT].stoi[s] for s in sample[1]]))
            lf.append(self._tensor([vocabs[LOGICAL_FORM].stoi[s] for s in sample[2]]))
            ner.append(self._tensor([vocabs[NER].stoi[s] for s in sample[3]]))
            coref.append(self._tensor([vocabs[COREF].stoi[s] for s in sample[4]]))
            predicate_pointer.append(self._tensor([vocabs[PREDICATE_POINTER].stoi[s] for s in sample[5]]))
            type_pointer.append(self._tensor([vocabs[TYPE_POINTER].stoi[s] for s in sample[6]]))
            entity_pointer.append(self._tensor([vocabs[ENTITY].stoi[s] for s in sample[7]]))

        self.id = self._tensor(id).to(device)
        self.input = pad_sequence(inp,
                                  padding_value=vocabs[INPUT].stoi[PAD_TOKEN],
                                  batch_first=True).to(device)
        self.logical_form = pad_sequence(lf,
                                         padding_value=vocabs[LOGICAL_FORM].stoi[PAD_TOKEN],
                                         batch_first=True).to(device)  # ANCHOR this is not gonna work as we assume all LFs have same length, which is not true
        self.ner = pad_sequence(ner,
                                padding_value=vocabs[NER].stoi[PAD_TOKEN],
                                batch_first=True).to(device)
        self.coref = pad_sequence(coref,
                                  padding_value=vocabs[COREF].stoi[PAD_TOKEN],
                                  batch_first=True).to(device)
        self.predicate_pointer = pad_sequence(predicate_pointer,
                                              padding_value=vocabs[PREDICATE_POINTER].stoi[PAD_TOKEN],
                                              batch_first=True).to(device)
        self.type_pointer = pad_sequence(type_pointer,
                                         padding_value=vocabs[TYPE_POINTER].stoi[PAD_TOKEN],
                                         batch_first=True).to(device)
        self.entity_pointer = pad_sequence(entity_pointer,
                                           padding_value=vocabs[ENTITY].stoi[PAD_TOKEN],
                                           batch_first=True).to(device)

    @staticmethod
    def _tensor(data):
        return torch.tensor(data)


def collate_fn(batch, vocabs: dict, device: str):
    return DataBatch(batch, vocabs, device)


# train_iter = IMDB(split='train')
# train_dataloader = DataLoader(list(train_iter), batch_size=8, shuffle=True,
#                               collate_fn=collate_batch)

def batch_sampler(split_list: list, batch_size: int, pool_size=10000):
    indices = [(i, len(s[1])) for i, s in enumerate(split_list)]
    random.shuffle(indices)
    pooled_indices = []
    # create pool of indices with similar lengths
    for i in range(0, len(indices), batch_size * pool_size):
        pooled_indices.extend(sorted(indices[i:i + batch_size * pool_size], key=lambda x: x[1]))

    pooled_indices = [x[0] for x in pooled_indices]

    # yield indices for current batch
    for i in range(0, len(pooled_indices), batch_size):
        yield pooled_indices[i:i + batch_size]


def main():
    # load data
    dataset = CSQADataset()
    vocabs = dataset.get_vocabs()
    train_data, val_data, _ = dataset.get_data()  # TODO
    train_helper, val_helper, _ = dataset.get_data_helper()  # TODO

    # load model
    model = CARTON(vocabs).to(DEVICE)

    # initialize model weights
    init_weights(model)

    LOGGER.info(f'The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')

    # define loss function (criterion)
    criterion = {
        LOGICAL_FORM: SingleTaskLoss,
        NER: SingleTaskLoss,
        COREF: SingleTaskLoss,
        PREDICATE_POINTER: SingleTaskLoss,
        TYPE_POINTER: SingleTaskLoss,
        MULTITASK: MultiTaskLoss
    }[args.task](ignore_index=vocabs[LOGICAL_FORM].stoi[PAD_TOKEN])

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

    # prepare training and validation loader
    train_loader = torch.utils.data.DataLoader(train_data,
                                               collate_fn=partial(collate_fn, vocabs=vocabs, device=DEVICE),
                                               batch_sampler=batch_sampler(train_data, args.batch_size, args.pool_size))

    val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             collate_fn=partial(collate_fn, vocabs=vocabs, device=DEVICE))
    # train_loader, val_loader = BucketIterator.splits((train_data, val_data),
    #                                                 batch_size=args.batch_size,
    #                                                 sort_within_batch=False,
    #                                                 sort_key=lambda x: len(x.input),
    #                                                 device=DEVICE)

    # ANCHOR: BucketIterator use deprecated ... use torch.utils.data.dataloader.DataLoader / DataLoader2


    LOGGER.info('Loaders prepared.')
    LOGGER.info(f"Training data: {len(train_data)}")
    LOGGER.info(f"Validation data: {len(val_data)}")
    # LOGGER.info(f'Question example: {train_data}')
    # LOGGER.info(f'Logical form example: {train_data.examples[0].logical_form}')
    LOGGER.info(f"Unique tokens in input vocabulary: {len(vocabs[INPUT])}")
    LOGGER.info(f"Unique tokens in logical form vocabulary: {len(vocabs[LOGICAL_FORM])}")
    LOGGER.info(f"Unique tokens in ner vocabulary: {len(vocabs[NER])}")
    LOGGER.info(f"Unique tokens in coref vocabulary: {len(vocabs[COREF])}")
    LOGGER.info(f'Batch: {args.batch_size}')
    LOGGER.info(f'Epochs: {args.epochs}')

    # run epochs
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(train_loader, model, vocabs, train_helper, criterion, optimizer, epoch)

        # evaluate on validation set
        if (epoch+1) % args.valfreq == 0:
            val_loss = validate(val_loader, model, vocabs, val_helper, criterion, single_task_loss)
            best_val = min(val_loss, best_val)  # log every validation step
            save_checkpoint({
                EPOCH: epoch + 1,
                STATE_DICT: model.state_dict(),
                BEST_VAL: best_val,
                OPTIMIZER: optimizer.optimizer.state_dict(),
                CURR_VAL: val_loss})
            LOGGER.info(f'* Val loss: {val_loss:.4f}')


def train(train_loader, model, vocabs, helper_data, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    total_batches = len(train_loader.dataset)//args.batch_size
    # switch to train mode
    model.train()

    end = time.time()
    batch_progress_old = -1
    for i, batch in tqdm(enumerate(train_loader), total=total_batches, desc=f"Epoch {epoch}:"):
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

        ner_out = output[NER].detach().argmax(1).tolist()
        LOGGER.debug(f'ner_out in train: ({len(ner_out)}) {ner_out}')
        ner_str = [vocabs[NER].itos[i] for i in ner_out][1:-1]
        LOGGER.debug(f'ner_str in train: ({len(ner_str)}) {ner_str}')
        ner_indices = {k: tag.split('-')[-1] for k, tag in enumerate(ner_str) if
                                   tag.startswith(B) or tag.startswith(I)}  # idx: type_id
        LOGGER.debug(f'ner_indices in train: ({len(ner_indices)}) {ner_indices}')
        # coref_indices = {k: tag for k, tag in enumerate(coref_str) if tag not in ['NA']}
        # create a ner dictionary with index as key and entity as value
        # NOTE: WE ACTUALLY DON'T NEED ANY OF THIS!
        #   THE NER MODULE IS NOT LEARING ANYTHING NEW ... we don't need a specific loss for that
        #   ONLY THING WE NEED IS TO ADD NEW ENTRIES TO THE CSQA Dataset!

        # NER module in TRAIN
        # TODO: implement the ner module functionality, as in Inference,
        #   goal: missing entities added to index
        #   loss: compare added entities and their labels with the args.elastic_index_ent_full (Levenshtein distance? naah, either it's right or not)
        # ner_prediction = output[NER]
        # coref_prediction = output[COREF]
        # ner_indices = OrderedDict({k: tag.split('-')[-1] for k, tag in enumerate(ner_prediction) if
        #                            tag.startswith(B) or tag.startswith(I)})  # idx: type_id
        # coref_indices = OrderedDict({k: tag for k, tag in enumerate(coref_prediction) if tag not in ['NA']})
        # # create a ner dictionary with index as key and entity as value
        # ner_idx_ent = self.create_ner_idx_ent_dict(ner_indices, context_question)
        # output[NER]

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

        batch_progress = int(((i+1)/total_batches)*100)  # percentage
        if batch_progress > batch_progress_old:
            LOGGER.info(f'Epoch: {epoch+1} - Train loss: {losses.val:.4f} ({losses.avg:.4f}) - Batch: {batch_progress:02d}% - Time: {batch_time.sum:0.2f}s')
        batch_progress_old = batch_progress


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
        for _, batch in enumerate(val_loader):
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