import os
import sys
import time
import random
import logging
import torch
import numpy as np
import torch.optim
import torch.nn as nn
from pathlib import Path
from args import get_parser
from model import CARTON
from dataset import CSQADataset
from torchtext.data import BucketIterator
# from torch.utils.data.dataloader import DataLoader  # TODO: implement BucketIterator with this
from utils import (NoamOpt, AverageMeter,
                    SingleTaskLoss, MultiTaskLoss,
                    save_checkpoint, init_weights)

from helpers import setup_logger

# import constants
from constants import *

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


def main():
    # load data
    dataset = CSQADataset()
    vocabs = dataset.get_vocabs()
    train_data, val_data, _ = dataset.get_data()
    train_helper, val_helper, _ = dataset.get_data_helper()

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
    train_loader, val_loader = BucketIterator.splits((train_data, val_data),
                                                    batch_size=args.batch_size,
                                                    sort_within_batch=False,
                                                    sort_key=lambda x: len(x.input),
                                                    device=DEVICE)

    # ANCHOR: BucketIterator use deprecated ... use torch.utils.data.dataloader.DataLoader / DataLoader2

    LOGGER.info('Loaders prepared.')
    LOGGER.info(f"Training data: {len(train_data.examples)}")
    LOGGER.info(f"Validation data: {len(val_data.examples)}")
    LOGGER.info(f'Question example: {train_data.examples[0].input}')
    LOGGER.info(f'Logical form example: {train_data.examples[0].logical_form}')
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

    # switch to train mode
    model.train()

    end = time.time()
    batch_progress_old = -1
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

        ner_out = output[NER].detach().argmax(1).tolist()
        LOGGER.debug(f'ner_out in train: ({len(ner_out)}) {ner_out}')
        ner_str = [vocabs[NER].itos[i] for i in ner_out][1:-1]
        LOGGER.debug(f'ner_str in train: ({len(ner_str)}) {ner_str}')
        ner_indices = {k: tag.split('-')[-1] for k, tag in enumerate(ner_str) if
                                   tag.startswith(B) or tag.startswith(I)}  # idx: type_id
        LOGGER.debug(f'ner_indices in train: ({len(ner_indices)}) {ner_indices}')
        # coref_indices = {k: tag for k, tag in enumerate(coref_str) if tag not in ['NA']}
        # create a ner dictionary with index as key and entity as value
        # TODO: WE ACTUALLY DON'T NEED ANY OF THIS!
        # TODO: THE NER MODULE IS NOT LEARING ANYTHING NEW ... we don't need a specific loss for that
        # TODO: ONLY THING WE NEED IS TO ADD NEW ENTRIES TO THE CSQA Dataset!

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

        batch_progress = int(((i+1)/len(train_loader))*100)  # percentage
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