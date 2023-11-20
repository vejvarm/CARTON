import os
import time
import random
import logging
import torch
import numpy as np
import torch.optim
import torch.nn as nn
from pathlib import Path
from model import CARTON
from dataset import CSQADataset
from utils import SingleTaskLoss, MultiTaskLoss, AverageMeter, Scorer, Predictor, Inference

# import constants
from constants import DEVICE, ROOT_PATH, ALL_QUESTION_TYPES, MODEL_NAME
from helpers import setup_logger
from args import get_parser

parser = get_parser()
args = parser.parse_args()


# set logger
logger = setup_logger(__name__,
                      loglevel=logging.INFO,
                      handlers=[logging.FileHandler(f'{args.path_results}/{MODEL_NAME}_{args.name}test_{args.question_type}.log', 'w'),
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
    dataset = CSQADataset(args)
    vocabs = dataset.get_vocabs()
    inference_data = dataset.get_inference_data()  # TODO: check and refactor this function

    logger.info(f'Inference question type: {args.question_type}')
    logger.info('Inference data prepared')
    logger.info(f"Num of inference data: {len(inference_data)}")

    # load model
    model = CARTON(vocabs, DEVICE).to(DEVICE)

    logger.info(f"=> loading checkpoint '{args.model_path}'")
    checkpoint = torch.load(f'{ROOT_PATH}/{args.model_path}', encoding='latin1', map_location=DEVICE)
    args.start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    logger.info(f"=> loaded checkpoint '{args.model_path}' (epoch {checkpoint['epoch']})")

    # construct actions
    # TODO: just scrap all this and make our own decoder of the model outputs!
    inference = Inference(logger)
    if args.question_type == 'all':
        for qtype in ALL_QUESTION_TYPES:
            args.question_type = qtype
            predictor = Predictor(model, vocabs)
            inference.construct_actions(inference_data, predictor)
        args.question_type = 'all'
    else:
        predictor = Predictor(model, vocabs)
        inference.construct_actions(inference_data, predictor)


if __name__ == '__main__':
    main()
