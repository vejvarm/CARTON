import random
import logging

import numpy as np

from dataset import CSQADataset
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
    _ = dataset.build_vocabs(args.stream_data)
    print("done")
    # data_dict, helper_dict = dataset.preprocess_data()