import multiprocessing
import random
import re
import numpy as np
from functools import partial

import pandas
from torch.utils.data import DataLoader
import torch

from dataset import CSQADataset, collate_fn
from constants import DEVICE
from args import get_parser

import pandas as pd

parser = get_parser()
args = parser.parse_args()

# args.seed = 69  # canada, queen victoria, lefty west
args.seed = 100

random.seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available() and not args.no_cuda:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    DEVICE = f"{DEVICE}:{args.cuda_device}"
else:
    DEVICE = "cpu"


# PUNCTUATION_PATTERN = r"\s(?=[.,:;!?(){}[\]<>@#$%^&*-_+=|\\\"'/~`])"
PUNCTUATION_PATTERN = r"\s(?=[.,:;!?@%^*-_|\\\"'/~`])"


def set_start_method():
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass  # The start method can be set only once; ignore subsequent attempts.


def convert_sample(sample, vocab):
    return [vocab.itos[tok] for tok in sample]


def parallel_convert(data_batch, vocab):
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        return pool.starmap(convert_sample, [(sample, vocab) for sample in data_batch])


if __name__ == "__main__":
    set_start_method()

    # load data
    dataset = CSQADataset(args, splits=('test', ))  # assuming we already have the correct vocab cache from all splits!
    vocabs = dataset.get_vocabs()
    data_dict = dataset.get_data()
    helper_dict = dataset.get_data_helper()

    test_loader = torch.utils.data.DataLoader(data_dict['test'],
                                              batch_size=10,
                                              shuffle=True,
                                              collate_fn=partial(collate_fn, vocabs=vocabs, device=DEVICE))

    # for i, data in random.sample(test_loader, 5):
    for i, data in enumerate(test_loader):
        """
        Using model to do inference
        """
        # TODO

        # run CARTON

        # use input and NER to extract entity labels and types
        # use KG to look for entities with that label and type

        # match found entities with expected entities (accuracy)

        # match predicate_pointer output (accuracy)
        # match type_pointer output (accuracy)
        # match logical_form output (accuracy)


        """
        Below are the labels
        """
        # print(data)

        input_decoded_batch = parallel_convert(data.input, vocabs['input'])
        ner_batch = parallel_convert(data.ner, vocabs['ner'])
        coref_batch = parallel_convert(data.coref, vocabs['coref'])

        # input_decoded = [vocabs['input'].itos[tok] for tok in data.input[0]]
        # ner = [vocabs['ner'].itos[tok] for tok in data.ner[0]]
        # coref = [vocabs['coref'].itos[tok] for tok in data.coref[0]]

        input_decoded = input_decoded_batch[0]
        ner = ner_batch[0]
        coref = coref_batch[0]

        lf_decoded = [vocabs['logical_form'].itos[tok] for tok in data.logical_form[0]]
        pp_decoded = [vocabs['predicate_pointer'].itos[tok] for tok in data.predicate_pointer[0]]
        tp_decoded = [vocabs['type_pointer'].itos[tok] for tok in data.type_pointer[0]]

        df_inp = pandas.DataFrame.from_dict({"input": input_decoded, "ner": ner, "coref": coref})
        df_out = pandas.DataFrame.from_dict({"lf": lf_decoded, "pp": pp_decoded, "tp": tp_decoded})

        # find all B-'s ... extract the type_id from there
        # get position spans for the entities from B I I ... extract labels from the input
        ### Make into mapping:
        entities = {"NA": []}
        sent = []
        entity = None
        pos = None
        idx = 0
        for tok in input_decoded:
            if ner[idx] == 'O':
                if pos is not None:
                    if pos == "NA":
                        entities["NA"].append(entity)
                    else:
                        entities[pos] = entity
                    sent.append(entity)
                    entity = None
                    pos = None
                sent.append(tok)
                idx += 1
                continue

            if ner[idx].startswith("B"):
                pos = coref[idx]
                entity = tok
            elif ner[idx].startswith("I"):
                if tok.startswith('##'):
                    entity += tok[2:]
                else:
                    entity += f" {tok}"
            idx += 1

        # print(f"### input: {re.sub(PUNCTUATION_PATTERN, '', ' '.join(input_decoded).replace(' ##', ''))}")
        print(f"### input: {sent}")
        print(entities)
        print(df_inp)
        print(df_out)
        print("##########################################\n")
        ###

        # in lf
        # fill relation with decoded relation_pointer
        # fill type with decoded type_pointer  # NOTE: Insert doesn't use type_pointer at all

        if i >= 5:
            break


"""
inp: tensor([[25, 19,  9, 305,  8, 16, 38,  7,  1, 1815, 11, 490, 1, 10, 85, 44, 16, 38, 382,  7]], device='cuda:0')
ner: tensor([[ 0,  0,  0,   0,  0,  0,  0,  0,  0,   20,  0,  20, 0,  0,  0,  0,  0,  0,   0,  0]], device='cuda:0')
crf: tensor([[ 2,  2,  2,   2,  2,  2,  2,  2,  2,    2,  2,   2, 2,  2,  2,  2,  2,  2,   2,  2]], device='cuda:0')

lf:  tensor([[7, 8, 4, 5, 6]], device='cuda:0')
pp:  tensor([[0, 0, 0, 5, 0]], device='cuda:0')
tp:  tensor([[0, 0, 0, 0, 8]], device='cuda:0')

ep: tensor([[1, 0, 105, 25, 107701]], device='cuda:0') ... # NOTE:  unused

"""

