import json
import pathlib
import random
import re
import numpy as np
from functools import partial

import pandas
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

from dataset import CSQADataset, collate_fn
from model import CARTON
from utils import Predictor, AverageMeter, MultiTaskAcc, MultiTaskAccTorchmetrics, MultiTaskRecTorchmetrics

from constants import DEVICE, LOGICAL_FORM, COREF, NER, INPUT, PREDICATE_POINTER, TYPE_POINTER, ROOT_PATH
from args import get_parser

parser = get_parser()
args = parser.parse_args()

# args.seed = 69  # canada, queen victoria, lefty west
# args.seed = 100
# args.batch_size = 1

# TODO: figure out what I just did :D
# TODO: what would it take to calculate accuracy based on completel logical form!?

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


def extract_entities_and_sentences(input_batch, ner_batch, coref_batch):

    # TODO: fix `extract_entities_and_sentences` (each token following after entity is cut off)

    batch_entities_sentences = []
    for input_decoded, ner, coref in zip(input_batch, ner_batch, coref_batch):
        entities = {"NA": []}
        sent = []
        entity = None
        pos = None
        for idx, tok in enumerate(input_decoded):
            ner_tok = ner[idx]
            coref_tok = coref[idx]

            if ner_tok.startswith("B"):
                pos = coref_tok
                entity = tok
            elif ner_tok.startswith("I"):
                if tok.startswith('##'):
                    entity += tok[2:]
                else:
                    entity += f" {tok}"
            elif ner_tok in ['O', "[PAD]"]:
                if pos is None:
                    if ner_tok == "[PAD]":
                        break
                    sent.append(tok)
                    continue

                if pos == "NA":
                    entities["NA"].append(entity)
                else:
                    entities[pos] = entity
                sent.append(entity)
                entity = None
                pos = None

        batch_entities_sentences.append({"entities": entities, "sent": sent})
    return batch_entities_sentences


def save_meter_to_file(meter_dict: dict[str: AverageMeter], path_to_file: pathlib.Path):
    results = {nm: metric.avg.cpu().tolist() for nm, metric in meter_dict.items()}
    results["average"] = np.mean([v for v in results.values()])

    with path_to_file.open("w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    save_path = ROOT_PATH.joinpath(args.path_inference).joinpath(args.name)
    print(f"results will be saved to `{save_path}`.")

    # load data
    dataset = CSQADataset(args, splits=('test', ))  # assuming we already have the correct vocab cache from all splits!
    vocabs = dataset.get_vocabs()
    data_dict = dataset.get_data()
    helper_dict = dataset.get_data_helper()

    test_loader = torch.utils.data.DataLoader(data_dict['test'],
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              collate_fn=partial(collate_fn, vocabs=vocabs, device=DEVICE))
    total_batches = (len(test_loader.dataset) + args.batch_size - 1) // args.batch_size

    pad = {k: v.stoi["[PAD]"] for k, v in vocabs.items() if k != "id"}
    num_classes = {k: len(v) for k, v in vocabs.items() if k != "id"}

    model = CARTON(vocabs, DEVICE).to(DEVICE)
    print(f"=> loading checkpoint '{args.model_path}'")
    checkpoint = torch.load(f'{ROOT_PATH}/{args.model_path}', encoding='latin1', map_location=DEVICE)
    args.start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    print(f"=> loaded checkpoint '{args.model_path}' (epoch {checkpoint['epoch']})")

    predictor = Predictor(model, vocabs)



    # acc_calculator = MultiTaskAcc(DEVICE)
    # accuracies = {LOGICAL_FORM: AverageMeter(),
    #               NER: AverageMeter(),
    #               COREF: AverageMeter(),
    #               PREDICATE_POINTER: AverageMeter(),
    #               TYPE_POINTER: AverageMeter()}

    acc_calculator = MultiTaskAccTorchmetrics(num_classes, pads=pad, device=DEVICE, averaging_type='micro')  # !we use 'micro' to NOT bloat up classes, which don't have much samples (that would be useful for training)
    accuracies = {LOGICAL_FORM: AverageMeter(),
                  NER: AverageMeter(),
                  COREF: AverageMeter(),
                  PREDICATE_POINTER: AverageMeter(),
                  TYPE_POINTER: AverageMeter()}

    rec_calculator = MultiTaskRecTorchmetrics(num_classes, pads=pad, device=DEVICE)
    recalls = {LOGICAL_FORM: AverageMeter(),
               NER: AverageMeter(),
               COREF: AverageMeter(),
               PREDICATE_POINTER: AverageMeter(),
               TYPE_POINTER: AverageMeter()}

    # for i, data in random.sample(test_loader, 5):
    with torch.no_grad():
        with tqdm(total=total_batches, desc=f'Inference') as pbar:
            for i, batch in enumerate(test_loader):
                """
                Using model to do inference
                """

                # ner = batch.ner
                # coref = batch.coref
                # predicate_t = batch.predicate_pointer
                # type_t = batch.type_pointer

                # compute output
                output = model(batch.input, batch.logical_form[:, :-1])
                # use input and NER to extract entity labels and types
                # use KG to look for entities with that label and type

                # match found entities with expected entities (accuracy)

                # match predicate_pointer output (accuracy)
                # match type_pointer output (accuracy)
                # match logical_form output (accuracy)

                target = {
                    LOGICAL_FORM: batch.logical_form[:, 1:].contiguous().view(-1),
                    NER: batch.ner.contiguous().view(-1),
                    COREF: batch.coref.contiguous().view(-1),
                    PREDICATE_POINTER: batch.predicate_pointer[:, 1:].contiguous().view(-1),
                    TYPE_POINTER: batch.type_pointer[:, 1:].contiguous().view(-1),
                }

                accs = acc_calculator(output, target)
                for name, meter in accuracies.items():
                    meter.update(accs[name])

                recs = rec_calculator(output, target)
                for name, meter in recalls.items():
                    meter.update(recs[name])

                # # ### DEBUG
                # """
                # Below are the labels
                # """
                # # Convert tensors to lists
                # input_batch = [[vocabs['input'].itos[tok] for tok in sample if tok != pad['input']] for sample in batch.input]
                # ner_batch = [[vocabs['ner'].itos[tok] for tok in sample if tok != pad['ner']] for sample in batch.ner]
                # coref_batch = [[vocabs['coref'].itos[tok] for tok in sample if tok != pad['coref']] for sample in batch.coref]
                # lf_batch = [[vocabs['logical_form'].itos[tok] for tok in sample if tok != pad['logical_form']] for sample in batch.logical_form]
                # pp_batch = [[vocabs['predicate_pointer'].itos[tok] for tok in sample if tok != pad['predicate_pointer']] for sample in batch.predicate_pointer]
                # tp_batch = [[vocabs['type_pointer'].itos[tok] for tok in sample if tok != pad['type_pointer']] for sample in batch.type_pointer]
                #
                # batch_results = extract_entities_and_sentences(input_batch, ner_batch, coref_batch)
                #
                # # TODO: what do we do with [PAD] tokens (Remove/keep and mask?) when calculating accuracy?
                # # find all B-'s ... extract the type_id from there
                # entities = batch_results[0]['entities']
                # sent = batch_results[0]['sent']
                #
                # input_decoded = input_batch[0]
                # ner = ner_batch[0]
                # coref = coref_batch[0]
                #
                # lf_decoded = lf_batch[0]
                # pp_decoded = pp_batch[0]
                # tp_decoded = tp_batch[0]
                #
                # df_inp = pandas.DataFrame.from_dict({"input": input_decoded, "ner": ner, "coref": coref})
                # df_out = pandas.DataFrame.from_dict({"lf": lf_decoded, "pp": pp_decoded, "tp": tp_decoded})
                #
                # csv_path = ROOT_PATH.joinpath("csv")
                # csv_path.mkdir(exist_ok=True, parents=True)
                # with csv_path.joinpath(f'test_{i}-asent.json').open("w") as f:
                #     json.dump({'sent': sent, 'entities': entities}, f, indent=4)
                # with csv_path.joinpath(f"test_{i}-binp.csv").open("w") as f:
                #     df_inp.to_csv(f)
                # with csv_path.joinpath(f"test_{i}-cout.csv").open("w") as f:
                #     df_out.to_csv(f)
                #
                # # print(f"### input: {re.sub(PUNCTUATION_PATTERN, '', ' '.join(input_decoded).replace(' ##', ''))}")
                # print(f"### input: {sent}")
                # print(entities)
                # print(df_inp)
                # print(df_out)
                # print("##########################################\n")
                #
                # # in lf
                # #   fill relation with decoded relation_pointer
                # #   fill type with decoded type_pointer  # NOTE: Insert doesn't use type_pointer
                # #   fill entities with id=search(label, type) but first order them by coref
                # # TODO: \O.o/ dont forget our nice extraction code above
                # # ### DEBUG

                pbar.set_postfix({'lf': f"{accuracies[LOGICAL_FORM].avg:.4f}|{recalls[LOGICAL_FORM].avg:.4f}",
                                  'ner': f"{accuracies[NER].avg:.4f}|{recalls[NER].avg:.4f}",
                                  'coref': f"{accuracies[COREF].avg:.4f}|{recalls[COREF].avg:.4f}",
                                  'pp': f"{accuracies[PREDICATE_POINTER].avg:.4f}|{recalls[PREDICATE_POINTER].avg:.4f}",
                                  'tp': f"{accuracies[TYPE_POINTER].avg:.4f}|{recalls[TYPE_POINTER].avg:.4f}"})
                pbar.update(1)

                # break

                # if i >= 5:
                #     break

    # save metric results
    save_path.mkdir(exist_ok=True, parents=True)
    path_to_acc = save_path.joinpath("acc.json")
    path_to_rec = save_path.joinpath("rec.json")
    save_meter_to_file(accuracies, path_to_acc)
    save_meter_to_file(recalls, path_to_rec)


"""
inp: tensor([[25, 19,  9, 305,  8, 16, 38,  7,  1, 1815, 11, 490, 1, 10, 85, 44, 16, 38, 382,  7]], device='cuda:0')
ner: tensor([[ 0,  0,  0,   0,  0,  0,  0,  0,  0,   20,  0,  20, 0,  0,  0,  0,  0,  0,   0,  0]], device='cuda:0')
crf: tensor([[ 2,  2,  2,   2,  2,  2,  2,  2,  2,    2,  2,   2, 2,  2,  2,  2,  2,  2,   2,  2]], device='cuda:0')

lf:  tensor([[7, 8, 4, 5, 6]], device='cuda:0')
pp:  tensor([[0, 0, 0, 5, 0]], device='cuda:0')
tp:  tensor([[0, 0, 0, 0, 8]], device='cuda:0')

ep: tensor([[1, 0, 105, 25, 107701]], device='cuda:0') ... # NOTE:  unused

"""

