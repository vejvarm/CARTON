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

from dataset import CSQADataset, collate_fn, prepad_tensors_with_start_tokens
from model import CARTON
from utils import Predictor, AverageMeter, MultiTaskAcc, MultiTaskAccTorchmetrics, MultiTaskRecTorchmetrics

from constants import DEVICE, LOGICAL_FORM, COREF, NER, INPUT, PREDICATE_POINTER, TYPE_POINTER, ROOT_PATH
from args import get_parser

parser = get_parser()
args = parser.parse_args()

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
    data_dict, helper_dict = dataset.preprocess_data()
    vocabs = dataset.build_vocabs(args.stream_data)

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

    # ACCURACY Metric
    # for !we use 'micro' to NOT bloat up classes, which don't have much samples (that would be useful for training)
    acc_modules = [LOGICAL_FORM, NER, COREF, PREDICATE_POINTER, TYPE_POINTER]
    acc_averaging_types = {mn: 'micro' for mn in acc_modules}
    acc_calculator = MultiTaskAccTorchmetrics(num_classes, pads=pad, device=DEVICE,
                                              averaging_types=acc_averaging_types, module_names=acc_modules)
    accuracies = {mn: AverageMeter() for mn in acc_modules}

    # RECALL Metric (Macro averaged) (except NER)
    # TODO: FIX >>> !this is a HACK, we omit NER as we do not have enough GPU memory to calculate macro averaged recall for NER vocab
    recall_modules = [LOGICAL_FORM, COREF, PREDICATE_POINTER, TYPE_POINTER]
    # TODO: <<< FIX
    recall_averaging_types = {mn: 'macro' for mn in recall_modules}
    rec_calculator = MultiTaskRecTorchmetrics(num_classes, pads=pad, device=DEVICE,
                                              averaging_types=recall_averaging_types, module_names=recall_modules)
    recalls = {mn: AverageMeter() for mn in recall_modules}

    with torch.no_grad():
        with tqdm(total=total_batches, desc=f'Inference') as pbar:
            for i, batch in enumerate(test_loader):
                """
                Using model to do inference
                """

                logical_form, predicate_t, type_t = prepad_tensors_with_start_tokens(batch, vocabs, device=DEVICE)

                # compute output
                output = model(batch.input, logical_form[:, :-1])  # TODO: we should feed one lf token at a time

                target = {
                    LOGICAL_FORM: logical_form[:, 1:].contiguous().view(-1),
                    NER: batch.ner.contiguous().view(-1),
                    COREF: batch.coref.contiguous().view(-1),
                    PREDICATE_POINTER: predicate_t[:, 1:].contiguous().view(-1),
                    TYPE_POINTER: type_t[:, 1:].contiguous().view(-1),
                }

                accs = acc_calculator(output, target)
                for name, meter in accuracies.items():
                    meter.update(accs[name])

                recs = rec_calculator(output, target)
                for name, meter in recalls.items():
                    meter.update(recs[name])

                pbar.set_postfix({'lf': f"{accuracies[LOGICAL_FORM].avg:.4f}|{recalls[LOGICAL_FORM].avg:.4f}",
                                  'ner': f"{accuracies[NER].avg:.4f}|-",  # TODO: <<< fix NER
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

