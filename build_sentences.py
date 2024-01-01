import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))
import json
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

# args.seed = 69  # canada, queen victoria, lefty west
# args.seed = 100
args.batch_size = 1
assert args.batch_size == 1, "batch_size must be 1 for building filled logical forms"

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
                if entity is not None:
                    # Add the previous entity to the dictionary and sentence
                    if pos == "NA":
                        entities["NA"].append(entity)
                    else:
                        entities[pos] = entity
                    sent.append(entity)

                # Start a new entity
                pos = coref_tok
                entity = tok
            elif ner_tok.startswith("I"):
                if tok.startswith('##'):
                    entity += tok[2:]
                else:
                    entity += f" {tok}"
            elif ner_tok in ['O', "[PAD]"]:
                if entity is not None:
                    # Finish the current entity and add it to the dictionary and sentence
                    if pos == "NA":
                        entities["NA"].append(entity)
                    else:
                        entities[pos] = entity
                    sent.append(entity)
                    entity = None
                    pos = None

                if ner_tok == "[PAD]":
                    break

                # Add the current non-entity token to the sentence
                sent.append(tok)

        # Check if there's an unfinished entity at the end
        if entity is not None:
            if pos == "NA":
                entities["NA"].append(entity)
            else:
                entities[pos] = entity
            sent.append(entity)

        batch_entities_sentences.append({"entities": entities, "sent": sent})

    return batch_entities_sentences


def compose_logical_form(inp, pred_lf, pred_coref, pred_pp, pred_tp, entities):
    inp_str = " ".join(inp)

    lf = pred_lf
    # ner = preds[NER][0]
    coref = pred_coref
    pp = pred_pp
    tp = pred_tp

    # for CSQA it works, but we get coref indexing errors for Merged, as one entity label belongs to more than one lf `entity` slot
    # TODO: fix this:
    #   ['entity', 'relation', 'entity', 'insert', 'entity', 'relation', 'entity']
    #   ['0', '1']
    #   {'NA': [], '0': 'japan national route 415', '1': 'national highway of japan'}
    #   ['1']
    #   {'NA': [], '0': 'japan national route 415', '1': 'national highway of japan'}
    #   []
    #   {'NA': [], '0': 'japan national route 415', '1': 'national highway of japan'}

    composed_lf = []
    ent_keys = sorted([k for k in entities.keys() if k != "NA"], key=lambda x: int(x))
    ent_keys_filled = []
    if ent_keys:
        for i in range(int(ent_keys[-1]) + 1):
            if str(i) in ent_keys:
                ent_keys_filled.append(str(i))
            else:
                ent_keys_filled.append(ent_keys[0])
    for i, act in enumerate(lf):
        if act == "entity":
            try:
                composed_lf.append(entities[ent_keys_filled.pop(0)])
            except IndexError:
                # print(f"ent idx: {ent_idx} | {entities}")
                composed_lf.append(entities["NA"].pop())
        elif act == "relation":
            composed_lf.append(pp[i])
        elif act == "type":
            composed_lf.append(tp[i])
        else:
            composed_lf.append(act)

    return composed_lf


if __name__ == "__main__":
    save_path = ROOT_PATH.joinpath(args.path_inference).joinpath(args.name)
    print(f"BATCH SIZE: {args.batch_size}")
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

    csv_path = ROOT_PATH.joinpath("csv")
    csv_path.mkdir(exist_ok=True, parents=True)
    with torch.no_grad():
        with tqdm(total=total_batches, desc=f'Inference') as pbar:
            for i, batch in enumerate(test_loader):
                logical_form, predicate_t, type_t = prepad_tensors_with_start_tokens(batch, vocabs, device=DEVICE)

                # print(vocabs[LOGICAL_FORM].stoi)
                # print(logical_form.shape)
                # tg_lf = torch.zeros(logical_form.shape[0], 1, dtype=torch.long).to(DEVICE)
                tg_lf = logical_form[:, :1]
                # tg_lf = torch.hstack([tg_lf, logical_form[:, :-1]])
                # print(tg_lf)
                # exit()

                # infer predictions from model
                for j in range(logical_form.shape[1] - 1):
                    output = model(batch.input, tg_lf)  # dict

                    pred = torch.argmax(output[LOGICAL_FORM], dim=1).view(args.batch_size, -1)

                    tg_lf = torch.hstack([tg_lf, pred[:, j:j+1]])
                    print(tg_lf)
                preds = {
                    k: torch.argmax(output[k], dim=1).view(args.batch_size, -1) for k in [LOGICAL_FORM, NER,
                                                                                          COREF, PREDICATE_POINTER,
                                                                                          TYPE_POINTER]
                }
                print(preds[LOGICAL_FORM])
                print(preds[LOGICAL_FORM].shape)

                # get labels from data
                target = {
                    LOGICAL_FORM: logical_form[:, 1:].contiguous().view(args.batch_size, -1),
                    NER: batch.ner.contiguous().view(args.batch_size, -1),
                    COREF: batch.coref.contiguous().view(args.batch_size, -1),
                    PREDICATE_POINTER: predicate_t[:, 1:].contiguous().view(args.batch_size, -1),
                    TYPE_POINTER: type_t[:, 1:].contiguous().view(args.batch_size, -1),
                }

                # Convert batches of tensors to lists
                i_decoded = [[vocabs[INPUT].itos[tok] for tok in sample if tok != pad[INPUT]] for sample in batch.input]
                # ner_batch = [[vocabs['ner'].itos[tok] for tok in sample if tok != pad['ner']] for sample in batch.ner]
                t_decoded = {
                    k: [[vocabs[k].itos[tok] for tok in sample if tok != pad[k]] for sample in target[k]] for k in target.keys()

                }
                # TODO: fix PADDING token MISMATCH in predictions
                preds_decoded = {
                    k: [[vocabs[k].itos[tok] for tok in sample] for sample in preds[k]][:len(t_decoded[k])] for k in preds.keys()  # !HACK with len(t_decoded[k])
                    # k: [[vocabs[k].itos[tok] for tok in sample] for sample in preds[k]] for k in preds.keys()
                    # k: [[vocabs[k].itos[tok] for tok in preds[k][i] if tok != pad[k]] for i in range(len(t_decoded[k]))] for k in preds.keys()
                }

                print(t_decoded)
                print(preds_decoded)

                exit()

                batch_results = extract_entities_and_sentences(i_decoded, t_decoded[NER], t_decoded[COREF])

                # TODO: what do we do with [PAD] tokens (Remove/keep and mask?) when calculating accuracy?
                # find all B-'s ... extract the type_id from there
                composed_lfs = []
                for b in range(args.batch_size):
                    entities = batch_results[b]['entities']
                    sent = batch_results[b]['sent']

                    composed_lf = compose_logical_form(i_decoded[b], preds_decoded[LOGICAL_FORM][b],
                                                       preds_decoded[COREF][b], preds_decoded[PREDICATE_POINTER][b],
                                                       preds_decoded[TYPE_POINTER][b], entities)

                    # make into function >>>
                    df_inp = pandas.DataFrame.from_dict({"input": i_decoded[b],
                                                         "ner (p)": preds_decoded[NER][b],
                                                         "ner (t)": t_decoded[NER][b],
                                                         "coref (p)": preds_decoded[COREF][b],
                                                         "coref (t)": t_decoded[COREF][b]})
                    df_out = pandas.DataFrame.from_dict({"lf (p)": preds_decoded[LOGICAL_FORM][b],
                                                         "lf (t)": t_decoded[LOGICAL_FORM][b],
                                                         "pp (p)": preds_decoded[PREDICATE_POINTER][b],
                                                         "pp (t)": t_decoded[PREDICATE_POINTER][b],
                                                         "tp (p)": preds_decoded[TYPE_POINTER][b],
                                                         "tp (t)": t_decoded[TYPE_POINTER][b]})

                    with csv_path.joinpath(f'test_{i}-{b}-asent.json').open("w") as f:
                        json.dump({'sent': sent, 'entities': entities}, f, indent=4)
                    with csv_path.joinpath(f"test_{i}-{b}-binp.csv").open("w") as f:
                        df_inp.to_csv(f)
                    with csv_path.joinpath(f"test_{i}-{b}-cout.csv").open("w") as f:
                        df_out.to_csv(f)
                    # <<< make into function

                # print(f"### input: {re.sub(PUNCTUATION_PATTERN, '', ' '.join(input_decoded).replace(' ##', ''))}")
                print(f"### input: {sent}")
                # print(preds_decoded[LOGICAL_FORM])
                print(t_decoded[LOGICAL_FORM])
                # print(entities)
                print(composed_lf)
                # print(preds_decoded[NER])
                # print(t_decoded[NER])
                # print(preds_decoded[COREF])
                # print(t_decoded[COREF])
                # print(entities)
                # print(df_inp)
                # print(df_out)
                print("##########################################\n")

                # in lf
                #   fill relation with decoded relation_pointer
                #   fill type with decoded type_pointer  # NOTE: Insert doesn't use type_pointer
                #   fill entities with id=search(label, type) but first order them by coref
                # TODO: \O.o/ dont forget our nice extraction code above

                pbar.update(1)

                # break

                if i >= 100:
                    break
