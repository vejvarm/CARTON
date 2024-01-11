import pathlib
import sys
sys.path.append(str(pathlib.Path(__file__).parent.parent))

import json
import random
import numpy as np

import pandas
import torch
from transformers import BertTokenizer

from dataset import CSQADataset, prepad_tensors_with_start_tokens, SingleInput
from model import CARTON

from constants import DEVICE, LOGICAL_FORM, COREF, NER, INPUT, PREDICATE_POINTER, TYPE_POINTER, ROOT_PATH
from args import get_parser

parser = get_parser()
args = parser.parse_args()

# TODO: what would it take to calculate accuracy based on complete logical form!?

if torch.cuda.is_available() and not args.no_cuda:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    DEVICE = f"{DEVICE}:{args.cuda_device}"
else:
    DEVICE = "cpu"


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
                if entity is not None:
                    if tok.startswith('##'):
                        entity += tok[2:]
                    else:
                        entity += f" {tok}"
                else:
                    # Start a new entity
                    pos = coref_tok
                    entity = tok

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


def compose_logical_form(inp, pred_lf, pred_coref, pred_pp, pred_tp, entities, eid2lab: dict = None, pid2lab: dict = None):
    inp_str = " ".join(inp)

    lf = pred_lf
    coref = pred_coref
    pp = pred_pp
    tp = pred_tp

    composed_lf = ""
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
                composed_lf += entities[ent_keys_filled.pop(0)]
            except IndexError:
                # print(f"ent idx: {ent_idx} | {entities}")
                try:
                    composed_lf += entities["NA"].pop()
                except IndexError:
                    # print("No more entities to fill in logical form")
                    composed_lf += "[UNK]$ENTITY"
            composed_lf += ", "
        elif act == "relation":
            if pid2lab is not None:
                composed_lf += pid2lab[pp[i]]
            else:
                composed_lf += pp[i]
            composed_lf += ", "
        elif act == "type":
            if eid2lab is not None:
                composed_lf += eid2lab[tp[i]]
            else:
                composed_lf += tp[i]
            composed_lf += ", "
        else:
            composed_lf += act + "("

    return composed_lf


if __name__ == "__main__":
    # load data
    dataset = CSQADataset(args, splits=('test', ))  # assuming we already have the correct vocab cache from all splits!
    vocabs = dataset.build_vocabs(args.stream_data)

    # load KG labels
    eid2lab_dict = json.load(ROOT_PATH.joinpath("knowledge_graph/items_wikidata_n.json").open())
    pid2lab_dict = json.load(ROOT_PATH.joinpath("knowledge_graph/index_rel_dict.json").open())

    eid2lab_dict.update({"NA": "[UNK]$TYPE"})
    pid2lab_dict.update({"NA": "[UNK]$RELATION"})

    pad = {k: v.stoi["[PAD]"] for k, v in vocabs.items() if k != "id"}
    num_classes = {k: len(v) for k, v in vocabs.items() if k != "id"}

    model = CARTON(vocabs, DEVICE).to(DEVICE)
    model.eval()
    print(f"=> loading checkpoint '{args.model_path}'")
    checkpoint = torch.load(f'{ROOT_PATH}/{args.model_path}', encoding='latin1', map_location=DEVICE)
    args.start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    print(f"=> loaded checkpoint '{args.model_path}' (epoch {checkpoint['epoch']})")

    csv_path = ROOT_PATH.joinpath("csv").joinpath("infer_one")
    csv_path.mkdir(exist_ok=True, parents=True)

    max_lf_len = 10
    while True:
        utterance = input("Enter query: ")
        if utterance == "exit":
            break
        if utterance == "":
            print("Please enter a sentence or type `exit` to quit.")
            continue

        # tokenize user input
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        tokens = tokenizer(utterance)['input_ids']
        tokens = tokenizer.convert_ids_to_tokens(tokens)
        # print(tokenizer.convert_tokens_to_string(tokens))
        sample = SingleInput(tokens[1:-1], vocabs, device=DEVICE)

        with torch.no_grad():
            batch = sample
            logical_form, predicate_t, type_t = prepad_tensors_with_start_tokens(batch, vocabs, device=DEVICE)
            tg_lf = logical_form[:, :1]

            # infer predictions from model
            for j in range(max_lf_len):  # TODO: fix manual length setting
                output = model(batch.input, tg_lf)  # dict

                pred = torch.argmax(output[LOGICAL_FORM], dim=1).view(1, -1)
                # print(f"pred[{j}]: {pred.shape}")
                tg_lf = torch.hstack([tg_lf, pred[:, -1:]])
                # print(f"tg_lf[{j}]: {tg_lf}")
            # print(f"tg_lf: {tg_lf.shape} | lf: {logical_form.shape} | pred: {pred.shape}")
            preds = {
                k: torch.argmax(output[k], dim=1).view(1, -1) for k in [LOGICAL_FORM, NER,
                                                                               COREF, PREDICATE_POINTER,
                                                                               TYPE_POINTER]
            }

            # Convert batches of tensors to lists
            i_decoded = [[vocabs[INPUT].itos[tok] for tok in sample if tok != pad[INPUT]] for sample in batch.input]
            # TODO: fix PADDING token MISMATCH in predictions
            preds_decoded = {
                k: [[vocabs[k].itos[tok] for tok in sample if tok != pad[k]] for sample in preds[k]] for k in preds.keys()  # removing [PAD] tokens
            }

            # print(preds_decoded)

            batch_results = extract_entities_and_sentences(i_decoded, preds_decoded[NER], preds_decoded[COREF])

            # TODO: what do we do with [PAD] tokens (Remove/keep and mask?) when calculating accuracy?
            # find all B-'s ... extract the type_id from there
            composed_lfs = []
            b = 0
            entities = batch_results[b]['entities']
            sent = batch_results[b]['sent']

            composed_lf = compose_logical_form(i_decoded[b], preds_decoded[LOGICAL_FORM][b],
                                               preds_decoded[COREF][b], preds_decoded[PREDICATE_POINTER][b],
                                               preds_decoded[TYPE_POINTER][b], entities, eid2lab_dict, pid2lab_dict)

            # make into function >>>
            df_inp = pandas.DataFrame.from_dict({"input": i_decoded[b],
                                                 "ner (p)": preds_decoded[NER][b],
                                                 "coref (p)": preds_decoded[COREF][b]})
            df_out = pandas.DataFrame.from_dict({"lf (p)": preds_decoded[LOGICAL_FORM][b],
                                                 "pp (p)": preds_decoded[PREDICATE_POINTER][b],
                                                 "tp (p)": preds_decoded[TYPE_POINTER][b]})

            with csv_path.joinpath(f'infer_one-asent.json').open("w") as f:
                json.dump({'sent': sent, 'entities': entities}, f, indent=4)
            with csv_path.joinpath(f"infer_one-binp.csv").open("w") as f:
                df_inp.to_csv(f)
            with csv_path.joinpath(f"infer_one-cout.csv").open("w") as f:
                df_out.to_csv(f)
            # <<< make into function

            print(f"### input: {sent}")
            print(preds_decoded[LOGICAL_FORM])
            print(preds_decoded[NER])
            print(composed_lf)
            print("##########################################\n")
