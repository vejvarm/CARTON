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

random.seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available() and not args.no_cuda:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    DEVICE = f"{DEVICE}:{args.cuda_device}"
else:
    DEVICE = "cpu"


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

    # RECALL Metric (Macro averaged) (except NER)
    # !this is a HACK, we only calculate NER here on the CPU, because it does not fit on GPU
    recall_modules = [NER]
    recall_averaging_types = {mn: 'macro' for mn in recall_modules}
    rec_calculator = MultiTaskRecTorchmetrics(num_classes, pads=pad, device='cpu',
                                              averaging_types=recall_averaging_types, module_names=recall_modules)
    recalls = {mn: AverageMeter() for mn in recall_modules}

    with torch.no_grad():
        with tqdm(total=total_batches, desc=f'Inference') as pbar:
            for i, batch in enumerate(test_loader):
                """
                Using model to do inference
                """

                # compute output
                output = model(batch.input, batch.logical_form[:, :-1])

                target = {
                    LOGICAL_FORM: batch.logical_form[:, 1:].contiguous().view(-1),
                    NER: batch.ner.contiguous().view(-1),
                    COREF: batch.coref.contiguous().view(-1),
                    PREDICATE_POINTER: batch.predicate_pointer[:, 1:].contiguous().view(-1),
                    TYPE_POINTER: batch.type_pointer[:, 1:].contiguous().view(-1),
                }

                recs = rec_calculator({k: v.detach().cpu() for k, v in output.items()},
                                      {k: v.detach().cpu() for k, v in target.items()})
                for name, meter in recalls.items():
                    meter.update(recs[name])

                pbar.set_postfix({'ner': f"{recalls[NER].avg:.4f}"})
                pbar.update(1)

    # save metric results
    save_path.mkdir(exist_ok=True, parents=True)
    path_to_rec = save_path.joinpath("rec-ner.json")
    save_meter_to_file(recalls, path_to_rec)
