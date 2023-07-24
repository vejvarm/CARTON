import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import logging
import json
import argparse
from glob import glob
from pathlib import Path
from action_annotators.annotate import ActionAnnotator
from ner_annotators.annotate import NERAnnotator
from constants import ROOT_PATH
from helpers import connect_to_elasticsearch
from tqdm import tqdm

# add arguments to parser
parser = argparse.ArgumentParser(description='Preprocess CSQA dataset')
parser.add_argument('--partition', default='train', choices=['train', 'val', 'test'], type=str, help='partition to preprocess')
parser.add_argument('--annotation_task', default='all', choices=['actions', 'ner', 'all'], help='annotation task to perform')
parser.add_argument('--read_folder', default='/data/CSQA_v9', help='folder to read conversations')
parser.add_argument('--write_folder', default='/data/final/csqa', help='folder to write the annotated conversations')
args = parser.parse_args()

# read data and create csqa data dict
# dict: partition -> folder -> file -> conversation
path_to_csqa_split = ROOT_PATH.joinpath(args.read_folder).joinpath(args.partition)
print(path_to_csqa_split)
# csqa_files = glob(f'{ROOT_PATH}{args.read_folder}/{args.partition}/*/*.json')
csqa_files = list(path_to_csqa_split.glob("**/QA_*.json"))

csqa_data = {}
for path in csqa_files:
    folder = path.parent.name
    file = path.name
    # folder = path.rsplit('/', 1)[0].rsplit('/', 1)[-1]
    # file = path.rsplit('/', 1)[-1]
    if folder not in csqa_data:
        csqa_data[folder] = {}

    with open(path) as json_file:
        csqa_data[folder][file] = json.load(json_file)
print(f'Done, {len(csqa_data)} folders loaded!')

# load kg
# TODO: make more universal for other types of data source (ZODBKG vs JSONKG vs Elasticsearch client)
client = connect_to_elasticsearch()

# create ner and action annotator
action_annotator = None
ner_annotator = None
if args.annotation_task == 'all':
    action_annotator = ActionAnnotator(client)
    ner_annotator = NERAnnotator(client, args.partition)
elif args.annotation_task == 'actions':
    action_annotator = ActionAnnotator(client)
elif args.annotation_task == 'ner':
    ner_annotator = NERAnnotator(client, args.partition)
else:
    raise KeyError("--annotation_task must be one of 'all', 'actions', 'ner' ('{args.annotation_task}' was given).")

# create annotated data
total_conv = len(csqa_files)
conv_counter = 0
tic = time.perf_counter()
print(f'Preprocessing folders for partition {args.partition}')
for path in tqdm(csqa_files):
    folder = path.parent.name
    file = path.name

    # get conversation
    # NOTE: conversation == one QA_*.json file
    conversation = json.load(path.open())

    # annotate conversation
    if action_annotator is not None:
        conversation = action_annotator(conversation)
    if ner_annotator is not None:
        conversation = ner_annotator(conversation)

    # create path
    write_folder = Path(args.write_folder)
    if write_folder.is_relative_to(ROOT_PATH):
        rel_write_folder = Path(args.write_folder).relative_to(ROOT_PATH)
        conv_path = ROOT_PATH.joinpath(rel_write_folder).joinpath(args.partition).joinpath(folder)
    else:
        conv_path = write_folder.joinpath(args.partition).joinpath(folder)
    # conv_path = f'{str(ROOT_PATH)}{args.write_folder}/{args.partition}/{folder}'
    conv_path.mkdir(parents=True, exist_ok=True)

    # write conversation
    with conv_path.joinpath(file).open('w') as json_file:
        json.dump(conversation, json_file, ensure_ascii=False, indent=4)
