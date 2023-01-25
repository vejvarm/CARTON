import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import json
import argparse
from pathlib import Path
from constants import ROOT_PATH
from tqdm import tqdm

# add arguments to parser
parser = argparse.ArgumentParser(description='Reduce dataset only to "question_type": "Simple Question (*)"')
parser.add_argument('--partition', default='train', choices=['train', 'val', 'test'], type=str, help='partition to preprocess')
parser.add_argument('--read_folder', default='data/final_w_turn_pos/csqa', help='folder to read conversations')
parser.add_argument('--write_folder', default='data/final_simple/csqa', help='folder to write the reduced dataset')
args = parser.parse_args()

# read data and create csqa data dict
# dict: partition -> folder -> file -> conversation
read_folder = ROOT_PATH.joinpath(args.read_folder).joinpath(args.partition)  # D:\data\final\csqa\train
csqa_files = read_folder.glob("*/*.json")

csqa_data = {}
print(f'Reading folders for partition {args.partition}')
total_conv = 0
for path in csqa_files:
    folder = path.parent.name
    file = path.name
    if folder not in csqa_data:
        csqa_data[folder] = {}

    with open(path, 'r', encoding='utf8') as json_file:
        csqa_data[folder][file] = json.load(json_file)
    total_conv += 1
print(f'Done, {len(csqa_data)} folders loaded!')

# NOTE: conversation == one QA_*.json file

conv_counter = 0
tic = time.perf_counter()
for folder in tqdm(csqa_data.keys()):
    csqa_folder = csqa_data[folder]
    for file in csqa_folder.keys():
        # get conversation
        conversation = csqa_folder[file]

        conversation_simple = []
        for i in range(0, len(conversation), 2):
            q_type = conversation[i]["question-type"]
            q_utterance = conversation[i]["utterance"]
            if "Simple Question (Coreferenced)" in q_type and q_utterance.startswith(("Yes", "No, ")):
                continue  # NOTE: IGNORE "Yes"/"No, I meant" types of questions
            if "Simple Question" in q_type:
                conversation_simple.extend(conversation[i:i+2])

        # create path
        conv_path = ROOT_PATH.joinpath(args.write_folder).joinpath(args.partition).joinpath(folder)
        # conv_path = f'{str(ROOT_PATH)}/{args.write_folder}/{args.partition}/{folder}'
        Path(conv_path).mkdir(parents=True, exist_ok=True)

        # write conversation
        with open(f'{conv_path}/{file}', 'w', encoding='utf8') as json_file:
            json.dump(conversation_simple, json_file, ensure_ascii=False, indent=4)

        conv_counter += 1