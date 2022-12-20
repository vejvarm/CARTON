import json
import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from lab.expand_csqa import CSQAInsertBuilder
from lab.qa2d import get_model
from action_executor.actions import ESActionOperator
from helpers import connect_to_elasticsearch, setup_logger
from constants import ROOT_PATH, QA2DModelChoices, RepresentEntityLabelAs
from args import parse_and_get_args
args = parse_and_get_args()

CLIENT = connect_to_elasticsearch()
LOGGER = setup_logger(__name__, loglevel=logging.INFO)


def compare_generated_utterances(model_choices: list[QA2DModelChoices] or QA2DModelChoices,
                                 labels_as_list: list[RepresentEntityLabelAs] or RepresentEntityLabelAs):
    data_folder = Path(f'{ROOT_PATH}{args.read_folder}/{args.partition}')
    # csqa_files = data_folder.glob('**/QA*.json')
    LOGGER.info(f'Reading folders for partition {args.partition}')

    op = ESActionOperator(CLIENT)

    results = {}

    for model_choice in model_choices:
        transformer = get_model(model_choice)
        builder = CSQAInsertBuilder(op, transformer)

        results[model_choice.name] = {}

        csqa_files = data_folder.glob('**/d_dataset_like_example_file.json')
        for pth in csqa_files:
            results[model_choice.name][pth.parent.name] = {}

            with open(pth, encoding='utf8') as json_file:
                conversation = json.load(json_file)

            for i in tqdm(range(len(conversation) // 2)):
                entry_user = conversation[2 * i]  # USER
                entry_system = conversation[2 * i + 1]  # SYSTEM

                if 'Simple' not in entry_user['question-type']:
                    continue

                if entry_user['question-type'] not in results[model_choice.name][pth.parent.name].keys():
                    results[model_choice.name][pth.parent.name][entry_user['question-type']] = {}

                if entry_user['description'] not in results[model_choice.name][pth.parent.name][entry_user['question-type']].keys():
                    results[model_choice.name][pth.parent.name][entry_user['question-type']][entry_user['description']] = {
                        'utterances': (entry_user['utterance'], entry_system['utterance']),
                        'entities': (entry_user['entities_in_utterance'], entry_system['entities_in_utterance'])
                    }

                # 2) TRANSFORM utterances to statements
                for labels_as in labels_as_list:
                    statement = builder.transorm_utterances(entry_user, entry_system, labels_as=labels_as)
                    LOGGER.info(f'statement: {statement}')
                    LOGGER.info(f"".center(50, "-") + "\n\n")

                    if labels_as.name not in results[model_choice.name][pth.parent.name][entry_user['question-type']][entry_user['description']]:
                        results[model_choice.name][pth.parent.name][entry_user['question-type']][entry_user['description']][labels_as.name] = {}

                    results[model_choice.name][pth.parent.name][entry_user['question-type']][entry_user['description']][labels_as.name] = {
                        'statement': statement
                    }

        json.dump(results, data_folder.joinpath('utterance_comparison.json').open('w', encoding='utf8'), indent=4, ensure_ascii=False)


def make_question_specific_csv_from_utterance_comparison2():
    data_folder = Path(f'{ROOT_PATH}{args.read_folder}/{args.partition}')
    # csqa_files = data_folder.glob('**/QA*.json')
    utterance_file = data_folder.joinpath('utterance_comparison.json')
    LOGGER.info(f'Reading file {utterance_file.name} for partition {args.partition}')

    d = json.load(utterance_file.open('r', encoding='utf8'))

    label_repr_names = [l.name for l in RepresentEntityLabelAs]

    folder = "original"
    model_names = ['T5-small', 'T5-base', 'T5-3B', 'T5-WH&YN']

    qa2dt5small_data = d[QA2DModelChoices.T5_SMALL.name][folder]
    qa2dt5_data = d[QA2DModelChoices.T5_BASE.name][folder]
    qc3b_data = d[QA2DModelChoices.T5_3B.name][folder]
    ghasqa2d_data = d[QA2DModelChoices.T5_WHYN.name][folder]

    for q_type in qc3b_data.keys():

        utterances = {}

        for q_subtype in qc3b_data[q_type].keys():
            index = ['original']
            utterances['model'] = ['-', ]
            utterances[q_subtype] = [' '.join(qa2dt5small_data[q_type][q_subtype]['utterances']), ]

            for labels_as_name in qc3b_data[q_type][q_subtype].keys():
                if labels_as_name not in label_repr_names:
                    continue
                index.extend([labels_as_name]*len(model_names))
                utterances['model'].extend(model_names)
                utterances[q_subtype].append(qa2dt5small_data[q_type][q_subtype][labels_as_name]['statement'])
                utterances[q_subtype].append(qa2dt5_data[q_type][q_subtype][labels_as_name]['statement'])
                utterances[q_subtype].append(qc3b_data[q_type][q_subtype][labels_as_name]['statement'])
                utterances[q_subtype].append(ghasqa2d_data[q_type][q_subtype][labels_as_name]['statement'])

        df = pd.DataFrame(data=utterances,
                          index=[index])
        df.to_csv(data_folder.joinpath(f'f{q_type}.csv'))


if __name__ == '__main__':
    # options
    args.read_folder = '/data'  # 'folder to read conversations'
    args.partition = ''  # 'train', 'test', 'val', ''

    model_choices = QA2DModelChoices
    labels_as_list = RepresentEntityLabelAs
    compare_generated_utterances(model_choices, labels_as_list)

    # make_question_specific_csv_from_utterance_comparison2()