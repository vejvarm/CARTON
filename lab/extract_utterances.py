import json
import logging
from pathlib import Path
from unidecode import unidecode
from constants import args, ROOT_PATH

from helpers import connect_to_elasticsearch, setup_logger

LOGGER = setup_logger(__name__, logging.INFO)


def main():
    # read data and create csqa data dict
    # dict: partition -> folder -> file -> conversation
    data_folder = Path(f'{ROOT_PATH}{args.read_folder}/{args.partition}')
    csqa_files = data_folder.glob('**/QA*.json')
    # csqa_files = data_folder.glob('**/d_dataset_like_example_file.json')
    LOGGER.info(f'Reading folders for partition {args.partition}')

    conv_dict = {}

    num_convs_to_extract = 10

    for i, pth in enumerate(csqa_files):
        if i >= num_convs_to_extract:
            break

        conv_list = []

        with open(pth, encoding='utf8') as json_file:
            conversation = json.load(json_file)

        for i in range(len(conversation) // 2):
            entry_user = conversation[2 * i]  # USER
            entry_system = conversation[2 * i + 1]  # SYSTEM

            conv_list.extend(['U: '+entry_user['utterance'], 'S: '+entry_system['utterance']])

        conv_dict[f"{pth.parent.name}/{pth.name}"] = conv_list

    print(conv_dict)
    json.dump(conv_dict, data_folder.joinpath('conversations.json').open('w', encoding='utf8'), indent=4)


if __name__ == '__main__':
    # options
    args.read_folder = '/data'  # 'folder to read conversations'
    args.partition = ''  # 'train', 'test', 'val', ''

    main()