import json
from pathlib import Path

from constants import ROOT_PATH
from args import parse_and_get_args
args = parse_and_get_args()

if __name__ == "__main__":
    # pop unneeded conversations right here?
    args.read_folder = '/data'  # 'folder to read conversations'
    args.partition = ''  # 'train', 'test', 'val', ''

    # read data and create csqa data dict
    # dict: partition -> folder -> file -> conversation
    data_folder = Path(f'{ROOT_PATH}{args.read_folder}/{args.partition}')
    csqa_files = data_folder.glob('**/d_example.json')
    print(f'Reading folders for partition {args.partition}')

    for pth in csqa_files:
        folder = pth.parent.name
        file = pth.name

        dataset_like_example_file = []

        with open(pth, encoding='utf8') as json_file:
            example_dict = json.load(json_file)

        for _, desc_dict in example_dict.items():
            for _, conv_turn in desc_dict.items():
                dataset_like_example_file.extend(conv_turn)

        with open(pth.parent.joinpath("d_dataset_like_example_file.json"), 'w', encoding='utf8') as f:
            json.dump(dataset_like_example_file, f, indent=4, ensure_ascii=False)