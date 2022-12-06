import json
import pandas

from glob import glob
from constants import args, ROOT_PATH

from os import path
from collections import Counter

from matplotlib import pyplot as plt

if __name__ == '__main__':

    args.read_folder = '/data/original' # 'folder to read conversations'
    args.partition = ''  # 'train', 'test', 'val'

    # read data and create csqa data dict
    # dict: partition -> folder -> file -> conversation
    data_folder = f'{ROOT_PATH}{args.read_folder}/{args.partition}'
    csqa_files = glob(f'{data_folder}/*/*.json')
    print(f'Reading folders for partition {args.partition}')

    d_qt_desc = dict()

    c_question_type = Counter()
    c_description = Counter()
    d_example = dict()

    for pth in csqa_files:
        pth = path.normpath(pth)
        folder = pth.rsplit('\\', 1)[0].rsplit('\\', 1)[-1]
        file = pth.rsplit('\\', 1)[-1]

        with open(pth) as json_file:
            data_list = json.load(json_file)

        for i in range(len(data_list)//2):
            entry_user = data_list[2*i]  # USER
            entry_system = data_list[2*i + 1]  # SYSTEM

            if 'description' not in entry_user.keys():
                continue

            question_type = entry_user['question-type']
            description = entry_user['description']

            # entry['speaker'] == 'USER':
            if question_type not in d_qt_desc.keys():
                d_qt_desc[question_type] = Counter()
                d_example[question_type] = {}

            # capture first occurrence of each unique question-type+description combo as an example
            if description not in d_example[question_type]:
                d_example[question_type][description] = (entry_user, entry_system)

            c_question_type.update([question_type])
            c_description.update([description])
            d_qt_desc[question_type].update([description])
    print(f'Found {len(d_qt_desc.keys())} unique question-types and {len(c_description.keys())} uniqe descriptions.')

    print(f'Creating and saving contingecy table with Qtype/description:')
    df = pandas.DataFrame.from_dict(d_qt_desc)
    df.to_csv(f'{data_folder}\\qt_desc_contingency.csv')

    print(f'Plotting contingency table as bar plot and saving to {data_folder}')
    df.plot(kind="bar", stacked=True, legend=True)

    print(f'Saving d_example.json')
    with open(f"{data_folder}\\d_example.json", 'w', encoding='utf8') as f:
        json.dump(d_example, f, indent=4, ensure_ascii=False)

    plt.savefig(f'{data_folder}\\qt_desc_contingency.pdf')
    plt.savefig(f'{data_folder}\\qt_desc_contingency.png', dpi=200)

    plt.show()