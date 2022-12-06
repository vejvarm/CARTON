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

    for pth in csqa_files:
        pth = path.normpath(pth)
        folder = pth.rsplit('\\', 1)[0].rsplit('\\', 1)[-1]
        file = pth.rsplit('\\', 1)[-1]

        with open(pth) as json_file:
            data_list = json.load(json_file)

        for entry in data_list:
            if entry['speaker'] == 'SYSTEM':
                continue

            # entry['speaker'] == 'USER':
            if entry['question-type'] not in d_qt_desc.keys():
                d_qt_desc[entry['question-type']] = Counter()

            c_question_type.update([entry['question-type']])
            if 'description' in entry.keys():
                c_description.update([entry['description']])
                d_qt_desc[entry['question-type']].update([entry['description']])
    print(f'Found {len(d_qt_desc.keys())} unique question-types and {len(c_description.keys())} uniqe descriptions.')

    print(f'Creating and saving contingecy table with Qtype/description:')
    df = pandas.DataFrame.from_dict(d_qt_desc)
    df.to_csv(f'{data_folder}\\qt_desc_contingency.csv')

    print(f'Plotting contingency table as bar plot and saving to {data_folder}')
    df.plot(kind="bar", stacked=True, legend=True)

    plt.savefig(f'{data_folder}\\qt_desc_contingency.pdf')
    plt.savefig(f'{data_folder}\\qt_desc_contingency.png', dpi=200)

    plt.show()