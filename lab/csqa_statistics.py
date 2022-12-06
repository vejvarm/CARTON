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
    data_folder = path.normpath(f'{ROOT_PATH}{args.read_folder}/{args.partition}')
    csqa_files = glob(path.normpath(f'{data_folder}/*/*.json'))
    print(f'Reading folders for partition {args.partition}')

    d_qt_desc = dict()

    # description to unique index mapping
    path_to_desc2id_file = path.normpath(f"{ROOT_PATH}{args.read_folder}\\d_desc2id.json")
    try:
        with open(path_to_desc2id_file, 'r') as f:
            d_desc2id = json.load(f)
            unique_count = len(d_desc2id) - 1
    except FileNotFoundError:
        print(f"file '{path_to_desc2id_file}' doesn't exist. Creating empty dictionary.")
        d_desc2id = dict()
        unique_count = 0

    c_question_type = Counter()
    c_description = Counter()
    d_example = dict()
    df_example_utterances = pandas.DataFrame(columns=['Question Type', 'Description', 'USER', 'SYSTEM'])

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

            # capture all unique descriptions and assign them IDs
            if description not in d_desc2id.keys():
                d_desc2id[description] = unique_count
                unique_count += 1

            c_question_type.update([question_type])
            c_description.update([description])
            d_qt_desc[question_type].update([d_desc2id[description]])

            # capture first occurrence of each unique question-type+description combo as an example
            if description not in d_example[question_type]:
                d_example[question_type][description] = (entry_user, entry_system)
                df_row = pandas.DataFrame({'Question Type': question_type,
                                           'Description': description,
                                           'USER': entry_user['utterance'],
                                           'SYSTEM': entry_system['utterance']},
                                          index=[int(f"{entry_user['ques_type_id']}{d_desc2id[description]:02d}")])
                df_example_utterances = pandas.concat([df_example_utterances, df_row])

    print(f'Found {len(d_qt_desc.keys())} unique question-types and {len(c_description.keys())} uniqe descriptions.')

    print(f'Creating and saving contingecy table with Qtype/description:')
    df = pandas.DataFrame.from_dict(d_qt_desc)
    df.to_csv(f'{data_folder}\\qt_desc_contingency.csv')

    print(f'Plotting contingency table as bar plot and saving to {data_folder}')
    df.plot(kind="bar", stacked=True, legend=True, xlabel='description id', ylabel='number of entries')

    print(f'Saving d_example.json')
    with open(f"{data_folder}\\d_example.json", 'w', encoding='utf8') as f:
        json.dump(d_example, f, indent=4, ensure_ascii=False)

    print(f'Saving d_desc2id legend to d_desc2id.json')
    with open(path_to_desc2id_file, 'w', encoding='utf8') as f:
        json.dump(d_desc2id, f, indent=4, ensure_ascii=False)

    print(f'Saving df_example_utterances.csv')
    df_example_utterances.to_csv(f"{data_folder}\\df_example_utterances.csv")

    plt.savefig(f'{data_folder}\\qt_desc_contingency.pdf')
    plt.savefig(f'{data_folder}\\qt_desc_contingency.png', dpi=200)

    plt.show()