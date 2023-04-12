import logging
import json
import ujson
import os.path
import sqlite3
import numpy as np
from typing import Sequence, Union
from ordered_set import OrderedSet
from matplotlib import pyplot as plt
from elasticsearch import Elasticsearch

from constants import ROOT_PATH
from args import parse_and_get_args
args = parse_and_get_args()


def extract_individual_losses_from_train_log(file_name):
    with open(f"{ROOT_PATH}/{args.path_results}/{file_name}", 'r') as f:
        col_names = []
        out = []
        out_dict = {}
        epoch = 0
        for line in f:
            if 'Val losses::' in line:
                if not col_names:
                    col_names = ['EPOCH'] + [e.split(":")[-2] for e in line.split("|")]
                    out.append(col_names)
                values = [epoch + 1] + [float(e.split(":")[-1]) for e in line.split("|")]
                out.append(values)
                epoch += 1

        for atribute in zip(*out):
            out_dict[atribute[0]] = atribute[1:]

    # save to json file
    with open(f"{args.path_results}/out_{file_name.split('.')[0]}.json", 'w') as f:
        json.dump(out_dict, f, indent=4)


def extract_val_loss_from_train_log(file_name):
    with open(f"{ROOT_PATH}/{args.path_results}/{file_name}", 'r') as f:
        out = []
        epoch = 0
        for line in f:
            if "Val loss:" in line:
                line_split = line.split("Val loss: ")
                out.append(float(line_split[-1]))
                epoch += 1

        arr = np.array(out)

    with open(f"{args.path_results}/out_{file_name.split('.')[0]}.npy", 'wb') as f:
        np.save(f, arr)


def plot_from_jsom_multiloss_file(json_file_name, title="CrossEntropy losses during validation"):
    with open(f"{args.path_results}/{json_file_name}", 'r') as f:
        data = json.load(f)

    x = data['EPOCH']

    fig, ax = plt.subplots(len(data) - 1, 1)
    ax[0].set_title(title)
    for i, (attr, vals) in enumerate(data.items()):
        if 'EPOCH' in attr.upper():
            pass
        else:
            ax[i-1].plot(x, vals, label=attr)

            # ax[i-1].set_title(attr)
            ax[i-1].set_ylabel(attr)
    # plt.tight_layout()
    plt.xlabel('epoch')
    plt.show()


def plot_from_multi_json_files(json_file_names: list[str], labels: list[str] = tuple(),
                               title="CrossEntropy losses during validation",
                               plot_file_name: str = 'individual_losses',
                               plot_file_types: list[str] = ('png', 'pdf', )):
    data_dict = {}
    num_fields = 0
    colors = {}
    color_list = plt.rcParams['axes.prop_cycle'].by_key()['color']
    if labels:
        assert len(labels) == len(json_file_names), 'Length of labels must be same as length of log_file_names'
    else:
        labels = [file_name.split('.')[0].split('_')[-1] for file_name in json_file_names]

    field_names = set()
    for i, (file_name, label) in enumerate(zip(json_file_names, labels)):
        label = 'base' if label == 'multitask' else label
        with open(f"{args.path_results}/{file_name}", 'r') as f:
            data = json.load(f)
            for j, (attr, vals) in enumerate(data.items()):
                if attr not in data_dict.keys():
                    data_dict[attr] = {label: vals}
                else:
                    data_dict[attr][label] = vals
        colors[label] = color_list[i]

    fig, ax = plt.subplots(len(data_dict)-1, 1, figsize=(16, 16))
    ax[0].set_title(title)

    font = {
        'size': 15
    }

    plt.rc('font', **font)

    x = data_dict['EPOCH']
    min_x = 100000
    max_x = 0
    for i, (attr, vals_dict) in enumerate(data_dict.items()):
        if 'EPOCH' in attr.upper():
            pass
        else:
            for k, vals in vals_dict.items():
                ax[i-1].plot(x[k], vals, label=f'{k}', color=colors[k])
                min_x = min(x[k][0], min_x)
                max_x = max(len(x[k]), max_x)

            # ax[i-1].set_title(attr)
            ax[i-1].set_ylabel(attr)
            ax[i-1].set_xlim(min_x, max_x)
    ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
              ncol=3, fancybox=True, shadow=True)
    plt.xlabel('epoch')

    for ft in plot_file_types:
        output_plot_file_path = f"{args.path_results}/{plot_file_name}.{ft}"
        plt.savefig(output_plot_file_path, bbox_inches='tight', format=ft, dpi=200)


def plot_from_np_file(path_to_np_file, label="validation loss"):
    arr = np.load(path_to_np_file)
    plt.plot(arr, label=label)
    plt.tight_layout()
    plt.title("Training validation loss")
    plt.xlabel("epoch")
    plt.ylabel("validation loss")


def enforce_question_type(d, question_type):
    """ Check if the right question type is being Evaluated
    Clarification gets special treatment here!
    Inputs:
    arg d (dict): dictionary of the input data

    return (bool): True if question_type corresponds to current args.question_type
    """
    q_type_list = list()
    if isinstance(d['question_type'], list) and 'Clarification' in d['question_type']:
        q_type_list = d['question_type']
        d['question_type'] = 'Clarification'
    elif not isinstance(d['question_type'], list):
        q_type_list.append(d['question_type'])
    else:
        raise ValueError(f"Unsupported question type: {d['question_type']}")

    # check if question type is correct:
    if question_type in q_type_list:
        return True
    else:
        return False


def example_sqlite():
    subject = "Q7931"
    con = sqlite3.connect('\data\example.db')  # initialize and load database file
    cur = con.cursor()  # cursor for using SQL commands

    # create table, insert data
    cur.execute("CREATE TABLE sub_rel_ob (sub_id, rel_id, ob_id)")
    cur.execute(f"INSERT INTO sub_rel_ob ({subject}, 'P831', 'Q8401')")

    # find subject entries
    cur.execute(f"SELECT * FROM table WHERE subject = {subject}")

    con.close()  # close connection to database file


def out_file_name(input_string):
    return "out_" + os.path.splitext(input_string)[0] + ".npy"


# Inference ner edit distance
def get_edit_distance(query=None, confidence_levels=True, default_dist=1):
    """
    Return the matching maximum edit distance for a particular string length.
    As string length decreases, maximum edit distance also decreases.
    """

    # Check for appropriate formats
    assert isinstance(query, str), "queries can be str() type only"

    # We check if confidence levels are set
    if confidence_levels:
        num_chars = len(query)
        max_len = len(args.ner_max_distance)
        if num_chars <= max_len:
            max_dist = args.ner_max_distance[num_chars-1]
        else:
            max_dist = args.ner_max_distance[-1]
    # Fallback if confidence levels are not set
    else:
        max_dist = default_dist

    return int(max_dist)


def main_old(log_files: list[str], labels: list[str] = tuple(),
             plot_file_name: str = 'training_val_loss',
             plot_file_types: list[str] = ('png', 'pdf', )):

    fig = plt.figure()

    font = {  # 'family': 'normal',
        # 'weight': 'bold',
        'size': 15}

    plt.rc('font', **font)


    if labels:
        assert len(labels) == len(log_files), 'list of labels must be same length as list of log files'
    else:
        labels = [f'plot{i}' for i in range(len(log_files))]

    for lf, label in zip(log_files, labels):

        extract_val_loss_from_train_log(lf)

        input0 = f"{args.path_results}/{out_file_name(lf)}"

        plot_from_np_file(input0, label)

    plt.legend()

    for ft in plot_file_types:
        output_plot_file_path = f"{args.path_results}/{plot_file_name}.{ft}"
        plt.savefig(output_plot_file_path, bbox_inches='tight', format=ft, dpi=200)


def crop_json_file(path_to_json_file: str, num_entries=10000):

    full_dict = ujson.loads(open(path_to_json_file).read())

    keys = list(full_dict.keys())[:num_entries]

    ujson.dump({k: full_dict[k] for k in keys},
               open(f'{path_to_json_file.removesuffix(".json")}_first_{num_entries}.json', 'w'),
               indent=4, escape_forward_slashes=False)


def setup_logger(name=__name__, loglevel=logging.WARNING, handlers=(logging.StreamHandler(),)):
    logging.basicConfig(format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%d/%m/%Y %I:%M:%S %p',
                        level=loglevel,
                        handlers=handlers)

    # disable PUT INFO responses from ElasticSearch search command
    logging.getLogger('elastic_transport.transport').setLevel(logging.WARNING)

    return logging.getLogger(name)


def connect_to_elasticsearch(user=args.elastic_user, password=args.elastic_password):
    """Connect to Elasticsearch client using urls and credentials from args.py"""
    return Elasticsearch(
        args.elastic_host,
        ca_certs=f'{ROOT_PATH}/{args.elastic_certs.removeprefix("./")}',
        basic_auth=(user, password),  # refer to args.py --elastic_password for alternatives
        retry_on_timeout=True,
    )


def _uppercase_sequence(sequence: Union[Sequence[str], OrderedSet[str]], tp):
    if not isinstance(sequence, tp):
        return sequence

    new_sequence = []
    for ent in sequence:
        try:
            new_sequence.append(ent.upper())
        except AttributeError:
            raise AttributeError("Entity in sequence is not a string!")
    return tp(new_sequence)


def uppercase(f):

    def wrap(entry, *args, **kwargs):
        if isinstance(entry, str):
            entry = entry.upper()
        else:
            entry = _uppercase_sequence(entry, OrderedSet)
            entry = _uppercase_sequence(entry, list)
            entry = _uppercase_sequence(entry, tuple)

        return f(entry, *args, **kwargs)

    return wrap


if __name__ == '__main__':
    # log_files = ["train_multitask_original2.log", "train_multitask_LASAGNE.log", "train_multitask_CwNERwLinPtr2.log"]
    # labels = ['CARTON', 'LASAGNE', 'CARTONwNERwLinPtr']
    # main_old(log_files, labels)
    #
    # json_log_files = []
    # for log_file in log_files:
    #     extract_individual_losses_from_train_log(log_file)
    #
    #     json_log_files.append('out_' + log_file.split('.')[0] + '.json')
    #     # plot_from_jsom_multiloss_file(json_log_files[-1])
    #
    # # plot collectively:
    # plot_from_multi_json_files(json_log_files, labels)
    #
    # plt.show()

    # cropping for testing purposes
    num_entries = 10000
    path_to_rdf_index = 'knowledge_graph/index_rdf_dict.json'
    crop_json_file(path_to_rdf_index, num_entries)

    path_to_ent_index = 'knowledge_graph/index_ent_dict.json'
    crop_json_file(path_to_ent_index, num_entries)
