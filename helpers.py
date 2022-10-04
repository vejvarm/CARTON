import os.path
import sqlite3
import numpy as np
from matplotlib import pyplot as plt

from constants import *


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


if __name__ == '__main__':
    log_file_default = "train_multitask.log"
    log_file_orig = "train_multitask_original_code_params.log"
    log_file_paper = "train_multitask_paper_params.log"
    log_file_latest = "train_multitask_latest-params.log"
    # log_file_latest = "train_multitask_latest-params-short.log"

    plot_file_name = 'training-val_loss_original-vs-2layer-paper'
    plot_file_type = 'png'

    extract_val_loss_from_train_log(log_file_default)
    # extract_val_loss_from_train_log(log_file_orig)
    # extract_val_loss_from_train_log(log_file_paper)
    # extract_val_loss_from_train_log(log_file_latest)

    font = {#'family': 'normal',
            #'weight': 'bold',
            'size': 15}

    plt.rc('font', **font)

    input0 = f"{args.path_results}/{out_file_name(log_file_default)}"
    input1 = f"{args.path_results}/{out_file_name(log_file_orig)}"
    input2 = f"{args.path_results}/{out_file_name(log_file_paper)}"
    input3 = f"{args.path_results}/{out_file_name(log_file_latest)}"
    output_plot_file_path = f"{args.path_results}/{plot_file_name}.{plot_file_type}"
    plot_from_np_file(input0, "pf_dim = 4x emb_dim")
    plot_from_np_file(input1, "original code parameters")
#    plot_from_np_file(input2, "paper parameters")
#    plot_from_np_file(input3, "paper parameters (only 2 layers)")
    plt.legend()
    plt.show()

    plt.savefig(output_plot_file_path, format=plot_file_type, dpi=200)

