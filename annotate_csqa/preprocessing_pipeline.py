# Process to generate SQ-CSD (Simple Question-CSDeclarative):
# 0. csqa_v9
# 1. simple_annotators.py --> csqa_w_turn_pos (user for ner ...)
# 2. extract_simple.py --> csqa
# 3. active_set_annotator.py
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
from constants import ROOT_PATH
from simple_annotators import clean_up_qa_entries, annotate_position, extract_simple
from active_set_annotator import transform_active_set

# add arguments to parser
parser = argparse.ArgumentParser(description='Annotate dataset with "turn_position"')
parser.add_argument('--read_folder', default='data/original', help='folder to read conversations (relative to ROOT)')
parser.add_argument('--annot_write_folder', default='data/original_w_turn_pos', help='folder to write the turn-annotated dataset (relative to ROOT)')
parser.add_argument('--sqcsqa_write_folder', default='data/simple_direct', help='folder to write the simple question part of csqa dataset (relative to ROOT)')
args = parser.parse_args()


# read data and create csqa data dict
# dict: partition -> folder -> file -> conversation

if __name__ == "__main__":
    # GENERATE cleaned up and turn annotated and table_format and insert_active set annotated SQ (Direct)-CSQA set
    read_folder = ROOT_PATH.joinpath(args.read_folder)
    annot_write_folder = ROOT_PATH.joinpath(args.annot_write_folder)
    sqcsqa_write_folder = ROOT_PATH.joinpath(args.sqcsqa_write_folder)
    partitions = ("test", "train", "val")
    for partition in partitions:
        print(f'Processing partition: "{partition}"')
        part_read_folder = read_folder.joinpath(partition)
        part_annot_write_folder = annot_write_folder.joinpath(partition)
        part_sqcsqa_write_folder = sqcsqa_write_folder.joinpath(partition)

        print(f'Cleaning up utterances in conversations in {part_read_folder.parent}/{part_read_folder.name}...')
        clean_up_qa_entries(part_read_folder, part_annot_write_folder)
        print(f'\t new dataset saved to {part_annot_write_folder.parent}/{part_annot_write_folder.name}.')

        print(f'Annotating {part_annot_write_folder.parent}/{part_annot_write_folder.name} with turn_position...')
        annotate_position(part_annot_write_folder, part_annot_write_folder)
        print(f'\t new dataset saved to {part_annot_write_folder.parent}/{part_annot_write_folder.name}.')

        # TODO: expand for Coreferenced and Ellipsis
        print(f'Extracting Simple Question (Direct) types from {part_annot_write_folder.parent}/{part_annot_write_folder.name}...')
        extract_simple(part_annot_write_folder, part_sqcsqa_write_folder)
        print(f'\t new dataset saved to {part_sqcsqa_write_folder.parent}/{part_sqcsqa_write_folder.name}.')

        print(f'Transforming active_set to insert_active_set and table_format in {part_sqcsqa_write_folder.parent}/{part_sqcsqa_write_folder.name}.')
        transform_active_set(part_sqcsqa_write_folder)
        print(f'\t new dataset saved to {part_sqcsqa_write_folder.parent}/{part_sqcsqa_write_folder.name}.')


