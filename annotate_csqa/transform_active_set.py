import json
import logging
import re

from pathlib import Path
from tqdm import tqdm
from action_executor.actions import ESActionOperator
from helpers import connect_to_elasticsearch, setup_logger
from constants import ROOT_PATH
from args import parse_and_get_args
args = parse_and_get_args()

CLIENT = connect_to_elasticsearch()
LOGGER = setup_logger(__name__, loglevel=logging.WARNING)

# NOTE:
#   1. annotate_position.py
#   2. extract_simple.py
#   3. transform_active_set.py


class ActiveSetTransformer:

    def __init__(self, operator: ESActionOperator):
        self.op = operator

    def build_active_set(self, user: dict[list[str] or str], system: dict[list[str] or str]):
        user_ents = user['entities_in_utterance']
        system_ents = system['entities_in_utterance']
        rels = user['relations']

        # in case of Ellipsis question
        if not rels:
            rels = set()
            rels.update(*[re.findall(r'P\d+', entry) for entry in system['active_set']])
            rels = list(rels)

        active_set = []
        _all_possible_ids = []
        for ue in user_ents:
            for se in system_ents:
                for r in rels:
                    _all_possible_ids.extend([f'{ue}{r}{se}', f'{se}{r}{ue}'])

        for _id in _all_possible_ids:
            rdf = self.op.get_rdf(_id)
            if rdf:
                active_set.append(f"i({rdf['sid']},{rdf['rid']},{rdf['oid']})")

        return active_set

    @staticmethod
    def build_table(active_set: list[str]):
        tab_dict = {}
        for entry in active_set:
            s, r, o = entry[2:-1].split(",")
            if (s, r) not in tab_dict.keys():
                tab_dict[(s, r)] = [o, ]
            else:
                tab_dict[(s, r)].append(o)

        return tab_dict

    def tab_dict_to_jsonl(self, tab_dict: dict[tuple[str]: list[str]]):
        line = []
        for (subj, rel), objs in tab_dict.items():
            obs = [self.op.get_entity_label(o).split(" ") for o in objs]
            line.extend([["name", self.op.get_entity_label(subj).split(" ")],
                         [self.op.get_relation_label(rel), [o for ob in obs for o in ob]]])

        LOGGER.debug(f"line: {line}")
        return line


def transform_active_set(data_folder: Path):
    csqa_files = data_folder.glob('**/*.json')

    op = ESActionOperator(CLIENT)
    actset_builder = ActiveSetTransformer(op)

    for pth in tqdm(csqa_files):
        with pth.open("r", encoding="utf8") as json_file:
            conversation = json.load(json_file)

        for i in range(0, len(conversation), 2):
            user = conversation[i]
            system = conversation[i+1]
            LOGGER.info("\n")
            LOGGER.debug(f"turn {user['turn_position']} in json_file: {pth.parent.name}/{pth.name}")
            new_active_set = actset_builder.build_active_set(user, system)
            LOGGER.info(f"active_set: {new_active_set}")

            tab_dict = actset_builder.build_table(new_active_set)
            LOGGER.info(f"tab_dict: {tab_dict}")

            table = actset_builder.tab_dict_to_jsonl(tab_dict)

            system["insert_active_set"] = new_active_set
            system["table_format"] = table

        with pth.open("w", encoding="utf8") as json_file:
            json.dump(conversation, json_file, indent=4)


if __name__ == "__main__":
    partitions = {"test", "train", "val"}
    read_folder = "data/final_simple/csqa"

    for partition in partitions:
        data_folder = ROOT_PATH.joinpath(read_folder).joinpath(partition)

        print(f'Reading folders for partition {partition}')
        transform_active_set(data_folder)
