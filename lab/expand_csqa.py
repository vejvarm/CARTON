import json
import logging
import re
from pathlib import Path

from constants import args, ROOT_PATH, ENTITY, TYPE, RELATION


from action_executor.actions import ESActionOperator
from helpers import connect_to_elasticsearch, setup_logger

CLIENT = connect_to_elasticsearch()
LOGGER = setup_logger(__name__, loglevel=logging.INFO)


class CSQAInsertBuilder:

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

        # TODO: for Coreferecnce and Ellipsis, we must keep the previous questions in the dataset!!!

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

    def _replace_labels_with_id(self, utterance: str, entities: list[str]) -> tuple[str, dict]:
        inverse_map = dict()
        for ent in entities:
            label = self.op.get_label(ent)
            utterance = utterance.replace(label, ent)
            inverse_map[ent] = label

        return utterance, inverse_map

    def transorm_utterances(self, user: dict[list[str] or str], system: dict[list[str] or str]):
        user_utterance = user['utterance']
        user_ents = user['entities_in_utterance']
        system_utterance = system['utterance']
        system_ents = system['entities_in_utterance']

        # replace all labels with entity ids
        new_user_utterance, user_inverse_map = self._replace_labels_with_id(user_utterance, user_ents)
        new_system_utterance, system_inverse_map = self._replace_labels_with_id(system_utterance, system_ents)

        # Tranform user+system utterances into declarative statements using T5-QA2D
        # TODO: use T5-QA2D --- THIS IS JUST A PLACEHOLDER
        statement = f'{new_user_utterance} {new_system_utterance}'

        # replace entity ids back with labels
        for eid, lab in {**user_inverse_map, **system_inverse_map}.items():
            statement = statement.replace(eid, lab)

        return statement

    def transform_fields(self):
        pass


if __name__ == "__main__":
    # pop unneeded conversations right here?
    args.read_folder = '/data'  # 'folder to read conversations'
    args.partition = ''  # 'train', 'test', 'val', ''

    # read data and create csqa data dict
    # dict: partition -> folder -> file -> conversation
    data_folder = Path(f'{ROOT_PATH}{args.read_folder}/{args.partition}')
    # csqa_files = data_folder.glob('**/QA*.json')
    csqa_files = data_folder.glob('**/d_dataset_like_example_file.json')
    print(f'Reading folders for partition {args.partition}')

    # TODO: only take Simple Question types. Those we will need to transform (how about the questions around them?)

    op = ESActionOperator(CLIENT)
    builder = CSQAInsertBuilder(op)

    for pth in csqa_files:
        folder = pth.parent.name
        file = pth.name

        with open(pth, encoding='utf8') as json_file:
            conversation = json.load(json_file)

        for i in range(len(conversation)//2):
            entry_user = conversation[2*i]  # USER
            entry_system = conversation[2*i + 1]  # SYSTEM

            if 'Simple' in entry_user['question-type']:
                print(f"USER: {entry_user['question-type']}, {entry_user['entities_in_utterance']}, {entry_user['relations']}, {entry_user['utterance']}")
                print(f"SYSTEM: {entry_system['entities_in_utterance']} {entry_system['active_set']} {entry_system['utterance']}")

                # 1) TRANSFORM active_set field
                new_active_set = builder.build_active_set(entry_user, entry_system)
                print(f'new_active_set in {__name__}: {new_active_set}')

                # 2) TRANSFORM utterances to statements
                statement = builder.transorm_utterances(entry_user, entry_system)
                print(f'statement in {__name__}: {statement}')
                print(f"".center(50, "-"), end='\n\n')

            # conversation types to tweak:
            # and how?

            # TODO: Simple (Direct)
            #   Simple Question
            #   Simple Question|Single Entity
            #   Simple Question|Mult. Entity|Indirect

            # TODO: Simple (Coreference)
            #   Simple Question|Single Entity|Indirect
            #   Simple Question|Mult. Entity

            # TODO: Simple (Ellipsis)
            #   only subject is changed, parent and predicate remains same
            #   Incomplete|object parent is changed, subject and predicate remain same
