import json
import logging
import re
from pathlib import Path

from constants import args, ROOT_PATH, ENTITY, TYPE, RELATION


from action_executor.actions import ESActionOperator
from helpers import connect_to_elasticsearch, setup_logger

CLIENT = connect_to_elasticsearch()
LOGGER = setup_logger(__name__, loglevel=logging.DEBUG)

def decode_active_set(active_set: list[str]):
    decoded_list = []
    for entry in active_set:
        category = [ENTITY, RELATION, ENTITY]
        # entry is either '(Q, P, c(Q))' or '(c(Q), P, Q)'
        sub, rel, obj = entry[1:-1].split(',')

        if sub.startswith('c'):
            sub = sub[2:-1]
            category[0] = TYPE

        if obj.startswith('c'):
            obj = obj[2:-1]
            category[-1] = TYPE

        decoded_list.append(list(zip([sub, rel, obj], category)))

    return decoded_list


def fill_active_set_with_entities(active_set: list[str], entities: list[str], op: ESActionOperator):
    decoded_list = []
    pattern = r'c\(.+?\)'
    d_type2ent = {}
    for ent in entities:
        for tp in op.get_types(ent):
            if tp not in d_type2ent.keys():
                d_type2ent[tp] = [ent]
            else:
                d_type2ent[tp].append(ent)

    LOGGER.debug(f'd_type2ent in fill_active_set_with_entities: {d_type2ent}')

    for entry in active_set:
        LOGGER.debug(f'entry in fill_active_set_with_entities: {entry}')
        for tp, ent_list in d_type2ent.items():
            entry = re.sub(rf'c\({tp}\)', ent_list.pop(), entry, 1)  # ONLY ONE REPLACEMENT MADE per entity
            LOGGER.debug(f'entry (out) in fill_active_set_with_entities: {entry}')

        decoded_list.append([entry])

    return decoded_list


def ver3(user: dict[list[str] or str], system: dict[list[str] or str], op: ESActionOperator):
    entities = system['entities_in_utterance']
    rdfs = []
    for active_set in system['active_set']:
        sub, rel, obj = active_set[1:-1].split(',')
        if sub.startswith('c('):
            s_tp = sub[2:-1]
            rdfs.extend([(e, rel, obj) for e in op.filter_type(entities, s_tp)])
        elif obj.startswith('c('):
            o_tp = obj[2:-1]
            rdfs.extend([(sub, rel, e) for e in op.filter_type(entities, o_tp)])

    rdfs = set(rdfs)

    new_active_set = []
    for rdf in rdfs:
        s, r, o = rdf
        new_active_set.append(f'i({s},{r},{o})')

    return new_active_set


def ver4(user: dict[list[str] or str], system: dict[list[str] or str], op: ESActionOperator):
    user_ents = user['entities_in_utterance']
    system_ents = system['entities_in_utterance']
    rels = user['relations']

    active_set = []
    _all_possible_ids = []
    for ue in user_ents:
        for se in system_ents:
            for r in rels:
                _all_possible_ids.extend([f'{ue}{r}{se}', f'{se}{r}{ue}'])

    for _id in _all_possible_ids:
        rdf = op.get_rdf(_id)
        if rdf:
            active_set.append(f"i({rdf['sid']}{rdf['rid']}{rdf['oid']})")

    return active_set


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

                # new_active_set = fill_active_set_with_entities(entry_system['active_set'], entry_system['entities_in_utterance'], op)
                new_active_set = ver4(entry_user, entry_system, op)
                LOGGER.debug(f'new_active_set in {__name__}: {new_active_set}')
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
