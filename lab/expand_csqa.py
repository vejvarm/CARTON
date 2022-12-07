import json
from pathlib import Path

from constants import args, ROOT_PATH, ENTITY, TYPE, RELATION

from action_executor.actions import ESActionOperator
from helpers import connect_to_elasticsearch

CLIENT = connect_to_elasticsearch()


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
                print(f"USER: {entry_user['question-type']}, {entry_user['entities_in_utterance']}, {entry_user['utterance']}")
                print(f"SYSTEM: {entry_system['entities_in_utterance']}, {entry_system['active_set']} {entry_system['utterance']}")

                active_set_decoded = decode_active_set(entry_system['active_set'])
                print(active_set_decoded)

                new_active_set = []
                for entry in active_set_decoded:
                    sub, sub_cat = entry[0]
                    rel, rel_cat = entry[1]
                    obj, obj_cat = entry[2]

                    if sub_cat == TYPE and obj_cat == TYPE:
                        raise NotImplementedError('Both subject and object are of category TYPE.')

                    if sub_cat == TYPE:
                        subs = set(op.filter_type(op.find_reverse(obj, rel), sub)).intersection(entry_system['entities_in_utterance'])
                        new_active_set.append([f"i({e},{rel},{obj})" for e in subs])

                    if obj_cat == TYPE:
                        objs = set(op.filter_type(op.find(sub, rel), obj)).intersection(entry_system['entities_in_utterance'])
                        new_active_set.append([f"i({sub},{rel},{e})" for e in objs])

                print(new_active_set)
                print(f"".center(50, "-"), end='\n\n')

            # active_set
            # '(Q1, P1, c(Q2))' ... take all entities with type c(Q2) and cross reference them with entities in utterance
            # essentially
            # for each entity in filter_type(find(Q1, P1), Q2)
            # actually just for each entity E in sysyem['entities_in_utterance']: get(Q1, P1, E) where E replaces the c() part
            # if _id Q1P1E exists: append '(Q1, P1, E)' to active_set list



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
