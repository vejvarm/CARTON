import json
import pathlib
import pickle
from collections import Counter
from glob import glob
from itertools import chain

from torchtext.vocab import Vocab
from transformers import BertTokenizer
from torchtext.data import Field, Example, Dataset
import torch

from constants import (LOGICAL_FORM, ROOT_PATH, QUESTION_TYPE, ENTITY, GOLD, LABEL, NA_TOKEN, SEP_TOKEN,
                       GOLD_ACTIONS, PAD_TOKEN, ACTION, RELATION, TYPE, PREV_ANSWER, VALUE, QUESTION,
                       CONTEXT_QUESTION, CONTEXT_ENTITIES, ANSWER, RESULTS, PREV_RESULTS, START_TOKEN, CTX_TOKEN,
                       UNK_TOKEN, END_TOKEN, INPUT, ID, NER, COREF, PREDICATE_POINTER, TYPE_POINTER, B, I, O)


class CSQADataset:

    def __init__(self, args):
        data_path_rel = args.data_path
        self.train_path = ROOT_PATH.joinpath(data_path_rel).joinpath("train")
        self.val_path = ROOT_PATH.joinpath(data_path_rel).joinpath("val")
        self.test_path = ROOT_PATH.joinpath(data_path_rel).joinpath("test")

        self.id = 0
        self.no_data_cache = args.no_data_cache
        self.no_vocab_cache = args.no_vocab_cache
        self.cache_path = pathlib.Path(args.cache_path)
        self.cache_path.mkdir(parents=True, exist_ok=True)
        self.data, self.helpers = self.preprocess_data()
        # self.load_data_and_fields(data, helpers)
        self.vocabs = self.build_vocabs(self.data)
        # print(self.vocabs[ID].itos[0])
        # print(f"{self.vocabs[INPUT].itos[10]}: {self.vocabs[INPUT].vectors[10]}")
        # print(f"{self.vocabs[COREF].itos[10]}: {self.vocabs[COREF].freqs}")
        print("done")
        # exit()

    def build_vocabs(self, data: dict[str: list], cache_subfolder="vocabs"):
        # data[split]
        # [0] ... ID
        # [1] ... INPUT
        # [2] ... LOGICAL_FORM
        # [3] ... NER
        # [4] ... COREF
        # [5] ... PREDICATE_POINTER
        # [6] ... TYPE_POINTER
        # [7] ... ENTITY

        cache_sub_path = self.cache_path.joinpath(cache_subfolder)
        cache_sub_path.mkdir(exist_ok=True, parents=True)

        # Build vocabularies for each field
        vocabs = dict()
        data_aggregate = []
        for split in data.values():
            data_aggregate.extend(split)
        print("Building vocabularies...")
        vocabs[ID] = self._build_vocab([item[0] for item in data_aggregate],
                                       specials=[],
                                       vocab_cache=cache_sub_path.joinpath("id_vocab.pkl"))
        vocabs[INPUT] = self._build_vocab([item[1] for item in data_aggregate],
                                          specials=[NA_TOKEN, SEP_TOKEN, START_TOKEN, CTX_TOKEN, PAD_TOKEN, UNK_TOKEN],
                                          lower=True,
                                          vectors='glove.840B.300d',
                                          vocab_cache=cache_sub_path.joinpath("input_vocab.pkl"))
        vocabs[LOGICAL_FORM] = self._build_vocab([item[2] for item in data_aggregate],
                                                 specials=[START_TOKEN, END_TOKEN, PAD_TOKEN, UNK_TOKEN],
                                                 lower=True,
                                                 vocab_cache=cache_sub_path.joinpath("lf_vocab.pkl"))
        vocabs[NER] = self._build_vocab([item[3] for item in data_aggregate],
                                        specials=[O, PAD_TOKEN],
                                        vocab_cache=cache_sub_path.joinpath("ner_vocab.pkl"))
        vocabs[COREF] = self._build_vocab([item[4] for item in data_aggregate],
                                          specials=['0', PAD_TOKEN],
                                          vocab_cache=cache_sub_path.joinpath("coref_vocab.pkl"))
        vocabs[PREDICATE_POINTER] = self._build_vocab([item[5] for item in data_aggregate],
                                                      specials=[NA_TOKEN, PAD_TOKEN],
                                                      vocab_cache=cache_sub_path.joinpath("pred_vocab.pkl"))
        vocabs[TYPE_POINTER] = self._build_vocab([item[6] for item in data_aggregate],
                                                 specials=[NA_TOKEN, PAD_TOKEN],
                                                 vocab_cache=cache_sub_path.joinpath("type_vocab.pkl"))
        vocabs[ENTITY] = self._build_vocab([item[7] for item in data_aggregate],
                                           specials=[PAD_TOKEN, NA_TOKEN],
                                           vocab_cache=cache_sub_path.joinpath("ent_vocab.pkl"))

        return vocabs

    def preprocess_data(self, splits=('train', 'val', 'test')):
        source_paths = {'train': self.train_path,
                        'val': self.val_path,
                        'test': self.test_path}
        data = dict()
        helpers = dict()
        for split in splits:
            data_cache_file = self.cache_path.joinpath(split).with_suffix(".pkl")
            helper_cache_file = self.cache_path.joinpath(f"{split}_helper").with_suffix(".pkl")
            if data_cache_file.exists() and helper_cache_file.exists() and not self.no_data_cache:
                print(f"Loading {split} from cache...")
                data[split] = pickle.load(data_cache_file.open("rb"), encoding='utf8')
                helpers[split] = pickle.load(helper_cache_file.open("rb"), encoding='utf8')
            else:
                print(f"Building {split} from raw data...")
                raw_data = []
                split_files = source_paths[split].glob("*/QA_*.json")
                for f in split_files:
                    with open(f, encoding='utf8') as json_file:
                        raw_data.append(json.load(json_file))

                data[split], helpers[split] = self._prepare_data(raw_data)
                pickle.dump(data[split], data_cache_file.open("wb"))
                pickle.dump(helpers[split], helper_cache_file.open("wb"))

        return data, helpers

    def _prepare_data(self, data):
        input_data = []
        helper_data = {
            QUESTION_TYPE: [], ENTITY: {GOLD: {}, LABEL: {}}}
        for j, conversation in enumerate(data):
            prev_user_conv = None
            prev_system_conv = None
            is_clarification = False
            is_history_ner_spurious = False
            turns = len(conversation) // 2
            for i in range(turns):
                input = []
                logical_form = []
                ner_tag = []
                coref = []
                entity_pointer = set()
                entity_idx = []
                entity_label = []
                predicate_pointer = []
                type_pointer = []

                if is_clarification:
                    is_clarification = False
                    continue

                user = conversation[2*i]
                system = conversation[2*i + 1]

                if user['question-type'] == 'Simple Insert (Direct)':
                    # No Context
                    # NA + [SEP] + NA + [SEP] + current_question
                    input.extend([NA_TOKEN, SEP_TOKEN, NA_TOKEN, SEP_TOKEN])

                    # ner_tag
                    ner_tag.extend([O, O, O, O])
                    for context in user['context']:
                        input.append(context[1])
                        ner_tag.append(f'{context[-1]}-{context[-2]}' if context[-1] in [B, I] else context[-1])

                    # coref entities - prepare coref values
                    action_entities = [action[1] for action in system[GOLD_ACTIONS] if action[0] == ENTITY]
                    for context in reversed(user['context']):
                        if context[2] in action_entities and context[4] == B and str(action_entities.index(context[2])) not in coref and user['description'] not in ['Simple Question|Mult. Entity', 'Verification|one entity, multiple entities (as object) referred indirectly']:
                            coref.append(str(action_entities.index(context[2])))
                        else:
                            coref.append(NA_TOKEN)
                    coref.extend([NA_TOKEN, NA_TOKEN, NA_TOKEN, NA_TOKEN])

                    # entity pointer # TODO: this is a hack and we do not need it
                    if 'entities' in user: entity_pointer.update(user['entities'])
                    if 'entities_in_utterance' in user: entity_pointer.update(user['entities_in_utterance'])

                    # get gold actions
                    gold_actions = system[GOLD_ACTIONS]

                    # track context history
                    prev_user_conv = user.copy()
                    prev_system_conv = system.copy()
                elif user['question-type'] == 'Clarification':
                    # get next context
                    is_clarification = True
                    next_user = conversation[2*(i+1)]
                    next_system = conversation[2*(i+1) + 1]

                    # skip if ner history is spurious
                    if is_history_ner_spurious:
                        is_history_ner_spurious = False
                        if not next_user['is_ner_spurious'] and not next_system['is_ner_spurious']:
                            prev_user_conv = next_user.copy()
                            prev_system_conv = next_system.copy()
                        else:
                            is_history_ner_spurious = True
                        continue

                    # skip if ner is spurious
                    if user['is_ner_spurious'] or system['is_ner_spurious'] or next_user['is_ner_spurious'] or next_system['is_ner_spurious']:
                        is_history_ner_spurious = True
                        continue

                    # skip if no gold action (or spurious)
                    if 'gold_actions' not in next_system or next_system['is_spurious']:
                        prev_user_conv = next_user.copy()
                        prev_system_conv = next_system.copy()
                        continue

                    if i == 0: # NA + [SEP] + NA + [SEP] + current_question
                        input.extend([NA_TOKEN, SEP_TOKEN, NA_TOKEN, SEP_TOKEN])
                        ner_tag.extend([O, O, O, O])
                    else:
                        # add prev context user
                        for context in prev_user_conv['context']:
                            input.append(context[1])
                            ner_tag.append(f'{context[-1]}-{context[-2]}' if context[-1] in [B, I] else context[-1])

                        # sep token
                        input.append(SEP_TOKEN)
                        ner_tag.append(O)

                        # add prev context answer
                        for context in prev_system_conv['context']:
                            input.append(context[1])
                            ner_tag.append(f'{context[-1]}-{context[-2]}' if context[-1] in [B, I] else context[-1])

                        # sep token
                        input.append(SEP_TOKEN)
                        ner_tag.append(O)

                    # user context
                    for context in user['context']:
                        input.append(context[1])
                        ner_tag.append(f'{context[-1]}-{context[-2]}' if context[-1] in [B, I] else context[-1])

                    # system context
                    for context in system['context']:
                        input.append(context[1])
                        ner_tag.append(f'{context[-1]}-{context[-2]}' if context[-1] in [B, I] else context[-1])

                    # next user context
                    for context in next_user['context']:
                        input.append(context[1])
                        ner_tag.append(f'{context[-1]}-{context[-2]}' if context[-1] in [B, I] else context[-1])

                    # ANCHOR LASAGNE coref entities - prepare coref values
                    action_entities = [action[1] for action in next_system[GOLD_ACTIONS] if action[0] == ENTITY]
                    for context in reversed(user['context'] + system['context'] + next_user['context']):
                        if context[2] in action_entities and context[4] == B and str(action_entities.index(context[2])) not in coref:
                            coref.append(str(action_entities.index(context[2])))
                        else:
                            coref.append(NA_TOKEN)

                    if i == 0:
                        coref.extend([NA_TOKEN, NA_TOKEN, NA_TOKEN, NA_TOKEN])
                    else:
                        coref.append(NA_TOKEN)
                        for context in reversed(prev_system_conv['context']):
                            if context[2] in action_entities and context[4] == B and str(action_entities.index(context[2])) not in coref:
                                coref.append(str(action_entities.index(context[2])))
                            else:
                                coref.append(NA_TOKEN)

                        coref.append(NA_TOKEN)
                        for context in reversed(prev_user_conv['context']):
                            if context[2] in action_entities and context[4] == B and str(action_entities.index(context[2])) not in coref:
                                coref.append(str(action_entities.index(context[2])))
                            else:
                                coref.append(NA_TOKEN)

                    # entities turn # NOTE: this just takes all available entities in this turn (needed for Clarification)
                    if 'entities' in prev_user_conv: entity_pointer.update(prev_user_conv['entities'])
                    if 'entities_in_utterance' in prev_user_conv: entity_pointer.update(prev_user_conv['entities_in_utterance'])
                    entity_pointer.update(prev_system_conv['entities_in_utterance'])
                    if 'entities' in user: entity_pointer.update(user['entities'])
                    if 'entities_in_utterance' in user: entity_pointer.update(user['entities_in_utterance'])
                    entity_pointer.update(system['entities_in_utterance'])
                    if 'entities' in next_user: entity_pointer.update(next_user['entities'])
                    if 'entities_in_utterance' in next_user: entity_pointer.update(next_user['entities_in_utterance'])

                    # get gold actions
                    gold_actions = next_system[GOLD_ACTIONS]

                    # track context history
                    prev_user_conv = next_user.copy()
                    prev_system_conv = next_system.copy()
                else:
                    if is_history_ner_spurious: # skip if history is ner spurious
                        is_history_ner_spurious = False
                        if not user['is_ner_spurious'] and not system['is_ner_spurious']:
                            prev_user_conv = user.copy()
                            prev_system_conv = system.copy()
                        else:
                            is_history_ner_spurious = True

                        continue
                    if user['is_ner_spurious'] or system['is_ner_spurious']:  # skip if ner is spurious
                        is_history_ner_spurious = True
                        continue

                    if GOLD_ACTIONS not in system or system['is_spurious']:  # skip if logical form is spurious
                        prev_user_conv = user.copy()
                        prev_system_conv = system.copy()
                        continue

                    if i == 0: # NA + [SEP] + NA + [SEP] + current_question
                        input.extend([NA_TOKEN, SEP_TOKEN, NA_TOKEN, SEP_TOKEN])
                        ner_tag.extend([O, O, O, O])
                    else:
                        # add prev context user
                        for context in prev_user_conv['context']:
                            input.append(context[1])
                            ner_tag.append(f'{context[-1]}-{context[-2]}' if context[-1] in [B, I] else context[-1])

                        # sep token
                        input.append(SEP_TOKEN)
                        ner_tag.append(O)

                        # add prev context answer
                        for context in prev_system_conv['context']:
                            input.append(context[1])
                            ner_tag.append(f'{context[-1]}-{context[-2]}' if context[-1] in [B, I] else context[-1])

                        # sep token
                        input.append(SEP_TOKEN)
                        ner_tag.append(O)

                    # user context
                    for context in user['context']:
                        input.append(context[1])
                        ner_tag.append(f'{context[-1]}-{context[-2]}' if context[-1] in [B, I] else context[-1])

                    # coref entities - prepare coref values
                    action_entities = [action[1] for action in system[GOLD_ACTIONS] if action[0] == ENTITY]
                    for context in reversed(user['context']):
                        if context[2] in action_entities and context[4] == B and str(action_entities.index(context[2])) not in coref and user['description'] not in ['Simple Question|Mult. Entity', 'Verification|one entity, multiple entities (as object) referred indirectly']:
                            coref.append(str(action_entities.index(context[2])))
                        else:
                            coref.append(NA_TOKEN)

                    # ANCHOR LASAGNE
                    if i == 0:
                        coref.extend([NA_TOKEN, NA_TOKEN, NA_TOKEN, NA_TOKEN])
                    else:
                        coref.append(NA_TOKEN)
                        for context in reversed(prev_system_conv['context']):
                            if context[2] in action_entities and context[4] == B and str(action_entities.index(context[2])) not in coref and user['description'] not in ['Simple Question|Mult. Entity', 'Verification|one entity, multiple entities (as object) referred indirectly']:
                                coref.append(str(action_entities.index(context[2])))
                            else:
                                coref.append(NA_TOKEN)

                        coref.append(NA_TOKEN)
                        for context in reversed(prev_user_conv['context']):
                            if context[2] in action_entities and context[4] == B and str(action_entities.index(context[2])) not in coref and user['description'] not in ['Simple Question|Mult. Entity', 'Verification|one entity, multiple entities (as object) referred indirectly']:
                                coref.append(str(action_entities.index(context[2])))
                            else:
                                coref.append(NA_TOKEN)

                    # entities turn
                    if prev_user_conv is not None and prev_system_conv is not None:
                        if 'entities' in prev_user_conv: entity_pointer.update(prev_user_conv['entities'])
                        if 'entities_in_utterance' in prev_user_conv: entity_pointer.update(prev_user_conv['entities_in_utterance'])
                        entity_pointer.update(prev_system_conv['entities_in_utterance'])
                    if 'entities' in user: entity_pointer.update(user['entities'])
                    if 'entities_in_utterance' in user: entity_pointer.update(user['entities_in_utterance'])

                    # get gold actions
                    gold_actions = system[GOLD_ACTIONS]

                    # track context history
                    prev_user_conv = user.copy()
                    prev_system_conv = system.copy()

                # prepare entities
                entity_pointer = list(entity_pointer)
                entity_pointer.insert(0, PAD_TOKEN)
                entity_pointer.insert(0, NA_TOKEN)

                # prepare logical form
                for action in gold_actions:
                    if action[0] == ACTION:
                        logical_form.append(action[1])
                        predicate_pointer.append(NA_TOKEN)
                        type_pointer.append(NA_TOKEN)
                        entity_idx.append(entity_pointer.index(NA_TOKEN))
                        entity_label.append(NA_TOKEN)
                    elif action[0] == RELATION:
                        logical_form.append(RELATION)
                        predicate_pointer.append(action[1])
                        type_pointer.append(NA_TOKEN)
                        entity_idx.append(entity_pointer.index(NA_TOKEN))
                        entity_label.append(NA_TOKEN)
                    elif action[0] == TYPE:
                        logical_form.append(TYPE)
                        predicate_pointer.append(NA_TOKEN)
                        type_pointer.append(action[1])
                        entity_idx.append(entity_pointer.index(NA_TOKEN))
                        entity_label.append(NA_TOKEN)
                    elif action[0] == ENTITY:
                        logical_form.append(PREV_ANSWER if action[1] == PREV_ANSWER else ENTITY)
                        predicate_pointer.append(NA_TOKEN)
                        type_pointer.append(NA_TOKEN)
                        entity_idx.append(entity_pointer.index(action[1] if action[1] != PREV_ANSWER else NA_TOKEN))
                        entity_label.append(action[1] if action[1] != PREV_ANSWER else NA_TOKEN)
                    elif action[0] == VALUE:
                        logical_form.append(action[0])
                        predicate_pointer.append(NA_TOKEN)
                        type_pointer.append(NA_TOKEN)
                        entity_idx.append(entity_pointer.index(NA_TOKEN))
                        entity_label.append(NA_TOKEN)
                    else:
                        raise Exception(f'Unkown logical form action {action[0]}')

                assert len(input) == len(ner_tag)
                assert len(input) == len(coref)
                assert len(logical_form) == len(predicate_pointer)
                assert len(logical_form) == len(type_pointer)
                assert len(logical_form) == len(entity_idx)
                assert len(logical_form) == len(entity_label)

                input_data.append([str(self.id),
                                   input,
                                   logical_form,
                                   ner_tag,  # ANCHOR LASAGNE
                                   list(reversed(coref)),  # ANCHOR LASAGNE
                                   predicate_pointer,
                                   type_pointer,
                                   entity_pointer])

                helper_data[QUESTION_TYPE].append(user['question-type'])
                helper_data[ENTITY][GOLD][str(self.id)] = entity_idx
                helper_data[ENTITY][LABEL][str(self.id)] = entity_label

                self.id += 1

        return input_data, helper_data

    def get_inference_data(self, max_files: int = None):
        files = self.test_path.glob('*/QA_*.json')

        partition = []
        for i, f in enumerate(files):
            if max_files is not None and i > max_files:
                break

            with f.open(encoding='utf8') as json_file:
                partition.append(json.load(json_file))

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased').tokenize
        inference_data = []

        for conversation in partition:
            is_clarification = False
            prev_user_conv = {}
            prev_system_conv = {}
            turns = len(conversation) // 2
            for i in range(turns):
                input = []
                gold_entities = []

                if is_clarification:
                    is_clarification = False
                    continue

                user = conversation[2*i]
                system = conversation[2*i + 1]

                if i > 0 and 'context' not in prev_system_conv:
                    if len(prev_system_conv['entities_in_utterance']) > 0:
                        tok_utterance = tokenizer(prev_system_conv['utterance'].lower())
                        prev_system_conv['context'] = [[i, tok] for i, tok in enumerate(tok_utterance)]
                    elif prev_system_conv['utterance'].isnumeric():
                        prev_system_conv['context'] = [[0, 'num']]
                    elif prev_system_conv['utterance'] == 'YES':
                        prev_system_conv['context'] = [[0, 'yes']]
                    elif prev_system_conv['utterance'] == 'NO':
                        prev_system_conv['context'] = [[0, 'no']]
                    elif prev_system_conv['utterance'] == 'YES and NO respectively':
                        prev_system_conv['context'] = [[0, 'no']]
                    elif prev_system_conv['utterance'] == 'NO and YES respectively':
                        prev_system_conv['context'] = [[0, 'no']]
                    elif prev_system_conv['utterance'][0].isnumeric():
                        prev_system_conv['context'] = [[0, 'num']]

                if user['question-type'] == 'Clarification':
                    # get next context
                    is_clarification = True
                    next_user = conversation[2*(i+1)]
                    next_system = conversation[2*(i+1) + 1]

                    if i == 0: # NA + [SEP] + NA + [SEP] + current_question
                        input.extend([NA_TOKEN, SEP_TOKEN, NA_TOKEN, SEP_TOKEN])
                    else:
                        # add prev context user
                        for context in prev_user_conv['context']: input.append(context[1])

                        # sep token
                        input.append(SEP_TOKEN)

                        # add prev context answer
                        for context in prev_system_conv['context']: input.append(context[1])

                        # sep token
                        input.append(SEP_TOKEN)

                    # user context
                    for context in user['context']: input.append(context[1])

                    # system context
                    for context in system['context']: input.append(context[1])

                    # next user context
                    for context in next_user['context']: input.append(context[1])

                    question_type = [user['question-type'], next_user['question-type']] if 'question-type' in next_user else user['question-type']
                    results = next_system['all_entities']
                    answer = next_system['utterance']
                    gold_actions = next_system[GOLD_ACTIONS] if GOLD_ACTIONS in next_system else None
                    prev_answer = prev_system_conv['all_entities'] if 'all_entities' in prev_system_conv else None

                    # entities turn
                    context_entities = set()
#                    if 'entities' in prev_user_conv: context_entities.update(prev_user_conv['entities'])
                    if 'entities_in_utterance' in prev_user_conv: context_entities.update(prev_user_conv['entities_in_utterance'])
                    if prev_system_conv: context_entities.update(prev_system_conv['entities_in_utterance'])
#                    if 'entities' in user: context_entities.update(user['entities'])
                    if 'entities_in_utterance' in user: context_entities.update(user['entities_in_utterance'])
                    context_entities.update(system['entities_in_utterance'])
#                    if 'entities' in next_user: context_entities.update(next_user['entities'])
                    if 'entities_in_utterance' in next_user: context_entities.update(next_user['entities_in_utterance'])

                    # track context history
                    prev_user_conv = next_user.copy()
                    prev_system_conv = next_system.copy()
                else:
                    if i == 0: # NA + [SEP] + NA + [SEP] + current_question
                        input.extend([NA_TOKEN, SEP_TOKEN, NA_TOKEN, SEP_TOKEN])
                    else:
                        # add prev context user
                        for context in prev_user_conv['context']:
                            input.append(context[1])

                        # sep token
                        input.append(SEP_TOKEN)

                        # add prev context answer
                        for context in prev_system_conv['context']:
                            input.append(context[1])

                        # sep token
                        input.append(SEP_TOKEN)

                    if 'context' not in user:
                        tok_utterance = tokenizer(user['utterance'].lower())
                        user['context'] = [[i, tok] for i, tok in enumerate(tok_utterance)]

                    # user context
                    for context in user['context']: input.append(context[1])

                    question_type = user['question-type']
                    results = system['all_entities']
                    answer = system['utterance']
                    gold_actions = system[GOLD_ACTIONS] if GOLD_ACTIONS in system else None
                    prev_results = prev_system_conv['all_entities'] if 'all_entities' in prev_system_conv else None

                    # entities turn
                    context_entities = set()
                    if prev_user_conv is not None and prev_system_conv is not None:
                        # if 'entities' in prev_user_conv: context_entities.update(prev_user_conv['entities'])
                        if 'entities_in_utterance' in prev_user_conv: context_entities.update(prev_user_conv['entities_in_utterance'])
                        if prev_system_conv: context_entities.update(prev_system_conv['entities_in_utterance'])
#                    if 'entities' in user: context_entities.update(user['entities'])
                    if 'entities_in_utterance' in user: context_entities.update(user['entities_in_utterance'])
                    context_entities.update(system['entities_in_utterance'])

                    # track context history
                    prev_user_conv = user.copy()
                    prev_system_conv = system.copy()

                context_entities = list(context_entities)
                context_entities.insert(0, PAD_TOKEN)
                context_entities.insert(0, NA_TOKEN)

                inference_data.append({
                    QUESTION_TYPE: question_type,
                    QUESTION: user['utterance'],
                    CONTEXT_QUESTION: input,
                    CONTEXT_ENTITIES: context_entities,
                    ANSWER: answer,
                    RESULTS: results,
                    PREV_RESULTS: prev_results,
                    GOLD_ACTIONS: gold_actions
                })

        return inference_data

    @staticmethod
    def _make_torchtext_dataset(data, fields):
        examples = [Example.fromlist(i, fields) for i in data]
        return Dataset(examples, fields)

    def _build_vocab(self, tokens, specials: list[str], lower=False, min_freq=0,
                     vectors=None,
                     vocab_cache: pathlib.Path = None):
        if vocab_cache is not None and vocab_cache.exists() and not self.no_vocab_cache:
            print(f"\t...loading {vocab_cache.stem} from {vocab_cache}")
            return pickle.load(vocab_cache.open("rb"), encoding="utf8")

        # TODO: batch first
        if lower:
            tokens = list(token.lower() for token in chain(*tokens))
        else:
            tokens = list(chain(*tokens))
        # Count unique tokens
        counts = Counter(tokens)
        # Build a vocabulary by assigning a unique ID to each token
        vocab = Vocab(counts, specials=specials, min_freq=min_freq, vectors=vectors)

        if vocab_cache is not None:
            pickle.dump(vocab, vocab_cache.open("wb"))
        return vocab

    def get_data(self):
        return self.data['train'], self.data['val'], self.data['test']

    def get_data_helper(self):
        return self.helpers['train'], self.helpers['val'], self.helpers['test']

    # def get_fields(self):
    #     return {
    #         ID: self.id_field,
    #         INPUT: self.input_field,
    #         LOGICAL_FORM: self.lf_field,
    #         NER: self.ner_field,
    #         COREF: self.coref_field,
    #         PREDICATE_POINTER: self.predicate_field,
    #         TYPE_POINTER: self.type_field,
    #     }

    def get_vocabs(self):
        return {
            ID: self.vocabs[ID],
            INPUT: self.vocabs[INPUT],
            LOGICAL_FORM: self.vocabs[LOGICAL_FORM],
            NER: self.vocabs[NER],
            COREF: self.vocabs[COREF],
            PREDICATE_POINTER: self.vocabs[PREDICATE_POINTER],
            TYPE_POINTER: self.vocabs[TYPE_POINTER],
            ENTITY: self.vocabs[ENTITY],
        }
