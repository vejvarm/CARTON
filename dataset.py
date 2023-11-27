import json
import pathlib
import pickle
from collections import Counter
from dataclasses import dataclass
from glob import glob
from itertools import chain

from torchtext.vocab import Vocab
from tqdm import tqdm
from transformers import BertTokenizer
from torchtext.data import Field, Example, Dataset
import torch
from torch.nn.utils.rnn import pad_sequence

from constants import (LOGICAL_FORM, ROOT_PATH, QUESTION_TYPE, ENTITY, GOLD, LABEL, NA_TOKEN, SEP_TOKEN,
                       GOLD_ACTIONS, PAD_TOKEN, ACTION, RELATION, TYPE, PREV_ANSWER, VALUE, QUESTION,
                       CONTEXT_QUESTION, CONTEXT_ENTITIES, ANSWER, RESULTS, PREV_RESULTS, START_TOKEN, CTX_TOKEN,
                       UNK_TOKEN, END_TOKEN, INPUT, ID, NER, COREF, PREDICATE_POINTER, TYPE_POINTER, B, I, O)


@dataclass
class DataBatch:
    """
        data[split]
        [0] ... ID
        [1] ... INPUT
        [2] ... LOGICAL_FORM
        [3] ... NER
        [4] ... COREF
        [5] ... PREDICATE_POINTER
        [6] ... TYPE_POINTER
        [7] ... ENTITY
    """
    id: torch.Tensor  # str
    input: torch.Tensor  # str
    logical_form: torch.Tensor  # list[str]
    ner: torch.Tensor  # list[str]
    coref: torch.Tensor  # list[str]
    predicate_pointer: torch.Tensor  # list[int]
    type_pointer = torch.Tensor  # list[int]
    entity_pointer = torch.Tensor  # list[int]

    def __init__(self, batch: list[list[any]], vocabs: dict, device: str):
        id = []
        inp = []
        lf = []
        ner = []
        coref = []
        predicate_pointer = []
        type_pointer = []
        entity_pointer = []
        for sample in batch:
            id.append(int(sample[0]))
            inp.append(self._tensor([vocabs[INPUT].stoi[s] for s in sample[1]]))
            lf.append(self._tensor([vocabs[LOGICAL_FORM].stoi[s] for s in sample[2]]))
            ner.append(self._tensor([vocabs[NER].stoi[s] for s in sample[3]]))
            coref.append(self._tensor([vocabs[COREF].stoi[s] for s in sample[4]]))
            predicate_pointer.append(self._tensor([vocabs[PREDICATE_POINTER].stoi[s] for s in sample[5]]))
            type_pointer.append(self._tensor([vocabs[TYPE_POINTER].stoi[s] for s in sample[6]]))
            entity_pointer.append(self._tensor([vocabs[ENTITY].stoi[s] for s in sample[7]]))

        self.id = self._tensor(id).to(device)
        self.input = pad_sequence(inp,
                                  padding_value=vocabs[INPUT].stoi[PAD_TOKEN],
                                  batch_first=True).to(device)
        self.logical_form = pad_sequence(lf,
                                         padding_value=vocabs[LOGICAL_FORM].stoi[PAD_TOKEN],
                                         batch_first=True).to(device)  # ANCHOR this is not gonna work as we assume all LFs have same length, which is not true
        self.ner = pad_sequence(ner,
                                padding_value=vocabs[NER].stoi[PAD_TOKEN],
                                batch_first=True).to(device)
        self.coref = pad_sequence(coref,
                                  padding_value=vocabs[COREF].stoi[PAD_TOKEN],
                                  batch_first=True).to(device)
        self.predicate_pointer = pad_sequence(predicate_pointer,
                                              padding_value=vocabs[PREDICATE_POINTER].stoi[PAD_TOKEN],
                                              batch_first=True).to(device)
        self.type_pointer = pad_sequence(type_pointer,
                                         padding_value=vocabs[TYPE_POINTER].stoi[PAD_TOKEN],
                                         batch_first=True).to(device)
        self.entity_pointer = pad_sequence(entity_pointer,
                                           padding_value=vocabs[ENTITY].stoi[PAD_TOKEN],
                                           batch_first=True).to(device)

    @staticmethod
    def _tensor(data):
        return torch.tensor(data)


def collate_fn(batch, vocabs: dict, device: str):
    return DataBatch(batch, vocabs, device)


class CSQADataset:

    def __init__(self, args, splits=('train', 'val', 'test')):
        self.data_path = ROOT_PATH.joinpath(args.data_path)
        self.source_paths = {split: self.data_path.joinpath(split) for split in splits}
        self.splits = splits

        self._field_names = [ID, INPUT, LOGICAL_FORM, NER, COREF, PREDICATE_POINTER, TYPE_POINTER, ENTITY]

        # Initialize counters for each field
        self.counters = {k: Counter() for k in self._field_names}
        self.vocabs = dict()
        self.data = None
        self.helpers = None

        self.id = 0
        self.rebuild_data_cache = args.rebuild_data_cache
        self.rebuild_vocab_cache = args.rebuild_vocab_cache
        self.vocab_cache = pathlib.Path(args.vocab_cache)
        self.data_cache = self.data_path.joinpath(".cache")
        # self.data, self.helpers = self.preprocess_data()
        # self.vocabs = self.build_vocabs(args.stream_data)
        # exit()

    def build_vocabs(self, stream_data: bool):
        self.vocab_cache.mkdir(exist_ok=True, parents=True)
        if stream_data:
            self.vocabs = self._build_vocabs_streaming()
        elif self.data is not None:
            self.vocabs = self._build_vocabs()
        else:
            raise ValueError("To build vocabs either set args.stream_data to True or run self.preprocess_data first")

        return self.vocabs

    def _build_vocabs(self):
        """ Build vocabularies for each field from loaded preprocessed data.

        # data[split]:
            # [0] ... ID
            # [1] ... INPUT
            # [2] ... LOGICAL_FORM
            # [3] ... NER
            # [4] ... COREF
            # [5] ... PREDICATE_POINTER
            # [6] ... TYPE_POINTER
            # [7] ... ENTITY
        """


        vocabs = dict()
        data_aggregate = []
        for split in self.data.values():
            data_aggregate.extend(split)
        print("Building vocabularies...")
        vocabs[ID] = self._build_vocab([item[0] for item in data_aggregate],
                                       specials=[],
                                       vocab_cache=self.vocab_cache.joinpath("id_vocab.pkl"))
        vocabs[INPUT] = self._build_vocab([item[1] for item in data_aggregate],
                                          specials=[NA_TOKEN, SEP_TOKEN, START_TOKEN, CTX_TOKEN, PAD_TOKEN, UNK_TOKEN],
                                          lower=True,
                                          vectors='glove.840B.300d',
                                          vocab_cache=self.vocab_cache.joinpath("input_vocab.pkl"))
        vocabs[LOGICAL_FORM] = self._build_vocab([item[2] for item in data_aggregate],
                                                 specials=[START_TOKEN, END_TOKEN, PAD_TOKEN, UNK_TOKEN],
                                                 lower=True,
                                                 vocab_cache=self.vocab_cache.joinpath("lf_vocab.pkl"))
        vocabs[NER] = self._build_vocab([item[3] for item in data_aggregate],
                                        specials=[O, PAD_TOKEN],
                                        vocab_cache=self.vocab_cache.joinpath("ner_vocab.pkl"))
        vocabs[COREF] = self._build_vocab([item[4] for item in data_aggregate],
                                          specials=['0', PAD_TOKEN],
                                          vocab_cache=self.vocab_cache.joinpath("coref_vocab.pkl"))
        vocabs[PREDICATE_POINTER] = self._build_vocab([item[5] for item in data_aggregate],
                                                      specials=[NA_TOKEN, PAD_TOKEN],
                                                      vocab_cache=self.vocab_cache.joinpath("pred_vocab.pkl"))
        vocabs[TYPE_POINTER] = self._build_vocab([item[6] for item in data_aggregate],
                                                 specials=[NA_TOKEN, PAD_TOKEN],
                                                 vocab_cache=self.vocab_cache.joinpath("type_vocab.pkl"))
        vocabs[ENTITY] = self._build_vocab([item[7] for item in data_aggregate],
                                           specials=[PAD_TOKEN, NA_TOKEN],
                                           vocab_cache=self.vocab_cache.joinpath("ent_vocab.pkl"))

        return vocabs

    def _build_vocabs_streaming(self):
        """Build vocabularies from the dataset files in a streaming way (when memory is problem)."""

        for split in self.splits:
            for file_path in tqdm(self.source_paths[split].glob("*/QA_*.json"), desc=f"Loading {split} split"):
                with open(file_path, encoding='utf8') as json_file:
                    raw_data = json.load(json_file)

                processed_data, _ = self._prepare_data([raw_data])
                self.update_counters(processed_data)

        # Create and save vocabularies
        vocabs = dict()
        print("Building vocabularies (streaming)...")
        vocabs[ID] = self._build_vocab(self.counters[ID],
                                       specials=[],
                                       vocab_cache=self.vocab_cache.joinpath("id_vocab.pkl"))
        vocabs[INPUT] = self._build_vocab(self.counters[INPUT],
                                          specials=[NA_TOKEN, SEP_TOKEN, START_TOKEN, CTX_TOKEN, PAD_TOKEN, UNK_TOKEN],
                                          lower=True,
                                          vectors='glove.840B.300d',
                                          vocab_cache=self.vocab_cache.joinpath("input_vocab.pkl"))
        vocabs[LOGICAL_FORM] = self._build_vocab(self.counters[LOGICAL_FORM],
                                                 specials=[START_TOKEN, END_TOKEN, PAD_TOKEN, UNK_TOKEN],
                                                 lower=True,
                                                 vocab_cache=self.vocab_cache.joinpath("lf_vocab.pkl"))
        vocabs[NER] = self._build_vocab(self.counters[NER],
                                        specials=[O, PAD_TOKEN],
                                        vocab_cache=self.vocab_cache.joinpath("ner_vocab.pkl"))
        vocabs[COREF] = self._build_vocab(self.counters[COREF],
                                          specials=['0', PAD_TOKEN],
                                          vocab_cache=self.vocab_cache.joinpath("coref_vocab.pkl"))
        vocabs[PREDICATE_POINTER] = self._build_vocab(self.counters[PREDICATE_POINTER],
                                                      specials=[NA_TOKEN, PAD_TOKEN],
                                                      vocab_cache=self.vocab_cache.joinpath("pred_vocab.pkl"))
        vocabs[TYPE_POINTER] = self._build_vocab(self.counters[TYPE_POINTER],
                                                 specials=[NA_TOKEN, PAD_TOKEN],
                                                 vocab_cache=self.vocab_cache.joinpath("type_vocab.pkl"))
        vocabs[ENTITY] = self._build_vocab(self.counters[ENTITY],
                                           specials=[PAD_TOKEN, NA_TOKEN],
                                           vocab_cache=self.vocab_cache.joinpath("ent_vocab.pkl"))

        return vocabs

    def preprocess_data(self):

        data = dict()
        helpers = dict()
        self.data_cache.mkdir(parents=True, exist_ok=True)
        for split, path_to_split in self.source_paths.items():
            data_cache_file = self.data_cache.joinpath(split).with_suffix(".pkl")
            helper_cache_file = self.data_cache.joinpath(f"{split}_helper").with_suffix(".pkl")
            if data_cache_file.exists() and helper_cache_file.exists() and not self.rebuild_data_cache:
                print(f"Loading {split} from cache...")
                data[split] = pickle.load(data_cache_file.open("rb"), encoding='utf8')
                helpers[split] = pickle.load(helper_cache_file.open("rb"), encoding='utf8')
            else:
                print(f"Building {split} from raw data...")
                raw_data = []
                split_files = path_to_split.glob("*/QA_*.json")
                for f in split_files:
                    with open(f, encoding='utf8') as json_file:
                        raw_data.append(json.load(json_file))

                data[split], helpers[split] = self._prepare_data(raw_data)
                pickle.dump(data[split], data_cache_file.open("wb"))
                pickle.dump(helpers[split], helper_cache_file.open("wb"))

        self.data = data
        self.helpers = helpers

        return self.data, self.helpers

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
        files = self.source_paths['test'].glob('*/QA_*.json')

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
        if vocab_cache is not None and vocab_cache.exists() and not self.rebuild_vocab_cache:
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
        # return self.data['train'], self.data['val'], self.data['test']
        return self.data

    def get_data_helper(self):
        # return self.helpers['train'], self.helpers['val'], self.helpers['test']
        return self.helpers

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

    def update_counters(self, processed_data):
        """Update counters for each field based on processed data.
        # [0] ... ID
        # [1] ... INPUT
        # [2] ... LOGICAL_FORM
        # [3] ... NER
        # [4] ... COREF
        # [5] ... PREDICATE_POINTER
        # [6] ... TYPE_POINTER
        # [7] ... ENTITY
        """

        for i, key in enumerate(self._field_names):
            self.counters[key].update([item[i] for item in processed_data])