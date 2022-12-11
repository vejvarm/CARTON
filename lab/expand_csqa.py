import json
import logging
import re
from pathlib import Path
from unidecode import unidecode
from typing import Protocol
from abc import ABC, abstractmethod
from enum import Enum, auto

from constants import args, ROOT_PATH, ENTITY, TYPE, RELATION

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from action_executor.actions import ESActionOperator
from helpers import connect_to_elasticsearch, setup_logger

# TODOlist
# Question Answer processing during QA2D transformation:
#   DONE: Use plain question and answer strings (use_ent_id_in_transformations = False)
#   DONE: Use entity identifiers (id) in place of labels (use_ent_id_in_transformations = True)
#   DONE: Make entitiy ids lowercased (e.g. Q1235 -> q1235)
#   TODO: use placeholder names for entities (instead of Q1234, use e.g. f'entity{i}' )
#   TODO: Use type labels as placeholder for multiple entities in answer
#   TODO: Use type ids as placeholder for multiple entities in answer
#   TODO: Always look on the bright side of life
# Dataset requirements:
#   DONE: only take Simple Question types. Those we will need to transform
#   TODO: how about the questions around them? ...
#   TODO: for Coreferecnce and Ellipsis, we must keep the previous questions in the dataset!!!
# Question types:
# Simple (Direct)
#   Simple Question
#   Simple Question|Single Entity
#   Simple Question|Mult. Entity|Indirect
#   TODO fix:
#       07/12/2022 11:22:25 PM __main__     INFO     qa_str in transform_utterances: Which people are the life partner of Q63749 and Q213671 ? Q288703, Q813294, Q542719
#       07/12/2022 11:22:25 PM __main__     INFO     declarative_str in transform_utterances: Q288703, Q813294, Q542719 are the life partner of
#       07/12/2022 11:22:25 PM __main__     INFO     qa_str in transform_utterances: Which person have that architectural structure as their work location ? Q91103
#       statement in __main__: Landgravine Caroline Louise of Hesse-Darmstadt, Beatrix of Julich-Berg, Louise Caroline of Hochberg are the life partner of

# Simple (Coreference)
#   Simple Question|Single Entity|Indirect
#   Simple Question|Mult. Entity
#   TODO fix:
#       07/12/2022 11:22:27 PM __main__     INFO     qa_str in transform_utterances: Which political territories are bordered by those ones ? Q183 for 1st, 2nd, Q834010 for 3rd
#       07/12/2022 11:22:27 PM __main__     INFO     declarative_str in transform_utterances: Q183 for 1st, 2nd, Q834010 for 3rd are
#       07/12/2022 11:22:27 PM __main__     INFO     qa_str in transform_utterances: And what about Q19893635? Q21
#       statement in __main__: Germany for 1st, 2nd, Villar del Rio for 3rd are

# Simple (Ellipsis)
#   only subject is changed, parent and predicate remains same
#   Incomplete|object parent is changed, subject and predicate remain same

CLIENT = connect_to_elasticsearch()
LOGGER = setup_logger(__name__, loglevel=logging.INFO)


class QA2DModelChoices(Enum):
    QA2DT5_SMALL = 'domenicrosati/QA2D-t5-small'
    QA2DT5_BASE = 'domenicrosati/QA2D-t5-base'
    QC3B = 'domenicrosati/question_converter-3b'


class RepresentEntityLabelAs(Enum):
    LABEL = auto
    ENTITY_ID = auto
    PLACEHOLDER = auto
    TYPE_ID = auto  # TODO: Implement


class Preprocessor(ABC):
    SEP: str

    @classmethod
    @abstractmethod
    def combine_qa(cls, question: str, answer: str) -> str:
        raise NotImplementedError


class QA2DPreprocessor(Preprocessor):
    SEP = ". "

    @classmethod
    def combine_qa(cls, question: str, answer: str) -> str:
        q = question.replace("?", "").strip()
        a = answer.strip()
        return f'{q}{cls.SEP}{a}'


class B3Preprocessor(Preprocessor):
    SEP = "</s>"

    @classmethod
    def combine_qa(cls, question: str, answer: str) -> str:
        q = question.replace(" ?", "?").strip()
        a = answer.strip()
        return f"{q} {cls.SEP} {a}"


class QA2DModel:

    def __init__(self, model_type: QA2DModelChoices):
        self.model_type = model_type
        self.tokenizer, self.model = self._get_tokenizer_and_model(model_type)
        self.preprocessor = self._get_preprocessor(model_type)

    @staticmethod
    def _get_tokenizer_and_model(model_type: QA2DModelChoices):
        tokenizer = AutoTokenizer.from_pretrained(model_type.value)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_type.value)

        return tokenizer, model

    @staticmethod
    def _get_preprocessor(model_type: QA2DModelChoices):
        if model_type in (QA2DModelChoices.QA2DT5_SMALL, QA2DModelChoices.QA2DT5_BASE):
            return QA2DPreprocessor
        elif model_type == QA2DModelChoices.QC3B:
            return B3Preprocessor
        else:
            raise NotImplementedError(
                f'Chosen model type ({model_type}) is not supported. Refer to QA2DModelChoices class.')

    def infer_one(self, question: str, answer: str) -> str:
        qa_string = self.preprocessor.combine_qa(question, answer)
        LOGGER.info(f"qa_string in infer_one: {qa_string}")
        input_ids = self.tokenizer(qa_string, return_tensors="pt").input_ids
        LOGGER.debug(f"input_ids in infer_one: ({input_ids.shape}) {input_ids}")

        outputs = self.model.generate(input_ids)
        LOGGER.debug(f"outputs in infer_one: ({outputs.shape}) {outputs}")

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


class CSQAInsertBuilder:

    def __init__(self, operator: ESActionOperator, qa2d_model: QA2DModel):
        self.op = operator
        self.qa2d_model = qa2d_model

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

    def _replace_labels_in_utterance(self, utterance: str, entities: list[str], labels_as: RepresentEntityLabelAs) -> tuple[str, dict]:
        inverse_map = dict()
        utterance = unidecode(utterance)
        if labels_as == RepresentEntityLabelAs.LABEL or labels_as not in RepresentEntityLabelAs:
            return utterance, inverse_map

        for i, ent in enumerate(entities):
            label = self.op.get_label(ent)
            if labels_as == RepresentEntityLabelAs.ENTITY_ID:
                repl = ent.lower()
            elif labels_as == RepresentEntityLabelAs.PLACEHOLDER:
                repl = f"entity{i}"
            else:
                raise NotImplementedError(f'Chosen RepresentEntityLabeAs ({labels_as}) Enum is not supported.')
            utterance = utterance.replace(label, repl)
            inverse_map[repl] = label

        return utterance, inverse_map

    def transorm_utterances(self, user: dict[list[str] or str], system: dict[list[str] or str], labels_as: RepresentEntityLabelAs) -> str:
        """ Transform user utterance (Question) and system utterance (Answer) to declarative statements.

        :param user: conversation turn of the user from the CSQA dataset
        :param system: conversation turn of the system from the CSQA dataset
        :param labels_as: substitute labels to respective choice before transformation (and back after trans)
        :return: declarative string
        """
        user_utterance = user['utterance']
        user_ents = user['entities_in_utterance']
        system_utterance = system['utterance']
        system_ents = system['entities_in_utterance']

        # substitute all labels with chosen replacement
        user_utterance, user_inverse_map = self._replace_labels_in_utterance(user_utterance, user_ents, labels_as)
        system_utterance, system_inverse_map = self._replace_labels_in_utterance(system_utterance, system_ents, labels_as)
        inverse_map = {**user_inverse_map, **system_inverse_map}

        LOGGER.info(f'utterances in transform_utterances: U: {user_utterance} S: {system_utterance}')
        declarative_str = self.qa2d_model.infer_one(user_utterance, system_utterance)
        LOGGER.info(f'declarative_str in transform_utterances: {declarative_str}')

        # replace entity ids back with labels
        for eid, lab in inverse_map.items():
            declarative_str = re.sub(eid, lab, declarative_str, flags=re.IGNORECASE)

        return declarative_str

    def transform_fields(self):
        pass


if __name__ == "__main__":
    model_choice = QA2DModelChoices.QA2DT5_SMALL
    use_ent_id_in_transformations = True

    # pop unneeded conversations right here?
    args.read_folder = '/data'  # 'folder to read conversations'
    args.partition = ''  # 'train', 'test', 'val', ''

    # read data and create csqa data dict
    # dict: partition -> folder -> file -> conversation
    data_folder = Path(f'{ROOT_PATH}{args.read_folder}/{args.partition}')
    # csqa_files = data_folder.glob('**/QA*.json')
    csqa_files = data_folder.glob('**/d_dataset_like_example_file.json')
    LOGGER.info(f'Reading folders for partition {args.partition}')

    op = ESActionOperator(CLIENT)
    transformer = QA2DModel(model_choice)  # or QuestionConverter3B
    builder = CSQAInsertBuilder(op, transformer)

    for pth in csqa_files:
        folder = pth.parent.name
        file = pth.name

        with open(pth, encoding='utf8') as json_file:
            conversation = json.load(json_file)

        for i in range(len(conversation)//2):
            entry_user = conversation[2*i]  # USER
            entry_system = conversation[2*i + 1]  # SYSTEM

            if 'Simple' in entry_user['question-type']:
                LOGGER.info(f"USER: {entry_user['description']}, {entry_user['entities_in_utterance']}, {entry_user['relations']}, {entry_user['utterance']}")
                LOGGER.info(f"SYSTEM: {entry_system['entities_in_utterance']} {entry_system['utterance']}")
                LOGGER.info(f"active_set: {entry_system['active_set']}")

                # 1) TRANSFORM active_set field
                new_active_set = builder.build_active_set(entry_user, entry_system)
                LOGGER.info(f'new_active_set: {new_active_set}')

                # 2) TRANSFORM utterances to statements  # TODO: still needs a lot of tweaking
                statement = builder.transorm_utterances(entry_user, entry_system, use_ids=use_ent_id_in_transformations)
                LOGGER.info(f'statement: {statement}')
                LOGGER.info(f"".center(50, "-")+"\n\n")

                # 3) TRANSFORM all other fields in conversation turns TODO: implement

            # conversation types to tweak:
            # and how?
