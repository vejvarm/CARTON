import json
import logging
import re
from pathlib import Path
from unidecode import unidecode
from typing import Protocol
from abc import ABC, abstractmethod

from constants import args, ROOT_PATH, ENTITY, TYPE, RELATION

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from action_executor.actions import ESActionOperator
from helpers import connect_to_elasticsearch, setup_logger

CLIENT = connect_to_elasticsearch()
LOGGER = setup_logger(__name__, loglevel=logging.INFO)


# class QA2DModel(Protocol):
#     tokenizer: AutoTokenizer
#     model: AutoModelForSeq2SeqLM
#     SEP: str
#
#     @staticmethod
#     def _preprocess(question: str, answer: str, sep: str) -> str:
#         raise NotImplementedError
#
#     @classmethod
#     def infer_one(cls, question: str, answer: str) -> str:
#         raise NotImplementedError


class QA2DModel(ABC):
    tokenizer: AutoTokenizer
    model: AutoModelForSeq2SeqLM
    SEP: str

    @staticmethod
    @abstractmethod
    def _preprocess(question: str, answer: str, sep: str) -> str:
        raise NotImplementedError

    @classmethod
    def infer_one(cls, question: str, answer: str) -> str:
        qa_string = cls._preprocess(question, answer, cls.SEP)
        input_ids = cls.tokenizer(qa_string, return_tensors="pt").input_ids
        LOGGER.debug(f"input_ids in infer_one: ({input_ids.shape}) {input_ids}")

        outputs = cls.model.generate(input_ids)
        LOGGER.debug(f"outputs in infer_one: ({outputs.shape}) {outputs}")

        return cls.tokenizer.decode(outputs[0], skip_special_tokens=True)


class QA2DT5(QA2DModel):
    tokenizer = AutoTokenizer.from_pretrained("domenicrosati/QA2D-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("domenicrosati/QA2D-t5-small")
    SEP = ". "

    @staticmethod
    def _preprocess(question: str, answer: str, sep=SEP) -> str:
        q = question.replace("?", "").strip()
        a = answer.strip()
        return f'{q}{sep}{a}'


class QuestionConverter3B(QA2DModel):
    tokenizer = AutoTokenizer.from_pretrained('domenicrosati/question_converter-3b')
    model = AutoModelForSeq2SeqLM.from_pretrained('domenicrosati/question_converter-3b')
    SEP = "</s>"

    @staticmethod
    def _preprocess(question: str, answer: str, sep=SEP) -> str:
        q = question.replace(" ?", "?").strip()
        a = answer.strip()
        return f"{q} {sep} {a}"


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
        utterance = unidecode(utterance)
        for ent in entities:
            label = self.op.get_label(ent)
            utterance = utterance.replace(label, ent)
            inverse_map[ent] = label

        return utterance, inverse_map

    def transorm_utterances(self, user: dict[list[str] or str], system: dict[list[str] or str], use_ids: bool = True) -> str:
        """ transform user utterance (Question) and system utterance (Answer) to declarative statements

        :param user: conversation turn of the user from the CSQA dataset
        :param system: conversation turn of the system from the CSQA dataset
        :param use_ids: if True, replace labels with respective entity IDs before transformation (and back after trans)
        :return: declarative string
        """
        user_utterance = user['utterance']
        user_ents = user['entities_in_utterance']
        system_utterance = system['utterance']
        system_ents = system['entities_in_utterance']

        # replace all labels with entity ids
        inverse_map = {}
        if use_ids:
            user_utterance, user_inverse_map = self._replace_labels_with_id(user_utterance, user_ents)
            system_utterance, system_inverse_map = self._replace_labels_with_id(system_utterance, system_ents)
            inverse_map = {**user_inverse_map, **system_inverse_map}

        # Tranform user+system utterances into declarative statements using T5-QA2D
        # TODO: Tweak for different question types
        # TODO: Simple (Direct)
        #   Simple Question
        #   Simple Question|Single Entity
        #   Simple Question|Mult. Entity|Indirect
        #       07/12/2022 11:22:25 PM __main__     INFO     qa_str in transform_utterances: Which people are the life partner of Q63749 and Q213671 ? Q288703, Q813294, Q542719
        #       07/12/2022 11:22:25 PM __main__     INFO     declarative_str in transform_utterances: Q288703, Q813294, Q542719 are the life partner of
        #       07/12/2022 11:22:25 PM __main__     INFO     qa_str in transform_utterances: Which person have that architectural structure as their work location ? Q91103
        #       statement in __main__: Landgravine Caroline Louise of Hesse-Darmstadt, Beatrix of Julich-Berg, Louise Caroline of Hochberg are the life partner of

        # TODO: Simple (Coreference)
        #   Simple Question|Single Entity|Indirect
        #   Simple Question|Mult. Entity
        #       07/12/2022 11:22:27 PM __main__     INFO     qa_str in transform_utterances: Which political territories are bordered by those ones ? Q183 for 1st, 2nd, Q834010 for 3rd
        #       07/12/2022 11:22:27 PM __main__     INFO     declarative_str in transform_utterances: Q183 for 1st, 2nd, Q834010 for 3rd are
        #       07/12/2022 11:22:27 PM __main__     INFO     qa_str in transform_utterances: And what about Q19893635? Q21
        #       statement in __main__: Germany for 1st, 2nd, Villar del Rio for 3rd are

        # TODO: Simple (Ellipsis)
        #   only subject is changed, parent and predicate remains same
        #   Incomplete|object parent is changed, subject and predicate remain same
        # DONE: Solve this problem: Villar del Río for 3rd -> Villar del Ro for 3rd
        print(f'utterances in transform_utterances: U: {user_utterance} S: {system_utterance}')
        declarative_str = self.qa2d_model.infer_one(user_utterance, system_utterance)
        print(f'declarative_str in transform_utterances: {declarative_str}')

        # replace entity ids back with labels
        for eid, lab in inverse_map.items():
            declarative_str = declarative_str.replace(eid, lab)

        return declarative_str

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

    # DONE: only take Simple Question types. Those we will need to transform
    #  TODO: how about the questions around them?

    op = ESActionOperator(CLIENT)
    transformer = QA2DT5()  # or QuestionConverter3B
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
                print(f"USER: {entry_user['description']}, {entry_user['entities_in_utterance']}, {entry_user['relations']}, {entry_user['utterance']}")
                print(f"SYSTEM: {entry_system['entities_in_utterance']} {entry_system['utterance']}")
                print(f"active_set: {entry_system['active_set']}")


                # 1) TRANSFORM active_set field
                new_active_set = builder.build_active_set(entry_user, entry_system)
                print(f'new_active_set: {new_active_set}')

                # 2) TRANSFORM utterances to statements  # TODO: still needs a lot of tweaking
                statement = builder.transorm_utterances(entry_user, entry_system)
                print(f'statement: {statement}')
                print(f"".center(50, "-"), end='\n\n')

                # 3) TRANSFORM all other fields in conversation turns TODO: implement

            # conversation types to tweak:
            # and how?
