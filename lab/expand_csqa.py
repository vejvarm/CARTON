import json
import logging
import re
from pathlib import Path

import pandas as pd
from unidecode import unidecode
from typing import Protocol
from abc import ABC, abstractmethod
from enum import Enum, auto

from constants import args, ROOT_PATH, ENTITY, TYPE, RELATION

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

from action_executor.actions import ESActionOperator
from helpers import connect_to_elasticsearch, setup_logger

# TODOlist
# Question Answer processing during QA2D transformation:
#   DONE: Use plain question and answer strings (use_ent_id_in_transformations = False)
#   DONE: Use entity identifiers (id) in place of labels (use_ent_id_in_transformations = True)
#   DONE: Make entitiy ids lowercased (e.g. Q1235 -> q1235)
#   DONE: use placeholder names for entities (instead of Q1234, use e.g. f'entity{i}' )
#   DONE: use common english names as placeholders for entities (insead of Q1234, use e.g. Mary)
#   DONE: use regexps for substituting sequences of entities (e.g. Q1, Q2 and Q3 -> entities1) NOTE: WORKS PRETTY GOOD
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
LOGGER = setup_logger(__name__, loglevel=logging.WARNING)


class QA2DModelChoices(Enum):
    QA2DT5_SMALL = 'domenicrosati/QA2D-t5-small'
    QA2DT5_BASE = 'domenicrosati/QA2D-t5-base'    # T5
    QC3B = 'domenicrosati/question_converter-3b'  # T5-3B model fine tuned on the QA2D dataset
    GhasQA2D = 'Farnazgh/QA2D'


class RepresentEntityLabelAs(Enum):
    LABEL = auto()
    ENTITY_ID = auto()
    PLACEHOLDER = auto()
    PLACEHOLDER_NAMES = auto()
    GROUP = auto()  # TODO: Implement
    # TYPE_ID = auto()  # TODO: Implement


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


class GhasPreprocessor(Preprocessor):
    SEP = " "

    @classmethod
    def combine_qa(cls, question: str, answer: str) -> str:
        q = question.replace(" ?", "?").strip()
        a = answer.strip()
        return f"q: {q}{cls.SEP}a: {a}"


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
        elif model_type == QA2DModelChoices.GhasQA2D:
            return GhasPreprocessor
        else:
            raise NotImplementedError(
                f'Chosen model type ({model_type}) is not supported. Refer to QA2DModelChoices class.')

    def preprocess_and_combine(self, question: str, answer: str) -> str:
        return self.preprocessor.combine_qa(question, answer)

    def infer_one(self, qa_string: str) -> str:
        input_ids = self.tokenizer(qa_string, return_tensors="pt").input_ids
        LOGGER.debug(f"input_ids in infer_one: ({input_ids.shape}) {input_ids}")

        outputs = self.model.generate(input_ids)
        LOGGER.debug(f"outputs in infer_one: ({outputs.shape}) {outputs}")

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


class GhasQA2DModel(QA2DModel):

    def infer_one(self, qa_string: str, max_length=150) -> str:
        input_ids = self.tokenizer.encode(qa_string, return_tensors="pt", add_special_tokens=True)
        LOGGER.debug(f"input_ids in infer_one: ({input_ids.shape}) {input_ids}")

        outputs = self.model.generate(input_ids=input_ids, num_beams=2, max_length=max_length, early_stopping=True)
        LOGGER.debug(f"outputs in infer_one: ({outputs.shape}) {outputs}")

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)


def get_model(model_choice: QA2DModelChoices) -> QA2DModel or GhasQA2DModel:
    if model_choice not in QA2DModelChoices:
        raise NotImplementedError()

    if model_choice == QA2DModelChoices.GhasQA2D:
        return GhasQA2DModel(model_choice)
    else:
        return QA2DModel(model_choice)


class CSQAInsertBuilder:

    def __init__(self, operator: ESActionOperator, qa2d_model: QA2DModel):
        self.op = operator
        self.qa2d_model = qa2d_model

        self.placeholder_names = ["Mary", "Anna", "Emma", "Elizabeth", "Minnie", "Margaret", "Ida", "Alice", "Bertha", "Sarah", "Annie", "Clara", "Ella", "Florence", "Cora", "Martha", "Laura", "Nellie", "Grace", "Carrie", "Maude", "Mabel", "Bessie", "Jennie", "Gertrude", "Julia", "Hattie", "Edith", "Mattie", "Rose", "Catherine", "Lillian", "Ada", "Lillie", "Helen", "Jessie", "Louise", "Ethel", "Lula", "Myrtle", "Eva", "Frances", "Lena", "Lucy", "Edna", "Maggie", "Pearl", "Daisy", "Fannie", "Josephine", "Dora", "Rosa", "Katherine", "Agnes", "Marie", "Nora", "May", "Mamie", "Blanche", "Stella", "Ellen", "Nancy", "Effie", "Sallie", "Nettie", "Della", "Lizzie", "Flora", "Susie", "Maud", "Mae", "Etta", "Harriet", "Sadie", "Caroline", "Katie", "Lydia", "Elsie", "Kate", "Susan", "Mollie", "Alma", "Addie", "Georgia", "Eliza", "Lulu", "Nannie", "Lottie", "Amanda", "Belle", "Charlotte", "Rebecca", "Ruth", "Viola", "Olive", "Amelia", "Hannah", "Jane", "Virginia"]

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

    def replace_labels(self, utterance: str, entities: list[str], labels_as: RepresentEntityLabelAs) -> tuple[str, dict]:
        if labels_as not in RepresentEntityLabelAs:
            raise NotImplementedError(f'Chosen RepresentEntityLabeAs Enum ({labels_as}) is not supported.')

        if labels_as == RepresentEntityLabelAs.LABEL:
            return unidecode(utterance), dict()

        if labels_as == RepresentEntityLabelAs.GROUP:
            return self._replace_labels_by_groups(utterance, entities)

        return self._replace_labels_in_utterance(utterance, entities, labels_as)

    def _replace_labels_by_groups(self, utterance: str, entities: list[str]) -> tuple[str, dict]:
        utterance, inverse_map = self._replace_labels_in_utterance(utterance, entities, RepresentEntityLabelAs.ENTITY_ID)

        group_inverse_map = dict()
        pattern = r"Q\d+(?:\s?(?:,|and)\s?Q\d+)*"
        matches = re.findall(pattern, utterance, flags=re.IGNORECASE)
        for j, match in enumerate(matches):
            repl = f"group{j}"
            utterance = utterance.replace(match, repl)
            group_inverse_map[repl] = self._replace_ids_with_labels(match, inverse_map)

        return utterance, group_inverse_map

    def _replace_labels_in_utterance(self, utterance: str, entities: list[str], labels_as: RepresentEntityLabelAs) -> tuple[str, dict]:
        # Use a dictionary to map the values of the labels_as parameter to the appropriate replacement string
        label_replacements = {
            RepresentEntityLabelAs.ENTITY_ID: lambda e, i: e.lower(),
            RepresentEntityLabelAs.PLACEHOLDER: lambda e, i: f"entity{i}",
            RepresentEntityLabelAs.PLACEHOLDER_NAMES: lambda e, i: self.placeholder_names[i],
        }

        if labels_as not in label_replacements.keys():
            raise NotImplementedError(f'Chosen RepresentEntityLabeAs Enum ({labels_as}) is not supported.')

        utterance = unidecode(utterance)
        inverse_map = dict()

        for idx, ent in enumerate(entities):
            label = self.op.get_label(ent)
            replacement = label_replacements[labels_as](ent, idx)
            utterance = utterance.replace(label, replacement)
            inverse_map[replacement] = label

        return utterance, inverse_map

    @staticmethod
    def _replace_ids_with_labels(utterance: str, inverse_map: dict[str:str]):
        for eid, lab in inverse_map.items():
            utterance = re.sub(eid, lab, utterance, flags=re.IGNORECASE)

        return utterance

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

        LOGGER.info(f'utterances in transform_utterances: U: {user_utterance} S: {system_utterance}')
        qa_string = self.qa2d_model.preprocess_and_combine(user_utterance, system_utterance)
        LOGGER.info(f"qa_string in infer_one before replace: {qa_string}")
        qa_entities = [*user_ents, *system_ents]
        qa_string, inverse_map = self.replace_labels(qa_string, qa_entities, labels_as)
        LOGGER.info(f"qa_string in infer_one after replace: {qa_string}")
        declarative_str = self.qa2d_model.infer_one(qa_string)
        LOGGER.info(f'declarative_str in transform_utterances: {declarative_str}')

        # replace entity ids back with labels
        declarative_str = self._replace_ids_with_labels(declarative_str, inverse_map)

        return declarative_str

    def transform_fields(self):
        pass


def main(model_choice: QA2DModelChoices, labels_as: RepresentEntityLabelAs):
    # read data and create csqa data dict
    # dict: partition -> folder -> file -> conversation
    data_folder = Path(f'{ROOT_PATH}{args.read_folder}/{args.partition}')
    # csqa_files = data_folder.glob('**/QA*.json')
    csqa_files = data_folder.glob('**/d_dataset_like_example_file.json')
    LOGGER.info(f'Reading folders for partition {args.partition}')

    op = ESActionOperator(CLIENT)
    transformer = get_model(model_choice)
    builder = CSQAInsertBuilder(op, transformer)

    for pth in csqa_files:

        with open(pth, encoding='utf8') as json_file:
            conversation = json.load(json_file)

        for i in range(len(conversation) // 2):
            entry_user = conversation[2 * i]  # USER
            entry_system = conversation[2 * i + 1]  # SYSTEM

            if 'Simple' not in entry_user['question-type']:
                continue

            LOGGER.info(
                f"USER: {entry_user['description']}, {entry_user['entities_in_utterance']}, {entry_user['relations']}, {entry_user['utterance']}")
            LOGGER.info(f"SYSTEM: {entry_system['entities_in_utterance']} {entry_system['utterance']}")
            LOGGER.info(f"active_set: {entry_system['active_set']}")

            # 1) TRANSFORM active_set field
            new_active_set = builder.build_active_set(entry_user, entry_system)
            LOGGER.info(f'new_active_set: {new_active_set}')

            # 2) TRANSFORM utterances to statements  # TODO: still needs a lot of tweaking
            statement = builder.transorm_utterances(entry_user, entry_system, labels_as=labels_as)
            LOGGER.info(f'statement: {statement}')
            LOGGER.info(f"".center(50, "-") + "\n\n")

            # 3) TRANSFORM all other fields in conversation turns TODO: implement

            # conversation types to tweak:
            # and how?


def compare_generated_utterances(model_choices: list[QA2DModelChoices] or QA2DModelChoices,
                                 labels_as_list: list[RepresentEntityLabelAs] or RepresentEntityLabelAs):
    data_folder = Path(f'{ROOT_PATH}{args.read_folder}/{args.partition}')
    # csqa_files = data_folder.glob('**/QA*.json')
    LOGGER.info(f'Reading folders for partition {args.partition}')

    op = ESActionOperator(CLIENT)

    results = {}

    for model_choice in model_choices:
        transformer = get_model(model_choice)  # or QuestionConverter3B
        builder = CSQAInsertBuilder(op, transformer)

        results[model_choice.name] = {}

        csqa_files = data_folder.glob('**/d_dataset_like_example_file.json')
        for pth in csqa_files:
            results[model_choice.name][pth.parent.name] = {}

            with open(pth, encoding='utf8') as json_file:
                conversation = json.load(json_file)

            for i in tqdm(range(len(conversation) // 2)):
                entry_user = conversation[2 * i]  # USER
                entry_system = conversation[2 * i + 1]  # SYSTEM

                if 'Simple' not in entry_user['question-type']:
                    continue

                if entry_user['question-type'] not in results[model_choice.name][pth.parent.name].keys():
                    results[model_choice.name][pth.parent.name][entry_user['question-type']] = {}

                if entry_user['description'] not in results[model_choice.name][pth.parent.name][entry_user['question-type']].keys():
                    results[model_choice.name][pth.parent.name][entry_user['question-type']][entry_user['description']] = {
                        'utterances': (entry_user['utterance'], entry_system['utterance']),
                        'entities': (entry_user['entities_in_utterance'], entry_system['entities_in_utterance'])
                    }

                # 2) TRANSFORM utterances to statements
                for labels_as in labels_as_list:
                    statement = builder.transorm_utterances(entry_user, entry_system, labels_as=labels_as)
                    LOGGER.info(f'statement: {statement}')
                    LOGGER.info(f"".center(50, "-") + "\n\n")

                    if labels_as.name not in results[model_choice.name][pth.parent.name][entry_user['question-type']][entry_user['description']]:
                        results[model_choice.name][pth.parent.name][entry_user['question-type']][entry_user['description']][labels_as.name] = {}

                    results[model_choice.name][pth.parent.name][entry_user['question-type']][entry_user['description']][labels_as.name] = {
                        'statement': statement
                    }

        json.dump(results, data_folder.joinpath('utterance_comparison.json').open('w', encoding='utf8'), indent=4)


def make_question_specific_csv_from_utterance_comparison():
    data_folder = Path(f'{ROOT_PATH}{args.read_folder}/{args.partition}')
    # csqa_files = data_folder.glob('**/QA*.json')
    utterance_file = data_folder.joinpath('utterance_comparison.json')
    LOGGER.info(f'Reading file {utterance_file.name} for partition {args.partition}')

    d = json.load(utterance_file.open('r', encoding='utf8'))

    folder = "original"
    qc3b_data = d[QA2DModelChoices.QC3B.name][folder]
    qa2dt5_data = d[QA2DModelChoices.QA2DT5_SMALL.name][folder]

    for q_type in qc3b_data.keys():

        for q_subtype in qc3b_data[q_type].keys():
            index = ['original']
            qc3b_utterances = [' '.join(qc3b_data[q_type][q_subtype][RepresentEntityLabelAs.LABEL]['utterances'])]
            qa2dt5_utterances = [' '.join(qa2dt5_data[q_type][q_subtype][RepresentEntityLabelAs.LABEL]['utterances'])]

            for labels_as_name in qc3b_data[q_type][q_subtype].keys():
                index.append(labels_as_name)
                qc3b_utterances.append(qc3b_data[q_type][q_subtype][labels_as_name]['statement'])
                qa2dt5_utterances.append(qa2dt5_data[q_type][q_subtype][labels_as_name]['statement'])

            df = pd.DataFrame(data=list(zip(qc3b_utterances, qa2dt5_utterances)), index=[index], columns=[QA2DModelChoices.QC3B.name, QA2DModelChoices.QA2DT5_SMALL.name])
            df.to_csv(data_folder.joinpath(f'f{q_type}_{q_subtype}.csv'))


if __name__ == "__main__":
    # options
    args.read_folder = '/data'  # 'folder to read conversations'
    args.partition = ''  # 'train', 'test', 'val', ''

    model_choice = QA2DModelChoices.GhasQA2D
    represent_entity_labels_as = RepresentEntityLabelAs.GROUP
    main(model_choice, represent_entity_labels_as)

    # model_choices = QA2DModelChoices
    # labels_as_list = RepresentEntityLabelAs
    # compare_generated_utterances(model_choices, labels_as_list)

    # make_question_specific_csv_from_utterance_comparison()
