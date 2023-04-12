import json
import logging
from functools import partial
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from text_generation.label_replacement import LabelReplacer
from text_generation.qa2d import get_model, QA2DModel
from action_executor.actions import ESActionOperator
from helpers import connect_to_elasticsearch, setup_logger
from constants import ROOT_PATH, QA2DModelChoices, RepresentEntityLabelAs

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

# Simple (Coreference)
#   Simple Question|Single Entity|Indirect
#   Simple Question|Mult. Entity

# Simple (Ellipsis)
#   only subject is changed, parent and predicate remains same
#   Incomplete|object parent is changed, subject and predicate remain same

CLIENT = connect_to_elasticsearch()
LOGGER = setup_logger(__name__, loglevel=logging.INFO)


class QA2DCSQARunner:

    def __init__(self, operator: ESActionOperator, qa2d_model: QA2DModel, label_replacer: LabelReplacer):
        self.op = operator
        self.qa2d_model = qa2d_model
        self.lbl_repl = label_replacer

    def transorm_utterances(self, user: dict[list[str] or str], system: dict[list[str] or str],
                            labels_as: RepresentEntityLabelAs) -> str:
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

        LOGGER.debug(f'utterances in transform_utterances: U: {user_utterance} S: {system_utterance}')
        qa_string = self.qa2d_model.preprocess_and_combine(user_utterance, system_utterance)
        LOGGER.debug(f"qa_string in infer_one before replace: {qa_string}")
        qa_entities = [*user_ents, *system_ents]
        qa_string, inverse_map = self.lbl_repl.labs2subs(qa_string, qa_entities, labels_as)
        LOGGER.debug(f"qa_string in infer_one after replace: {qa_string}")
        declarative_str = self.qa2d_model.infer_one(qa_string)
        LOGGER.debug(f'declarative_str in transform_utterances: {declarative_str}')

        # replace entity ids back with labels
        declarative_str = self.lbl_repl.subs2labs(declarative_str, inverse_map)

        return declarative_str


def _one_file_qa2d(pth: Path, builder: QA2DCSQARunner, labels_as: RepresentEntityLabelAs):
    with open(pth, encoding='utf8') as json_file:
        conversation = json.load(json_file)

    statements = []
    for i in range(len(conversation) // 2):
        entry_user = conversation[2 * i]  # USER
        entry_system = conversation[2 * i + 1]  # SYSTEM

        # 2) TRANSFORM utterances to statements  # TODO: still needs a lot of tweaking
        statements.append(builder.transorm_utterances(entry_user, entry_system, labels_as=labels_as).strip()+"\n")
    return statements


def main(model_choice: QA2DModelChoices, read_folder: Path, write_folder: Path, labels_as: RepresentEntityLabelAs, partition: str, max_workers=1):
    # read data and create csqa data dict
    # dict: partition -> folder -> file -> conversation
    data_folder = read_folder.joinpath(partition)
    csqa_files = list(data_folder.glob('**/*.json'))
    LOGGER.info(f'Reading folders for partition {partition}')

    op = ESActionOperator(CLIENT)
    transformer = get_model(model_choice)
    labeler = LabelReplacer(op)
    builder = QA2DCSQARunner(op, transformer, labeler)

    write_folder.mkdir(parents=True, exist_ok=True)

    with write_folder.joinpath(f'hypothesis_{partition}-{model_choice.name}-{labels_as.name}.txt').open('w', encoding="utf8") as f_hypothesis:
        for pth in tqdm(csqa_files):
            statements = _one_file_qa2d(pth, builder, labels_as)
            f_hypothesis.writelines(statements)

        # with ThreadPoolExecutor(max_workers=max_workers) as executor:
        #     statements = list(tqdm(executor.map(lambda pth: _one_file_qa2d(pth, builder, labels_as), csqa_files)))
        # f_hypothesis.writelines([st for stat in statements for st in stat])


if __name__ == "__main__":
    # NOTE: !!!Possible bug: implementing INDEX structure, we presumed that s-r-o automatically means o-r-s is also true.
    # options
    read_folder = ROOT_PATH.joinpath('data/simple_direct/')  # 'folder to read conversations'
    write_folder = ROOT_PATH.joinpath('data/qa2d/')
    partitions = ('test', 'train', 'val')[:2]  # 'test', 'train', 'val', ''
    model_choices = [QA2DModelChoices.T5_SMALL,
                     QA2DModelChoices.T5_BASE,
                     # QA2DModelChoices.T5_3B,  # TODO: run on CPU or RTX3080 later
                     QA2DModelChoices.T5_WHYN]
    # TODO: Run T5_SMALL, T5_BASE on train partition after the previous run finishes

    represent_entity_labels_as = RepresentEntityLabelAs.LABEL
    for partition in partitions:
        main_part = partial(main, read_folder=read_folder, write_folder=write_folder,
                            labels_as=represent_entity_labels_as, partition=partition)
        for model_choice in model_choices:
            LOGGER.info(f"model_choice: {model_choice}")
            main_part(model_choice)
        # with ThreadPoolExecutor(max_workers=5) as executor:
        #     executor.map(main_part, list(QA2DModelChoices))
        # for model_choice in QA2DModelChoices:
        #     LOGGER.info(f"model_choice: {model_choice}")
        #     main(model_choice, read_folder, write_folder, represent_entity_labels_as, partition)
