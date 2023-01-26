import json
import logging
import re
from pathlib import Path

from lab.label_replacement import LabelReplacer
from annotate_csqa.qa2d import get_model, QA2DModel
from action_executor.actions import ESActionOperator
from helpers import connect_to_elasticsearch, setup_logger
from constants import ROOT_PATH, QA2DModelChoices, RepresentEntityLabelAs
from args import parse_and_get_args
args = parse_and_get_args()

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


class CSQAInsertBuilder:

    def __init__(self, operator: ESActionOperator, qa2d_model: QA2DModel, label_replacer: LabelReplacer):
        self.op = operator
        self.qa2d_model = qa2d_model
        self.lbl_repl = label_replacer

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
        qa_string, inverse_map = self.lbl_repl.labs2subs(qa_string, qa_entities, labels_as)
        LOGGER.info(f"qa_string in infer_one after replace: {qa_string}")
        declarative_str = self.qa2d_model.infer_one(qa_string)
        LOGGER.info(f'declarative_str in transform_utterances: {declarative_str}')

        # replace entity ids back with labels
        declarative_str = self.lbl_repl.subs2labs(declarative_str, inverse_map)

        return declarative_str

    def transform_fields(self):
        pass


def main(model_choice: QA2DModelChoices, labels_as: RepresentEntityLabelAs):
    # read data and create csqa data dict
    # dict: partition -> folder -> file -> conversation
    data_folder = Path(f'{ROOT_PATH}{args.read_folder}/{args.partition}')
    # csqa_files = data_folder.glob('**/QA*.json')
    csqa_files = data_folder.glob('**/*.json')
    LOGGER.info(f'Reading folders for partition {args.partition}')

    op = ESActionOperator(CLIENT)
    transformer = get_model(model_choice)
    labeler = LabelReplacer(op)
    builder = CSQAInsertBuilder(op, transformer, labeler)

    # TODO: do this better
    with data_folder.joinpath(f'hypothesis_{args.partition}.txt').open('a', encoding="utf8") as f_hypothesis:
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

                # 2.5) EVALUATION: # TODO: Continue here (build hypothesis.txt files) using this
                f_hypothesis.write(statement)
                f_hypothesis.write('\n')

                # 3) TRANSFORM all other fields in conversation turns TODO: implement

                # conversation types to tweak:
                # and how?


if __name__ == "__main__":
    # TODO: !!!Possible bug: implementing INDEX structure, we presumed that s-r-o automatically means o-r-s is also true.
    # options
    args.read_folder = '/data/simple_direct'  # 'folder to read conversations'
    args.partition = 'val'  # 'train', 'test', 'val', ''

    model_choice = QA2DModelChoices.T5_WHYN
    represent_entity_labels_as = RepresentEntityLabelAs.LABEL
    main(model_choice, represent_entity_labels_as)
