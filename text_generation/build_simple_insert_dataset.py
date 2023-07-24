from path_utils import add_project_root_to_path
add_project_root_to_path()

import json
import logging
import re
from pathlib import Path

from text_generation.label_replacement import LabelReplacer
from text_generation.qa2d import get_model, QA2DModel
from action_executor.actions import ESActionOperator
from helpers import connect_to_elasticsearch, setup_logger
from constants import QA2DModelChoices, RepresentEntityLabelAs, ENTITY_SEPARATOR_FOR_NEW_DATASETS
from args import parse_and_get_args, get_parser
from tqdm import tqdm
from ordered_set import OrderedSet
args = parse_and_get_args()

parser = get_parser()

parser.add_argument('--partition', default='val', choices=['val', 'test', 'train'], type=str, help='partition to preprocess')
parser.add_argument('--read_folder', default='data/original_w_turn_pos', help='folder to read source csqa json files')
parser.add_argument('--write_folder', default='data/original_simple_direct_csqa', help='folder to write the transformed conversations')
args = parser.parse_args()

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

    @staticmethod
    def _parse_to_active_set(sid: str, rid: str, oid: str, insert_flag=False) -> str:
        """ parse given triple to active_set entry string format"""
        active_set = f"({sid},{rid},{oid})"
        if insert_flag:
            active_set = "i" + active_set
        return active_set

    def build_active_set(self, user: dict[list[str] or str], system: dict[list[str] or str]):
        user_ents = user['entities_in_utterance']
        system_ents = system['entities_in_utterance']
        rels = user['relations']

        # in case of Ellipsis question
        if not rels:
            rels = OrderedSet()
            rels.update(*[re.findall(r'P\d+', entry) for entry in system['active_set']])
            rels = list(rels)

        active_set = OrderedSet()
        _all_possible_ids = OrderedSet()
        for ue in user_ents:
            for se in system_ents:
                for r in rels:
                    _all_possible_ids.update([f'{ue}{r}{se}', f'{se}{r}{ue}'])

        for _id in _all_possible_ids:
            rdf = self.op.get_rdf(_id)
            if rdf:
                act_set_entry = self._parse_to_active_set(rdf['sid'], rdf['rid'], rdf['oid'], insert_flag=True)
                active_set.add(act_set_entry)

        LOGGER.debug(f"active_set@build_active_set: {active_set}")
        return list(active_set)

    def transform_utterances(self, user: dict[list[str] or str], system: dict[list[str] or str], labels_as: RepresentEntityLabelAs) -> str:
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

        LOGGER.debug(f'declarative_str@transform_utterances: {declarative_str}')
        return declarative_str

    def transform_fields(self, user: dict[list[str] or str], system: dict[list[str] or str]) -> tuple[dict[list[str] or str], dict[list[str] or str]]:
        # Transform user and system utterances into declarative statements
        declarative_str = self.transform_utterances(user, system, labels_as=RepresentEntityLabelAs.LABEL)
        active_set = self.build_active_set(user, system)
        LOGGER.debug(f"active_set@transform_fields: {active_set}")

        # Update the fields for the user and system turns
        user_new = user.copy()
        user_new["question-type"] = user_new["question-type"].replace("Question", "Insert")
        if "description" in user_new.keys():
            user_new["description"] = user_new["description"].replace("Question", "Insert")
        else:
            user_new["description"] = "Simple Insert"
        user_new["utterance"] = declarative_str
        user_new["entities_in_utterance"] = list(OrderedSet(user_new["entities_in_utterance"] + system["entities_in_utterance"]))

        system_new = system.copy()
        system_new["all_entities"] = user_new["entities_in_utterance"]
        system_new["entities_in_utterance"] = user_new["entities_in_utterance"]
        system_new["utterance"] = self.create_label_sequence_from_active_set(active_set)
        system_new["active_set"] = active_set
        system_new["table_format"] = self.update_table_format(active_set)

        return user_new, system_new

    def create_label_sequence_from_active_set(self, active_set):
        labels = []
        for entry in active_set:
            # Extract subject, predicate, and object IDs
            match = re.match(r'i\((Q\d+),(P\d+),(Q\d+)\)', entry)
            if match:
                sid, pid, oid = match.groups()

                # Get labels for subject, predicate, and object
                s_label = self.op.get_entity_label(sid)
                p_label = self.op.get_relation_label(pid)
                o_label = self.op.get_entity_label(oid)

                # Concatenate labels
                labels.extend([s_label, p_label, o_label])

        return f" {ENTITY_SEPARATOR_FOR_NEW_DATASETS} ".join(labels)

    def update_table_format(self, active_set):
        # Create a new table format based on user and system turns following v1 of table format
        table_format = []

        for entry in active_set:
            # Extract subject, predicate, and object IDs
            match = re.match(r'i\((Q\d+),(P\d+),(Q\d+)\)', entry)
            if match:
                sid, pid, oid = match.groups()

                # Get labels for subject, predicate, and object
                s_label = self.op.get_entity_label(sid)
                p_label = self.op.get_relation_label(pid)
                o_label = self.op.get_entity_label(oid)

                table_format.extend([["name", s_label.split(" ")], [p_label, o_label.split(" ")]])

        LOGGER.debug(f"table_format@update_table_format: {table_format}")
        return table_format


def main(model_choice: QA2DModelChoices, partition: str):
    # read data and create csqa data dict
    # dict: partition -> folder -> file -> conversation
    read_folder = Path(args.read_folder)
    read_split_folder = read_folder.joinpath(partition)
    if not read_folder.exists():
        raise FileNotFoundError(f"The specified folder at path '{read_folder}' doesn't exist. Check --read_folder arg.")
    if not read_split_folder.exists():
        raise FileNotFoundError(f"The specified partition '{partition}' doesn't exist in '{read_folder}'. Check --partition arg.")
    csqa_files = list(read_split_folder.glob('**/QA_*.json'))
    LOGGER.info(f'Reading folders for partition {partition}')

    write_folder = Path(args.write_folder)
    write_split_folder = write_folder.joinpath(partition)
    write_split_folder.mkdir(parents=True, exist_ok=True)
    LOGGER.info(f'Making write folder for partition {partition} at {write_split_folder}')

    op = ESActionOperator(CLIENT)
    transformer = get_model(model_choice)
    labeler = LabelReplacer(op)
    builder = CSQAInsertBuilder(op, transformer, labeler)

    with write_folder.joinpath(f'hypothesis_{partition}.txt').open('w', encoding="utf8") as f_hypothesis:
        for pth in tqdm(csqa_files):
            new_conversation = []
            with open(pth, encoding='utf8') as json_file:
                conversation = json.load(json_file)

            for i in range(len(conversation) // 2):
                entry_user = conversation[2 * i]  # USER
                entry_system = conversation[2 * i + 1]  # SYSTEM

                if 'Simple Question (Direct)' not in entry_user['question-type']:
                    continue

                LOGGER.debug(f" OLD ({pth.parent.name}/{pth.name})".center(50, "-"))
                LOGGER.debug(
                    f"USER: {entry_user['entities_in_utterance']}, {entry_user['relations']}, {entry_user['utterance']}")
                LOGGER.debug(f"SYSTEM: {entry_system['entities_in_utterance']} {entry_system['utterance']}")

                # do the transformation
                new_user, new_system = builder.transform_fields(entry_user, entry_system)

                LOGGER.debug(f" NEW ".center(50, "-"))
                LOGGER.debug(
                    f"USER: {new_user['entities_in_utterance']}, {new_user['relations']}, {new_user['utterance']}")
                LOGGER.debug(f"SYSTEM: {new_system['entities_in_utterance']} {new_system['utterance']}")

                # save declarative statement to sepparate hypothesis_[partition].txt file
                f_hypothesis.write(new_user['utterance']+'\n')

                new_conversation.extend([new_user, new_system])

            # save new conversation json file into the write_split_folder
            relative_path = pth.relative_to(read_split_folder)
            out_file_path = write_split_folder.joinpath(relative_path)
            out_file_path.parent.mkdir(parents=True, exist_ok=True)
            json.dump(new_conversation, out_file_path.open("w"), indent=4)


if __name__ == "__main__":
    model_choice = QA2DModelChoices.T5_WHYN
    # represent_entity_labels_as = RepresentEntityLabelAs.LABEL
    main(model_choice, args.partition)

