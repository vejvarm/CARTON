import json
import logging
import random
from pathlib import Path

from sentence_transformers import SentenceTransformer
from torch import cosine_similarity

from helpers import setup_logger
from dataset import CSQADataset
from annotate_csqa.qa2d import QA2DModel, get_model
from constants import QA2DModelChoices, ROOT_PATH
from args import parse_and_get_args
args = parse_and_get_args()

logging.getLogger('sentence_transformers.SentenceTransformer').setLevel(logging.WARNING)

RESULTS_PATH = Path(ROOT_PATH, args.path_results)
LOGGER = setup_logger(__name__,
                      logging.INFO,
                      handlers=[logging.FileHandler(RESULTS_PATH.joinpath('sts.log'), 'w'),  # TODO: for some reason log is empty!
                                logging.StreamHandler()])

def sample_random_simple_questions(max_files: int | None, num_samples: int | None) -> list[dict]:
    """Load 'max_files' worth of dataset files and randomly sample 'num_samples' Simple Questions from them."""
    inference_data = CSQADataset.get_inference_data(max_files)

    simple_questions = []

    for entry in inference_data:
        if 'Simple Question' not in entry['question_type']:
            continue

        simple_questions.append(entry)
        # print(entry['question'], entry['answer'])

    if num_samples is not None:
        return random.sample(simple_questions, num_samples)
    else:
        return simple_questions


def qa2d(question_list: list[dict], qa2d_model: QA2DModel) -> list[dict]:
    """Add new field 'statement', which is qa2d transformed statement."""

    qa2d_list = []

    for i, dict_qa in enumerate(question_list):
        qa_string = qa2d_model.preprocess_and_combine(dict_qa['question'], dict_qa['answer'])
        statement = qa2d_model.infer_one(qa_string)
        qa2d_list.append({**dict_qa, 'qa_string': qa_string, 'statement': statement})

    return qa2d_list


def compare_qas_with_statements(qa2d_list: list[dict], sts_model: SentenceTransformer):

    qa2d_sts_list = []

    for i, dict_qa2d in enumerate(qa2d_list):
        qa_emb = sts_model.encode(dict_qa2d['question']+dict_qa2d['answer'], convert_to_tensor=True)
        stmt_emb = sts_model.encode(dict_qa2d['statement'], convert_to_tensor=True)

        qa2d_sts_list.append({**dict_qa2d, 'cos_similarity': cosine_similarity(qa_emb, stmt_emb, dim=0).detach().cpu().numpy()})

    return qa2d_sts_list


if __name__ == "__main__":
    # randomly sample Simple Questions from the test part of dataset
    random.seed(42)
    max_files = None
    num_question_samples = None
    random_simple_question_list = sample_random_simple_questions(max_files, num_question_samples)

    q_count = len(random_simple_question_list)
    LOGGER.info(f"Num simple questions: {q_count}")

    # run QA2D on simple_question_list
    qa2d_choices = QA2DModelChoices
    cos_similarity_means = {'count': q_count}
    for choice in qa2d_choices:
        qa2d_model = get_model(choice)
        random_qa2d_list = qa2d(random_simple_question_list, qa2d_model)
        # print(random_qa2d_list)

        # compare sentences
        sts_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        random_qa2d_sts_list = compare_qas_with_statements(random_qa2d_list, sts_model)

        cos_sum = 0.
        for entry in random_qa2d_sts_list:
            cos_sum += entry['cos_similarity']
            # print(f"cos: {entry['cos_similarity']:.3f} | qa: {entry['qa_string']} | stat: {entry['statement']}")

        cos_similarity_means[choice.name] = cos_sum/len(random_qa2d_sts_list)
        LOGGER.info(f"{choice.name}:\t {cos_similarity_means[choice.name]:.3f}")

    with RESULTS_PATH.joinpath('sts_cos_similarity.json').open('w') as f:
        LOGGER.info(cos_similarity_means)
        json.dump(cos_similarity_means, f, indent=4)
