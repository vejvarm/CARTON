import os
import torch
from enum import Enum, auto
from pathlib import Path

# set root path
ROOT_PATH = Path(os.path.dirname(__file__))

LOGS_PATH = ROOT_PATH.joinpath("logs")  # default path for logfiles
LOGS_PATH.mkdir(parents=True, exist_ok=True)

ENTITY_SEPARATOR_FOR_NEW_DATASETS = "<&SEP>"  # legacy " | "

# model name
MODEL_NAME = 'CARTONNER'

# Elasticsearch
INDEX_ROOT = 'csqa_wikidata'


class ElasticIndices(Enum):
    ENT = f'{INDEX_ROOT}_ent'
    ENT_FULL = f'{INDEX_ROOT}_ent_full'
    REL = f'{INDEX_ROOT}_rel'
    RDF = f'{INDEX_ROOT}_rdf'
    RDF_FULL = f'{INDEX_ROOT}_rdf_full'


class KGType(Enum):
    MEMORY = 'memory'
    ZODB = 'zodb'
    ELASTICSEARCH = 'elasticsearch'


class Task(Enum):
    MULTITASK = 'multitask'
    LOGICAL_FORM = 'logical_form'
    PREDICATE_POINTER = 'predicate_pointer'
    TYPE_POINTER = 'type_pointer'
    NER = 'ner'
    COREF = 'coref'


class QuestionTypes(Enum):
    ALL = 'all'
    SIMPLE_DIRECT = 'Simple Question (Direct)'
    SIMPLE_COREFERENCED = 'Simple Question (Coreferenced)'
    SIMPLE_ELLIPSIS = 'Simple Question (Ellipsis)'
    COMPARATIVE = 'Comparative Reasoning (All)'
    LOGICAL = 'Logical Reasoning (All)'
    QUANTITATIVE = 'Quantitative Reasoning (All)'
    VERIFICATION = 'Verification (Boolean) (All)'
    QUANTITATIVE_COUNT = 'Quantitative Reasoning (Count) (All)'
    COMPARATIVE_COUNT = 'Comparative Reasoning (Count) (All)'
    CLARIFICATION = 'Clarification'


class InferencePartition(Enum):
    TEST = 'test'
    VAL = 'val'


class Passwords(Enum):
    NOTEBOOK = 'hZiYNU+ye9izCApoff-v'
    FREYA = '1jceIiR5k6JlmSyDpNwK'


class QA2DModelChoices(Enum):
    T5_SMALL = 'domenicrosati/QA2D-t5-small'  # T5-Small model fine-tuned on the QA2D dataset
    T5_BASE = 'domenicrosati/QA2D-t5-base'    # T5-Base model fine-tuned on the QA2D dataset
    T5_3B = 'domenicrosati/question_converter-3b'  # T5-3B model fine-tuned on the QA2D dataset TODO: doesn't fit on GPU, so run with CPU or get RTX3080
    T5_WHYN = 'Farnazgh/QA2D'  # T5-Large model fine-tuned on QA2D+YesNo type questions from SAMSum corpus


class RepresentEntityLabelAs(Enum):
    LABEL = auto()
    ENTITY_ID = auto()
    PLACEHOLDER = auto()
    PLACEHOLDER_NAMES = auto()
    GROUP = auto()
    # TYPE_ID = auto()  # TODO: Implement


# define device
CUDA = 'cuda'
CPU = 'cpu'
DEVICE = torch.device(CUDA if torch.cuda.is_available() else CPU)

# fields
ID = 'id'
INPUT = 'input'
LOGICAL_FORM = 'logical_form'
PREDICATE_POINTER = 'predicate_pointer'
TYPE_POINTER = 'type_pointer'
ENTITY_POINTER = 'entity_pointer'
NER = 'ner'
COREF = 'coref'
MULTITASK = 'multitask'

# helper tokens
START_TOKEN = '[START]'
END_TOKEN = '[END]'
CTX_TOKEN = '[CTX]'
PAD_TOKEN = '[PAD]'
UNK_TOKEN = '[UNK]'
SEP_TOKEN = '[SEP]'
NA_TOKEN = 'NA'

GOLD = 'gold'
LABEL = 'label'

# ner tag
B = 'B'
I = 'I'
O = 'O'

# model
ENCODER_OUT = 'encoder_out'
DECODER_OUT = 'decoder_out'
DECODER_H = 'decoder_h'

# training
EPOCH = 'epoch'
STATE_DICT = 'state_dict'
BEST_VAL = 'best_val'
OPTIMIZER = 'optimizer'
CURR_VAL = 'curr_val'

# question types
TOTAL = 'total'
OVERALL = 'Overall'
SIMPLE_DIRECT = QuestionTypes.SIMPLE_DIRECT.value
SIMPLE_COREFERENCED = QuestionTypes.SIMPLE_COREFERENCED.value
SIMPLE_ELLIPSIS = QuestionTypes.SIMPLE_ELLIPSIS.value
COMPARATIVE = QuestionTypes.COMPARATIVE.value
LOGICAL = QuestionTypes.LOGICAL.value
QUANTITATIVE = QuestionTypes.QUANTITATIVE.value
VERIFICATION = QuestionTypes.VERIFICATION.value
QUANTITATIVE_COUNT = QuestionTypes.QUANTITATIVE_COUNT.value
COMPARATIVE_COUNT = QuestionTypes.COMPARATIVE_COUNT.value
CLARIFICATION = QuestionTypes.CLARIFICATION.value

ALL_QUESTION_TYPES = tuple(qt.value for qt in QuestionTypes)

# action related
ENTITY = 'entity'
RELATION = 'relation'
TYPE = 'type'
VALUE = 'value'
PREV_ANSWER = 'prev_answer'
ACTION = 'action'

# other
QUESTION_TYPE = 'question_type'
IS_CORRECT = 'is_correct'
QUESTION = 'question'
ANSWER = 'answer'
ACTIONS = 'actions'
GOLD_ACTIONS = 'gold_actions'
RESULTS = 'results'
PREV_RESULTS = 'prev_results'
CONTEXT_QUESTION = 'context_question'
CONTEXT_ENTITIES = 'context_entities'
BERT_BASE_UNCASED = 'bert-base-uncased'
