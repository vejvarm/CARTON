import os
import torch
from pathlib import Path
from args import get_parser, QuestionTypes

# set root path
ROOT_PATH = Path(os.path.dirname(__file__))

# read parser
parser = get_parser()
args = parser.parse_args(args=[])

# model name
MODEL_NAME = 'CARTON'

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
