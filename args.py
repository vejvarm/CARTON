import argparse
from enum import Enum

INDEX_ROOT = 'csqa_wikidata'


class Task(Enum):
    MULTITASK = 'multitask'
    LOGICAL_FORM = 'logical_form'
    PREDICATE_POINTER = 'predicate_pointer'
    TYPE_POINTER = 'type_pointer'
    NER = 'ner'
    COREF = 'coref'


class QuestionTypes(Enum):
    ALL = 'all'
    CLARIFICATION = 'Clarification'
    COMPARATIVE = 'Comparative Reasoning (All)'
    LOGICAL = 'Logical Reasoning (All)'
    QUANTITATIVE = 'Quantitative Reasoning (All)'
    SIMPLE_COREF = 'Simple Question (Coreferenced)'
    SIMPLE_DIRECT = 'Simple Question (Direct)'
    SIMPLE_ELLIPSIS = 'Simple Question (Ellipsis)'
    VERIFICATION = 'Verification (Boolean) (All)'
    QUANTITATIVE_COUNT = 'Quantitative Reasoning (Count) (All)'
    COMPARATIVE_COUNT = 'Comparative Reasoning (Count) (All)'


class InferencePartition(Enum):
    TEST = 'test'
    VAL = 'val'


class Passwords(Enum):
    NOTEBOOK = 'hZiYNU+ye9izCApoff-v'
    FREYA = '1jceIiR5k6JlmSyDpNwK'


def get_parser():
    parser = argparse.ArgumentParser(description='CARTON')

    # general
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--cuda_device', default=0, type=int)

    # data
    parser.add_argument('--data_path', default='/data/final/csqa')
    parser.add_argument('--embedding_path', default='/knowledge_graph/entity_embeddings.json')
    parser.add_argument('--kg_path', default='./knowledge_graph/Wikidata.fs')

    # experiments
    parser.add_argument('--snapshots', default='experiments/snapshots', type=str)
    parser.add_argument('--path_results', default='experiments/results', type=str)
    parser.add_argument('--path_error_analysis', default='experiments/error_analysis', type=str)
    parser.add_argument('--path_inference', default='experiments/inference', type=str)

    # task
    parser.add_argument('--task', default=Task.MULTITASK.value, choices=[tsk.value for tsk in Task], type=str)

    # model
    parser.add_argument('--emb_dim', default=300, type=int)     # default: 300 (dkg?)  # ANCHOR EMBDIM in model.py if you want to change this
    parser.add_argument('--dropout', default=0.1, type=int)     # default: 0.1 (same)
    parser.add_argument('--heads', default=6, type=int)         # default: 6
    parser.add_argument('--layers', default=2, type=int)        # default: 2
    parser.add_argument('--max_positions', default=1000, type=int)  # ?
    parser.add_argument('--pf_dim', default=1200, type=int)     # default: 300 (Position-wise FF layer dim) # ANCHOR PWFF for more info
    parser.add_argument('--bert_dim', default=768, type=int)    # default: 768 (same, dent)
    parser.add_argument('--ptr_n_hidden', default=3, type=int)

    # ANCHOR PWFF: based on AttentionIsAllYouNeed, Point-wise FF network is usually set to 4x the embedding dim

    # training
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--warmup', default=4000, type=float)
    parser.add_argument('--factor', default=1, type=float)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--valfreq', default=1, type=int)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--clip', default=5, type=int)
    parser.add_argument('--batch_size', default=10, type=int)  # NOTE: changed from 25

    # test and inference
    parser.add_argument('--model_path', default='experiments/models/CARTONwNERwLinPtr_e42_v0.0145_multitask.pth.tar',
                        type=str)
    parser.add_argument('--file_path', default='/data/final/csqa/process/test.json', type=str)
    parser.add_argument('--inference_partition', default=InferencePartition.TEST.value,
                        choices=[ip.value for ip in InferencePartition], type=str)
    parser.add_argument('--question_type', default=QuestionTypes.SIMPLE_DIRECT.value,
                        choices=[qt.value for qt in QuestionTypes], type=str)
    parser.add_argument('--max_results', default=1000, help='maximum number of results', type=int)
    parser.add_argument('--ner_max_distance', default=[0, 0, 1, 1, 1, 2])

    # elasticsearch related
    # parser.add_argument('--elastic_index_root', default='csqa_wikidata')
    parser.add_argument('--elastic_index_ent', default=f'{INDEX_ROOT}_ent')
    parser.add_argument('--elastic_index_ent_full', default=f'{INDEX_ROOT}_ent_full')
    parser.add_argument('--elastic_index_rel', default=f'{INDEX_ROOT}_rel')  # TODO: implement relation search
    parser.add_argument('--elastic_index_rdf', default=f'{INDEX_ROOT}_rdf')
    parser.add_argument('--elastic_index_rdf_full', default=f'{INDEX_ROOT}_rdf_full')
    parser.add_argument('--elastic_host', default='https://localhost:9200')
    parser.add_argument('--elastic_certs', default='./knowledge_graph/certs/http_ca.crt')
    parser.add_argument('--elastic_user', default='elastic')
    parser.add_argument('--elastic_password', default=Passwords.NOTEBOOK.value, choices=[pw.value for pw in Passwords])

    return parser
