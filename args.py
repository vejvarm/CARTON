import argparse

from constants import ElasticIndices, Task, KGType, InferencePartition, QuestionTypes, Passwords, MODEL_NAME


def get_parser():
    parser = argparse.ArgumentParser(description=MODEL_NAME)

    # general
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--cuda-device', default=0, type=int)
    parser.add_argument('--name', default="", type=str)

    # data
    parser.add_argument('--data-path', default='data/final/csqa')
    parser.add_argument('--no-data-cache', action='store_true')
    parser.add_argument('--no-vocab-cache', action='store_true')
    parser.add_argument('--cache-path', default='.cache', type=str)
    # parser.add_argument('--embedding_path', default='/knowledge_graph/entity_embeddings.json')
    parser.add_argument('--ent_dict_path', default="knowledge_graph/items_wikidata_n.json")
    parser.add_argument('--rel_dict_path', default="knowledge_graph/index_rel_dict.json")
    parser.add_argument('--kg_type', default=KGType.ELASTICSEARCH.value, choices=[tp.value for tp in KGType])

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
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--valfreq', default=1, type=int)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--clip', default=5, type=int)
    parser.add_argument('--batch_size', default=25, type=int)  # NOTE: changed from 25
    parser.add_argument('--pool_size', default=100, type=int)

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
    parser.add_argument('--elastic_index_ent', default=ElasticIndices.ENT.value)
    parser.add_argument('--elastic_index_ent_full', default=ElasticIndices.ENT_FULL.value)
    parser.add_argument('--elastic_index_rel', default=ElasticIndices.REL.value)  # TODO: implement relation search
    parser.add_argument('--elastic_index_rdf', default=ElasticIndices.RDF.value)
    parser.add_argument('--elastic_index_rdf_full', default=ElasticIndices.RDF_FULL.value)
    parser.add_argument('--elastic_host', default='https://localhost:9200')
    parser.add_argument('--elastic_certs', default='./knowledge_graph/certs/http_ca.crt')
    parser.add_argument('--elastic_user', default='elastic')
    parser.add_argument('--elastic_password', default=Passwords.FREYA.value, choices=[pw.value for pw in Passwords])

    return parser


def parse_and_get_args(args=tuple()):
    # read parser
    parser = get_parser()
    return parser.parse_args(args=args)
