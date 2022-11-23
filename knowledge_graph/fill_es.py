# inspired by LASAGNE\scripts\csqa_elasticse.py
# made to work for LASAGNE and CARTONwNER

# delete all elasticsearch indices:
# curl -X DELETE --cacert ~/PycharmProjects/CARTON/knowledge_graph/certs/http_ca.crt -u elastic 'https://localhost:9200/csqa_wikidata'

import logging

from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm
from unidecode import unidecode
from elasticsearch import Elasticsearch
import ujson

from constants import args, ROOT_PATH

ELASTIC_USER = args.elastic_user
ELASTIC_PASSWORD = args.elastic_password['freya']  # refer to args.py --elastic_password for alternatives

CLIENT = Elasticsearch(
    args.elastic_host,
    ca_certs=f'{ROOT_PATH}/{args.elastic_certs.removeprefix("./")}',
    basic_auth=(ELASTIC_USER, ELASTIC_PASSWORD),
    retry_on_timeout=True,
)

logging.basicConfig(format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%d/%m/%Y %I:%M:%S %p',
                    level=logging.INFO,
                    handlers=[logging.StreamHandler()])

logging.getLogger('elastic_transport.transport').setLevel(logging.WARNING)
LOGGER = logging.getLogger(__name__)


# ENT INDEX
def create_and_map_ent_index(index=args.elastic_index_ent_full):
    # delete index if already exists:
    if CLIENT.indices.exists(index=index):
        CLIENT.indices.delete(index=index)

    mapping = {'properties': {
                    'label': {
                        'type': 'text',
                        'analyzer': 'fingerprint',
                    },
                    'types': {
                        'type': 'keyword',
                        'normalizer': 'lowercase'
                    }
              }}

    CLIENT.indices.create(index=index, mappings=mapping)


def _ent_index_op(item, index):
    eid = item[0]
    lab = unidecode(item[1]['label'])
    tps = item[1]['types']

    LOGGER.debug(f"Adding eid: {eid} | label: {lab} | types: {tps}")
    CLIENT.index(index=index, id=eid, document={'label': lab, 'types': tps})


def fill_ent_index(index=args.elastic_index_ent_full, source_json='index_ent_dict.json', max_workers=5):
    """entity-label-type index for all unique entities in KG

    :param index: (str) name of ElasticSearch index
    :param source_json: (str) json file with {entid[str]: {'label': [str], 'types': [list[str]]}} entries
    :param max_workers: (int) maximum number of threads for indexing

    :return: None
    """
    kg_root = f"{ROOT_PATH}/knowledge_graph/"
    index_dict = ujson.loads(open(f"{kg_root}{source_json}").read())

    LOGGER.info(f"Filling ent_index with entries from {source_json} ({len(index_dict)})")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(executor.map(lambda item: _ent_index_op(item, index), index_dict.items()), total=len(index_dict)))


# REL INDEX
def create_and_map_rel_index(index=args.elastic_index_rel):
    # delete index if already exists:
    if CLIENT.indices.exists(index=index):
        CLIENT.indices.delete(index=index)

    mapping = {'properties': {
                    'label': {
                        'type': 'text',
                        'analyzer': 'fingerprint',
                    }
              }}

    # create index with above mapping
    CLIENT.indices.create(index=index, mappings=mapping)


def fill_rel_index(index=args.elastic_index_rel, source_json='index_rel_dict.json'):
    """relation-label index for all unique relations (properties) in KG

    :param index: (str) name of ElasticSearch index
    :param source_json: (str) json file with {relid[str]: label[str]} entries

    :return: None
    """
    kg_root = f"{ROOT_PATH}/knowledge_graph/"
    index_dict = ujson.loads(open(f"{kg_root}{source_json}").read())

    LOGGER.info(f"Filling rel_index with entries from {source_json} ({len(index_dict)})")
    for rid, rlabel in tqdm(index_dict.items()):
        # NOTE: use unidecode to remove and decode special characters from labels!
        LOGGER.debug(f"Adding rid: {rid} | label: {unidecode(rlabel)}")
        CLIENT.index(index=index, id=rid, document={'label': unidecode(rlabel)})


# RDF INDEX
def create_and_map_rdf_index(index=args.elastic_index_rdf_full):
    # delete index if already exists:
    if CLIENT.indices.exists(index=index):
        CLIENT.indices.delete(index=index)

    mapping = {'properties': {
                    'sid': {
                        'type': 'keyword',
                        'normalizer': 'lowercase',
                    },
                    'rid': {
                        'type': 'keyword',
                        'normalizer': 'lowercase',
                    },
                    'oid': {
                        'type': 'keyword',
                        'normalizer': 'lowercase',
                    }
              }}

    CLIENT.indices.create(index=index, mappings=mapping)


def _rdf_index_op(item, index):
    sid = item[0]
    rel_dict = item[1]

    for rid, object_list in rel_dict.items():
        if not object_list:
            object_list = ['']  # !!! so that Properties with no Objects get added as well
        for oid in object_list:
            _id = f'{sid}{rid}{oid}'
            LOGGER.debug(f"Adding sid: {sid} | rid: {rid} | oid {oid}")
            CLIENT.index(index=index, id=_id, document={'sid': sid, 'rid': rid, 'oid': oid})


def fill_rdf_index(index=args.elastic_index_rdf_full, source_json='index_rdf_dict.json', max_workers=5):
    """rdf index for all unique rdf entries (subject-relation-object) in KG

    :param index: (str) name of ElasticSearch index
    :param source_json: (str) json file with {subid[str]: {predid[str]: obj_ids[list[str]]}} entries
    :param max_workers: (int) maximum number of threads for indexing

    the final index entry structure:
    _id: {sid: str, rid: str, oid: str}
    _id is a unique string identifier made from concatenation of sid+rid+oid

    (ensure atomic entries: only one object per entry)

    :return: None
    """
    kg_root = f"{ROOT_PATH}/knowledge_graph/"
    index_dict = ujson.loads(open(f"{kg_root}{source_json}").read())

    LOGGER.info(f"Filling rdf_index with entries from {source_json} ({len(index_dict)})")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(executor.map(lambda item: _rdf_index_op(item, index), index_dict.items()), total=len(index_dict)))


def fill_csqa_from_index_jsons(subset='', create=False, max_workers=5):
    index_rel, source_rel = (args.elastic_index_rel, 'index_rel_dict.json')
    index_ent, source_ent = (args.elastic_index_ent_full, f'index_ent_dict{subset}.json')
    index_rdf, source_rdf = (args.elastic_index_rdf_full, f'index_rdf_dict{subset}.json')

    if create:
        create_and_map_rel_index(index_rel)
    fill_rel_index(index=index_rel, source_json=source_rel)

    if create:
        create_and_map_ent_index(index_ent)
    fill_ent_index(index=index_ent, source_json=source_ent, max_workers=max_workers)

    if create:
        create_and_map_rdf_index(index_rdf)
    fill_rdf_index(index=index_rdf, source_json=source_rdf, max_workers=max_workers)


if __name__ == '__main__':
    # INDEX OPERATIONS
    old_index = 'csqa_wikidata'

    # # DELETE old index
    if CLIENT.indices.exists(index=old_index):
        CLIENT.indices.delete(index=old_index)  # BEWARE: This deletes EVERYTHING in the old index

    # CREATE, MAP and FILL indices
    subset = ''  # alternative for testing: '_first_10000'
    fill_csqa_from_index_jsons(subset, create=True, max_workers=5)

# TODO list
#  DONE 0: utils.search_by_label: test reimplementation of inverse index (entity label) search to work with new index layout!
#   ・ tested in es_search_and_indexing_tests.py
#  DONE 1: should we implement inverse index search in actions.py?
#  DONE 2: in actions.py: finish implementing insert, insert_reverse, set_labels, set_types
#  TODO 4: in train.py.train() implement ner module functionality and evaluation
#  TODO 5: how to deal with initial incompetence of the NER module while adding entities? Warmup with adding disabled, but reward gained?
#   .a what if we just simulate the adding to KG during training and give the reward ... but how then to account for rewarding changing the decision based on existance?
#   .b just simulate 50/50 chance that it already exists? (50% - search from index_ent 50% - search from index_ent_full)
#   .c how to evaluate the final performance?
#  TODO 6: implement resetting of the index at the beginning of training (not necessary if we implement the TODO 5.b)
#  TODO 7: T5 for transcribing questions to statements
#  BONUSES:
#  TODO B1: unify ES CLIENT inits to utils.py
#  TODO B2: polish kg_migration of index_rdf.json
#  TODO B3: analyze ES disk usage (https://www.elastic.co/guide/en/elasticsearch/reference/master/indices-disk-usage.html)
