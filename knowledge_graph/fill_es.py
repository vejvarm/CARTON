# inspired by LASAGNE\scripts\csqa_elasticse.py
# made to work for LASAGNE

# delete all elasticsearch indices:
# curl -X DELETE --cacert ~/PycharmProjects/CARTON/knowledge_graph/certs/http_ca.crt -u elastic 'https://localhost:9200/csqa_wikidata'

import time
import logging

from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor

import elastic_transport
from tqdm import tqdm
from unidecode import unidecode
from knowledge_graph import MiniKG
from elasticsearch import Elasticsearch
import elasticsearch
import ujson

from constants import args, ROOT_PATH

ELASTIC_USER = 'elastic'
ELASTIC_PASSWORD = 'hZiYNU+ye9izCApoff-v'  # '1jceIiR5k6JlmSyDpNwK'

CLIENT = Elasticsearch(
    "https://localhost:9200",
    ca_certs="./certs/http_ca.crt",
    basic_auth=(ELASTIC_USER, ELASTIC_PASSWORD),
    retry_on_timeout=True,
)

# CLIENT.indices.put_mapping(index='')

logging.basicConfig(format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%d/%m/%Y %I:%M:%S %p',
                    level=logging.INFO,
                    handlers=[logging.StreamHandler()])

logging.getLogger('elastic_transport.transport').setLevel(logging.WARNING)
LOGGER = logging.getLogger(__name__)


def _index_one(item, kg_types: dict = None, index='csqa_wikidata'):
    if kg_types is None:
        kg_types = {}
    id, label = item
    for i in range(10):
        try:
            CLIENT.index(index=index, id=id, document={'id': id,
                                                       'label': unidecode(label),
                                                       'type': kg_types[id] if id in kg_types else []})
            break
        except elastic_transport.ConnectionTimeout:
            print(f'Connection to ES timed out. Retrying (attempt {i})')
            pass


def create_mock_index(index_name: str):
    settings = {"analysis": {
        "analyzer": {
            "my_stop_analyzer": {
                "type": "stop",
                "stopwords": ['_english_']
            }
        }
    }}
    # same as normal stop at this point

    mapping = {'properties': {
        'label': {
            'type': 'text',
            'analyzer': 'fingerprint',
        },
        'types': {
            'type': 'keyword',
            'normalizer': 'lowercase'
        }
    }
    }

    CLIENT.indices.create(index=index_name, settings=settings, mappings=mapping)


def index_mock_entry(index):
    # id = 'Q99'
    # label = 'Brooklyn Nine-Nine'
    # type = 'Q911'  # Police
    # preds = ['P1', 'P2', 'P3', 'P4']
    # objs = [['Q1', 'Q2'], ['Q2'], ['Q1', 'Q3'], ['Q4', 'Q5', 'Q6']]

    id = 'Q69'
    label = 'NICE!'
    type = 'Q1337'  # leet
    preds = ['P1', 'P3', 'P5', 'P7']
    objs = [['Q23', 'Q1'], ['Q13'], ['Q2', 'Q5'], ['Q10', 'Q100', 'Q1']]

    # label is gonna be id'd be the actual entry
    CLIENT.index(index=index, id=id,
                 document={'label': label, 'type': type})  # TODO: Question: can there be more types for one entity?
    for p, o in zip(preds, objs):
        print(f's: {id} -> p:{p} -> o:{o}')
        CLIENT.index(index=index, document={'id': id,
                                            'p': p,
                                            'o': o})


def index_mock_label_type_entries(index):
    ids = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q23', 'Q13', 'Q10', 'Q100']
    labels = [f'I am the {id} label' for id in ids]
    types = [['Q911', 'Q1337'], ['Q911'], ['Q911'], ['Q888', 'Q911'], ['Q1337'], ['Q999', 'Q1337'], ['Q999'], ['Q888'],
             ['Q888'], ['Q911']]

    for id, lab, tp in zip(ids, labels, types):
        CLIENT.index(index=index, id=id, document={'label': lab, 'types': tp})


# ENT INDEX
def create_and_map_ent_index(index=args.elastic_index_ent):
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


def fill_ent_index(index=args.elastic_index_ent, source_json='index_ent_dict.json', max_workers=5):
    """entity-label-type index for all unique entities in KG

    :param index: (str) name of ElasticSearch index
    :param source_json: (str) json file with {entid[str]: {'label': [str], 'types': [list[str]]}} entries
    :param max_workers: (int) maximum number of threads for indexing

    :return: None
    """
    kg_root = f"{ROOT_PATH}/knowledge_graph/"
    index_dict = ujson.loads(open(f"{kg_root}{source_json}").read())

    LOGGER.info(f"Filling ent_index with entries from {source_json} ({len(index_dict)})")
    # for eid in tqdm(index_dict.keys()):
    #     lab = unidecode(index_dict[eid]['label'])  # NOTE: use unidecode to remove and decode special characters!
    #     tps = index_dict[eid]['types']
    #     LOGGER.debug(f"Adding eid: {eid} | label: {lab} | types: {tps}")
    #     CLIENT.index(index=index, id=eid, document={'label': lab, 'types': tps})
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(executor.map(lambda item: _ent_index_op(item, index), index_dict.items()), total=len(index_dict)))
    # with Pool(processes=4, maxtasksperchild=1000) as pool:
    #     for _ in tqdm(pool.imap(_ent_index_op, index_dict.items(), chunksize=100), total=len(index_dict)):
    #         pass


# REL INDEX
def create_and_map_rel_index(index=args.elastic_index_rel):
    mapping = {'properties': {
                    'label': {
                        'type': 'text',
                        'analyzer': 'fingerprint',
                    }
              }}

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
def create_and_map_rdf_index(index=args.elastic_index_rdf):
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


def fill_rdf_index(index=args.elastic_index_rdf, source_json='index_rdf_dict.json', max_workers=5):
    """rdf index for all unique rdf entries (subject-relation-object) in KG

    :param index: (str) name of ElasticSearch index
    :param source_json: (str) json file with {subid[str]: {predid[str]: obj_ids[list[str]]}} entries
    :param max_workers: (int) maximum number of threads for indexing

    the final index entry structure:
    _id: {sid: str, rid: str, oid: str}

    (ensure atomic entries: only one object per entry)

    :return: None
    """
    kg_root = f"{ROOT_PATH}/knowledge_graph/"
    index_dict = ujson.loads(open(f"{kg_root}{source_json}").read())

    LOGGER.info(f"Filling rdf_index with entries from {source_json} ({len(index_dict)})")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(executor.map(lambda item: _rdf_index_op(item, index), index_dict.items()), total=len(index_dict)))

    # for sid, rel_dict in tqdm(index_dict.items()):
    #     for rid, object_list in rel_dict.items():
    #         if not object_list:
    #             object_list = ['']  # !!! so that Properties with no Objects get added as well
    #         for oid in object_list:
    #             _id = f'{sid}{rid}{oid}'
    #             LOGGER.debug(f"Adding sid: {sid} | rid: {rid} | oid {oid}")
    #             CLIENT.index(index=index, id=_id, document={'sid': sid, 'rid': rid, 'oid': oid})


def mock_entries_experiment():
    mock_index = 'mock_ent_index'

    CLIENT.indices.delete(index=mock_index)
    create_mock_index(mock_index)
    index_mock_label_type_entries(mock_index)
    # index_mock_entry(mock_index)

    res = CLIENT.indices.get_mapping(index=mock_index)
    print(res)

    res = CLIENT.get(index=mock_index, id="Q1")
    print(res)

    CLIENT.indices.refresh(index=mock_index)

    # query = unidecode('Gödart is bák')  # actually removes special characters TODO: investigate in main code (should we be using it?)
    query = unidecode('I am the Q2 label')
    # query = unidecode('q999')
    res = CLIENT.search(index=mock_index, size=50,
                        query={
                            'match': {
                                'label': {
                                    'query': query,
                                    'operator': 'and',
                                }
                            }
                        })
    print(res['hits']['hits'])

    res = CLIENT.indices.analyze(index=mock_index,
                                 analyzer='standard',
                                 text=query)
    print(res)


def run_indexing_tests(index='csqa_wikidata'):
    # load necessary KG entries
    kg = MiniKG()
    kg_entities = kg.id_entity.items()
    kg_types = kg.entity_type
    print(f'Num of wikidata entities: {len(kg_entities)}')

    print(CLIENT.info())

    index_one = lambda item: _index_one(item, kg_types, index)

    with Pool(processes=12, maxtasksperchild=100000) as pool:
        for _ in tqdm(pool.imap(index_one, kg_entities, chunksize=100), total=len(kg_entities)):
            pass

    # tic = time.perf_counter()
    # for i, item in enumerate(kg_entities):
    #     index_one(item)
    #
    #     if (i + 1) % 10000 == 0: print(
    #         f'==> Finished {((i + 1) / len(kg_entities)) * 100:.4f}% -- {time.perf_counter() - tic:0.2f}s')


def run_search_tests():
    index = "csqa_wikidata"
    analyzer = 'standard'
    for i in range(1):
        tic = time.perf_counter()
        query = unidecode('Bank America')  # it's case-insensitive
        res = CLIENT.search(index=index, size=50,
                            query={
                                'match': {
                                    'label': {
                                        'query': query,
                                        'fuzziness': 'AUTO',
                                        'analyzer': analyzer,
                                        'operator': 'and',
                                    }
                                }
                            })
        print(f'Search time: {time.perf_counter() - tic}')

    an = CLIENT.indices.analyze(index=index, analyzer=analyzer,
                                text=query)

    print(an)
    for hit in res['hits']['hits']:
        print(hit)
        print(f'{hit["_source"]["id"]} - {hit["_source"]["label"]} - {hit["_score"]}')
        print('**********************')

    mock_index = 'mock_csqa'
    item = ('Q69', 'NICE!')
    _index_one(item, index=mock_index)
    index_mock_entry(mock_index)

    # Query for RDF
    res = CLIENT.search(index=index,
                        query={
                            'bool': {
                                'must': [
                                    {
                                        'match': {
                                            'id': 'Q99',
                                        },
                                    },
                                    {
                                        'terms': {
                                            'o': ['q1', 'q3'],
                                        },
                                    }
                                ]
                            }
                        })

    res = CLIENT.search(index=index,
                        query={
                            'bool': {
                                'must': [
                                    {
                                        'match': {
                                            'o': 'Q1',
                                        },
                                    },
                                    {
                                        'terms': {
                                            'p': ['p3', 'p1'],
                                        },
                                    }
                                ]
                            }
                        })
    print(res)
    for hit in res['hits']['hits']:
        print(hit['_source'])
    # testing simple id request:
    res = CLIENT.get(index=index, id=res['hits']['hits'][0]['_source']['id'])
    print(f"_id: {res['_id']} | label: {res['_source']['label']} | type: {res['_source']['type']}")

    # testing search for multiple documents by ids
    res = CLIENT.search(index=index,  # NOTE: Finds even partial string matches of ids
                        query={
                            'ids': {
                                'values': list({"Q99", "Q69"})
                            }
                        })
    print(res)

    res = CLIENT.mget(index=index,  # NOTE: Finds EXACT id match
                      ids=["Q99", ])  # has to be a list (not set, not str)
    print(res)


def fill_csqa_from_index_jsons(index, subset='', create=False, max_workers=5):
    source = 'index_rel_dict.json'
    index_rel = f'{index}_rel'
    if create:
        create_and_map_rel_index(index_rel)
    fill_rel_index(index=index_rel, source_json=source)

    source = f'index_ent_dict{subset}.json'
    index_ent = f'{index}_ent'
    if create:
        create_and_map_ent_index(index_ent)
    fill_ent_index(index=index_ent, source_json=source, max_workers=max_workers)

    source = f'index_rdf_dict{subset}.json'
    index_rdf = f'{index}_rdf'
    if create:
        create_and_map_rdf_index(index_rdf)
    fill_rdf_index(index=index_rdf, source_json=source, max_workers=max_workers)


if __name__ == '__main__':
    pass
    # mock_entries_experiment()

    # try out indexing
    # run_indexing_tests()

    # test out searching
    # run_search_tests()

    # INDEX OPERATIONS
    index = 'csqa_wikidata_test'

    # # DELETE indices
    CLIENT.indices.delete(index=f'{index}_ent')
    CLIENT.indices.delete(index=f'{index}_rel')
    CLIENT.indices.delete(index=f'{index}_rdf')

    # CREATE, MAP and FILL indices
    subset = '_first_10000'
    fill_csqa_from_index_jsons(index, subset, create=True, max_workers=10)
