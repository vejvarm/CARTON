# inspired by LASAGNE\scripts\csqa_elasticse.py
# made to work for LASAGNE and CARTONwNER

# delete all elasticsearch indices:
# curl -X DELETE --cacert ~/PycharmProjects/CARTON/knowledge_graph/certs/http_ca.crt -u elastic 'https://localhost:9200/csqa_wikidata'
import time
import logging

from multiprocessing import Pool
from random import randint

import elastic_transport
from tqdm import tqdm
from unidecode import unidecode
from knowledge_graph.KnowledgeGraphs import MiniKGWikidataJSON
from elasticsearch import Elasticsearch

from constants import args, ROOT_PATH
from action_executor.actions import search_by_label
from helpers import setup_logger, connect_to_elasticsearch

CLIENT = connect_to_elasticsearch()

# CLIENT.indices.put_mapping(index='')

LOGGER = setup_logger(__name__, loglevel=logging.INFO)


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
                 document={'label': label, 'types': [type]})
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
    kg = MiniKGWikidataJSON()
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


if __name__ == '__main__':
    pass
    # mock_entries_experiment()

    # try out indexing
    # run_indexing_tests()

    # test out searching
    # run_search_tests()

    # if tp exists in types field of entity:
    # query, tp = ('Cheb River', "Q4022")                           # expected -> ['Q13609920']
    # if type field is empty in ent_index:
    # query, tp = ("2002 in South Africa televison", "Q12737077")   # expected -> ['Q19570438']
    # if tp doesn't exists in types field of entity:
    query, tp = ("dubel poort", "Q12737077")                      # expected -> ['Q19057085']
    res = search_by_label(CLIENT, query, tp)
    print(res)

    res = CLIENT.exists(index=args.elastic_index_rdf, id='Q1253487P735Q381702')
    print(res)

    # res = CLIENT.exists(index=args.elastic_index_ent, id='')

    eid = f'Q19057085'
    # generate new id randomly until we generate unique id
    while True:
        eid = f'Q{randint(1000000, 9999999)}'
        print(eid)

        if not CLIENT.exists(index=args.elastic_index_ent, id=eid):
            break

    eid = f'Q13609920'  # Chet River
    # eid = f'Qnonexistant'
    print(CLIENT.get(index=args.elastic_index_ent, id=eid))
    print(CLIENT.update(index=args.elastic_index_ent, id=eid, doc={'label': 'Chet River'}))
    print(CLIENT.get(index=args.elastic_index_ent, id=eid)['_source']['label'])
    # update:
    #   if exists and is updated: res['result'] == 'updated' | res['_version'] += 1
    #   if exists and is the same: res['result'] == 'noop'
    #   if doesn't exist: raise elasticsearch.NotFoundError

    # setting types
    eid = f'Q13609920'  # Chet River
    tp = ['Qnew']
    print(CLIENT.get(index=args.elastic_index_ent, id=eid))
    print(CLIENT.update(index=args.elastic_index_ent, id=eid, doc={'types': tp}))
    print(CLIENT.get(index=args.elastic_index_ent, id=eid)['_source']['types'])
    #   if exists: the field is completely overwritten

    # updating types
    # TODO Alternative: use append pipeline processor?
    tp_list = ['Qnew', 'Qnew2', 'Qnew3']
    tp_set = set(CLIENT.get(index=args.elastic_index_ent, id=eid)['_source']['types'])
    tp_set.update(tp_list)
    res = CLIENT.update(index=args.elastic_index_ent, id=eid, doc={'types': list(tp_set)})
    print(CLIENT.get(index=args.elastic_index_ent, id=eid)['_source']['types'])
