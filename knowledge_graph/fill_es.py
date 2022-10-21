# inspired by LASAGNE\scripts\csqa_elasticse.py
# made to work for LASAGNE

# delete all elasticsearch indices:
# curl -X DELETE --cacert ~/PycharmProjects/CARTON/knowledge_graph/certs/http_ca.crt -u elastic 'https://localhost:9200/csqa_wikidata'

import time
import logging

from multiprocessing import Pool

import elastic_transport
from tqdm import tqdm
from unidecode import unidecode
from knowledge_graph import MiniKG
from elasticsearch import Elasticsearch
import elasticsearch

ELASTIC_USER = 'elastic'
ELASTIC_PASSWORD = '1jceIiR5k6JlmSyDpNwK'

CLIENT = Elasticsearch(
    "https://localhost:9200",
    ca_certs="./certs/http_ca.crt",
    basic_auth=(ELASTIC_USER, ELASTIC_PASSWORD),
    retry_on_timeout=True,
)

logging.basicConfig(format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%d/%m/%Y %I:%M:%S %p',
                    level=logging.INFO,
                    handlers=[logging.StreamHandler()])

logging.getLogger('elastic_transport.transport').setLevel(logging.WARNING)

# print(logging.root.setLevel(logging.WARNING))

def index_one(item):
    id, label = item
    for i in range(10):
        try:
            CLIENT.index(index='csqa_wikidata', document={'id': id,
                                                          'label': unidecode(label),
                                                          'type': kg_types[id] if id in kg_types else []})
            break
        except elastic_transport.ConnectionTimeout:
            print(f'Connection to ES timed out. Retrying (attempt {i})')
            pass


if __name__ == '__main__':
    # res = requests.get('https://localhost:9200')

    # load necessary KG entries
    # kg = MiniKG()
    # kg_entities = kg.id_entity.items()
    # kg_types = kg.entity_type
    # print(f'Num of wikidata entities: {len(kg_entities)}')
    #
    # print(CLIENT.info())
    #
    # with Pool(processes=12, maxtasksperchild=100000) as pool:
    #     for _ in tqdm(pool.imap(index_one, kg_entities, chunksize=100), total=len(kg_entities)):
    #         pass

    # tic = time.perf_counter()
    # for i, (id, label) in enumerate(kg_entities):
    #     index_one(CLIENT, id, label, kg_types)
    #
    #     if (i + 1) % 10000 == 0: print(
    #         f'==> Finished {((i + 1) / len(kg_entities)) * 100:.4f}% -- {time.perf_counter() - tic:0.2f}s')

    # test it out
    for i in range(100):
        tic = time.perf_counter()
        query = unidecode('suny in philadeplphia')
        res = CLIENT.search(index='csqa_wikidata', size=50, query={
                'match': {
                    'label': {
                        'query': query,
                        'fuzziness': 'AUTO',
                    }
                }
            })
        print(f'Search time: {time.perf_counter() - tic}')

    res = CLIENT.search(index='csqa_wikidata', size=50,
                    query={'match': {'label': {'query': unidecode(query), 'fuzziness': 'AUTO'}}})

    for hit in res['hits']['hits']:
        print(f'{hit["_source"]["id"]} - {hit["_source"]["label"]} - {hit["_score"]}')
        print('**********************')