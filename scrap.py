from unidecode import unidecode
from time import perf_counter
from knowledge_graph.ZODBConnector import BTreeDB
from knowledge_graph.knowledge_graph import MiniKG
from elasticsearch import Elasticsearch

from utils import rapidfuzz_query

from constants import *


def time_query(fun):

    def helper(*args, **kwargs):
        tic = perf_counter()
        res = fun(*args, **kwargs)
        print(f"t: {perf_counter() - tic} | {res}")

    return helper


def es_query(query, client):
    query = unidecode(query)
    res = client.search(index='csqa_wikidata', size=50, query={
            'match': {
                'label': {
                    'query': query,
                    'fuzziness': 'AUTO',
                }
            }
        })

    for hit in res['hits']['hits']:
        print(f'{hit["_source"]["id"]} - {hit["_source"]["label"]} - {hit["_score"]}', end='\n')


if __name__ == '__main__':
    kg = BTreeDB('./knowledge_graph/Wikidata.fs', run_adapter=True)
    kg_memory = MiniKG()
    client = Elasticsearch(args.elastic_host, ca_certs=args.elastic_certs,
                            basic_auth=(args.elastic_user, args.elastic_password['notebook']))  # for inverse index search

    print(client.info())

    timed_rp_query = time_query(rapidfuzz_query)
    timed_es_query = time_query(es_query)

    timed_es_query('Albret Enstein', client)  # warmup

    tic = perf_counter()
    timed_es_query('Albret Enstein', client)
    timed_es_query('Stargate', client)
    timed_es_query('Borat', client)
    timed_es_query('Boat', client)
    timed_es_query('Prety littl liers', client)

    print(f'ElasticSearch performance: {perf_counter() - tic}')

    timed_rp_query('Albret Enstein', 'Q1417412', kg_memory)  # warmup

    tic = perf_counter()
    timed_rp_query('Albret Enstein', 'Q1417412', kg_memory)
    timed_rp_query('Stargate', 'Q1417412', kg_memory)
    timed_rp_query('Borat', 'Q1417412', kg_memory)
    timed_rp_query('Boat', 'Q1417412', kg_memory)
    timed_rp_query('Prety littl liers', 'Q1417412', kg_memory)

    print(f'Memory performance: {perf_counter() - tic}')

    timed_rp_query('Albret Enstein', 'Q1417412', kg)  # warmup

    tic = perf_counter()
    timed_rp_query('Albret Enstein', 'Q1417412', kg)
    timed_rp_query('Stargate', 'Q1417412', kg)
    timed_rp_query('Borat', 'Q1417412', kg)
    timed_rp_query('Boat', 'Q1417412', kg)
    timed_rp_query('Prety littl liers', 'Q1417412', kg)

    print(f'ZODB performance: {perf_counter() - tic}')