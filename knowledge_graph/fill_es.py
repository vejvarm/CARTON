# inspired by LASAGNE\scripts\csqa_elasticse.py
# made to work for LASAGNE

import time
from unidecode import unidecode
from knowledge_graph import MiniKG
from elasticsearch import Elasticsearch

ELASTIC_USER = 'elastic'
ELASTIC_PASSWORD = 'hZiYNU+ye9izCApoff-v'


if __name__ == '__main__':
    # res = requests.get('https://localhost:9200')

    client = Elasticsearch(
        "https://localhost:9200",
        ca_certs="./certs/http_ca.crt",
        basic_auth=(ELASTIC_USER, ELASTIC_PASSWORD)
    )

    print(client.info())

    # # load necessary KG entries
    # kg = MiniKG()
    # kg_entities = list(kg.id_entity.items())
    # kg_types = kg.entity_type
    # print(f'Num of wikidata entities: {len(kg_entities)}')
    #
    # tic = time.perf_counter()
    # for i, (id, label) in enumerate(kg_entities):
    #     client.index(index='csqa_wikidata', id=str(i + 1), document={'id': id,
    #                                                                  'label': unidecode(label),
    #                                                                  'type': kg_types[id] if id in kg_types else []})
    #     if (i + 1) % 10000 == 0: print(
    #         f'==> Finished {((i + 1) / len(kg_entities)) * 100:.4f}% -- {time.perf_counter() - tic:0.2f}s')



    # test it out
    tic = time.perf_counter()
    query = unidecode('suny in philadeplphia')
    res = client.search(index='csqa_wikidata', size=50, query={
            'match': {
                'label': {
                    'query': query,
                    'fuzziness': 'AUTO',
                }
            }
        })
    print(f'Search time: {time.perf_counter() - tic}')

    for hit in res['hits']['hits']:
        print(f'{hit["_source"]["id"]} - {hit["_source"]["label"]} - {hit["_score"]}')
        print('**********************')