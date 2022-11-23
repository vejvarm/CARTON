# inspired by LASAGNE\scripts\csqa_elasticse.py
# made to work for LASAGNE

# delete all elasticsearch indices:
# curl -X DELETE --cacert ~/PycharmProjects/CARTON/knowledge_graph/certs/http_ca.crt -u elastic 'https://localhost:9200/csqa_wikidata'

import time
import logging

from multiprocessing import Pool

from ordered_set import OrderedSet

from action_executor.actions import ESActionOperator
from elasticsearch import Elasticsearch
import elasticsearch

from helpers import setup_logger

ELASTIC_USER = 'elastic'
ELASTIC_PASSWORD = 'hZiYNU+ye9izCApoff-v'  # '1jceIiR5k6JlmSyDpNwK'

CLIENT = Elasticsearch(
    "https://localhost:9200",
    ca_certs="../knowledge_graph/certs/http_ca.crt",
    basic_auth=(ELASTIC_USER, ELASTIC_PASSWORD),
    retry_on_timeout=True,
)

LOGGER = setup_logger(__name__, loglevel=logging.INFO)

if __name__ == '__main__':
    aop = ESActionOperator(CLIENT)

    # TEST IF INDEX LOOKS OK
    res = CLIENT.search(index='csqa_wikidata_test_rdf', query={'match_all': {}})
    print(f"_id: {res['hits']['hits'][0]['_id']} | _source: {res['hits']['hits'][0]['_source']}")

    # FIND OBJECTS by SUBJECT+PRED
    objects = aop.find(['Q1253484', 'Q1253487'], 'P105')  # should return -> Q7432  # ANCHOR: WORKS on csqa_test
    print(objects)

    # FIND SUBJECTS by OBJECT+PRED
    subjects = aop.find_reverse(['Q156737', 'Q7'], 'P108')  # should return -> Q1253487  # ANCHOR: WORKS on csqa_test
    print(subjects)

    # FILTER ENTITY SET BY TYPE
    entities = aop.filter_type(aop.find_reverse(['Q156737', 'Q7'], 'P108'), 'Q381702')  # ANCHOR: will work on full csqa_wikidata
    print(entities)

    entity_set = OrderedSet(['Q11606822', 'Q17570681'])
    entities = aop.filter_type(entity_set, 'Q771100')  # If entity types list is empty  # ANCHOR: WORKS on csqa_test
    print(f'Empty types: {entities}')  # expected return: []

    entity_set = OrderedSet(['Q11606822', 'Q17570681', 'Q13579708'])
    entities = aop.filter_type(entity_set, '')  # If we want to filter the empty types
    print(f'Empty types (want): {entities}')  # expected return: ['Q11606822', 'Q17570681']  # ANCHOR: WORKS on csqa_test

    entity_set = OrderedSet(['Q21550698', 'Q22108479', 'Q9091325', 'Q1959230', 'Q24684358'])
    entities = aop.filter_type(entity_set, 'Q5')  # filter entities with type "Q5" (person)  # ANCHOR: WORKS on csqa_test
    print(f'Humans: {entities}')  # expected return: ['Q21550698', 'Q22108479', 'Q9091325']

    # FILTER ENTITY SET BY Multiple types
    entity_set = OrderedSet(['Q21550698', 'Q22108479', 'Q9091325', 'Q1959230', 'Q24684358'])
    entities = aop.filter_multi_types(entity_set, 'Q502895', 'Q55488')  # this is implemented with OR operator  # ANCHOR: WORKS on csqa_test
    print(f'Humans & rails: {entities}')  # expected return: ['Q21550698', 'Q22108479', 'Q9091325', 'Q1959230']

    # TUPLE COUNTS (number of objects of type t2 for subject of type t1 and relation p)
    count = aop.find_tuple_counts('P646', 'P5', 'Q5')  # empty dict because none of subs have type P5 in ent_index (for csqa_test)
    print(count)
    count = aop.find_tuple_counts('P646', None, 'Q5')  # lot of 0 counts because none of the subs have objs with tp Q5
    print(count)
    count = aop.find_tuple_counts('P646', None, None)  # lot of 1 counts because we dont filter by types
    print(count)


    # REVERSE TUPLE COUNTS (number of objects of type t2 for subject of type t1 and relation p)
    count = aop.find_reverse_tuple_counts('P155', 'P5', 'Q5')  # empty dict because none of subs have type P5 in ent_index (for csqa_test)
    print(count)
    count = aop.find_reverse_tuple_counts('P155', None, 'Q5')  # lot of 0 counts because none of the subs have objs with tp Q5
    print(count)
    count = aop.find_reverse_tuple_counts('P155', None, None)  # lot of 1 counts because we dont filter by types
    print(count)


    # INSERT OP
    # 1. exists already
    # 2. doesn't exist


    # UPDATE_LABEL OP


    # UPDATE TYPES OP