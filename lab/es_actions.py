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

ELASTIC_USER = 'elastic'
ELASTIC_PASSWORD = 'hZiYNU+ye9izCApoff-v'  # '1jceIiR5k6JlmSyDpNwK'

CLIENT = Elasticsearch(
    "https://localhost:9200",
    ca_certs="../knowledge_graph/certs/http_ca.crt",
    basic_auth=(ELASTIC_USER, ELASTIC_PASSWORD),
    retry_on_timeout=True,
)

logging.basicConfig(format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%d/%m/%Y %I:%M:%S %p',
                    level=logging.INFO,
                    handlers=[logging.StreamHandler()])

logging.getLogger('elastic_transport.transport').setLevel(logging.WARNING)


if __name__ == '__main__':
    aop = ESActionOperator(CLIENT, index='csqa_wikidata')

    # FIND OBJECTS by SUBJECT+PRED
    objects = aop.find(['Q99', 'Q69'], 'P1')
    print(objects)

    # FIND SUBJECTS by OBJECT+PRED
    objects = aop.find_reverse(['Q1', 'Q7'], 'P1')
    print(objects)

    # FILTER ENTITY SET BY TYPE
    objects = aop.filter_type(aop.find_reverse(['Q1', 'Q7'], 'P1'), 'Q911')
    print(objects)

    objects = aop.filter_type(OrderedSet(['Q8', 'Q99']), 'Q911')  # If entity doesn't exist
    print(objects)

    # FILTER ENTITY SET BY Multiple types
    objects = aop.filter_multi_types(OrderedSet(['Q69', 'Q99']), 'Q91', 'Q1337')
    print(objects)

    # TUPLE COUNTS (number of objects of type t2 for subject of type t1 and relation p)
    count = aop.find_tuple_counts('P4', 'Q911', 'Q911')
    print(count)
    count = aop.find_tuple_counts('P4', 'Q911', 'Q1337')
    print(count)

    # REVERSE TUPLE COUNTS (number of objects of type t2 for subject of type t1 and relation p)
    count = aop.find_reverse_tuple_counts('P4', 'Q1337', 'Q911')
    print(count)
    count = aop.find_reverse_tuple_counts('P4', 'Q911', 'Q1337')
    print(count)