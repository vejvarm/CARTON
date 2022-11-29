# inspired by LASAGNE\scripts\csqa_elasticse.py
# made to work for LASAGNE

# delete all elasticsearch indices:
# curl -X DELETE --cacert ~/PycharmProjects/CARTON/knowledge_graph/certs/http_ca.crt -u elastic 'https://localhost:9200/csqa_wikidata'

import time
import logging

from ordered_set import OrderedSet

from action_executor.actions import ESActionOperator
from helpers import connect_to_elasticsearch

from helpers import setup_logger
from args import ElasticIndices

CLIENT = connect_to_elasticsearch()
LOGGER = setup_logger(__name__, loglevel=logging.INFO)

if __name__ == '__main__':
    index_ent = ElasticIndices.ENT_FULL.value
    index_rdf = ElasticIndices.RDF_FULL.value
    # aop = ESActionOperator(CLIENT, index_ent=index_ent, index_rdf=index_rdf)
    aop = ESActionOperator(CLIENT)

    # TEST deleting
    sid = "Q15140125"
    rid = "P31"
    oid = "Q20010800"
    _id = f"{sid}{rid}{oid}"
    print(f"Try delete: {aop.delete_rdf(sid, rid, oid)}")
    print(f"Try insert: {aop.insert(sid, rid, oid)}")
    print(f"Before delete: {aop.find(sid, rid)}")
    print(f"Try delete: {aop.delete_rdf(sid, rid, oid)}")
    print(f"After delete: {aop.find(sid, rid)}")
    print(f"Try insert: {aop.insert(sid, rid, oid)}")
    print(f"After insert: {aop.find(sid, rid)}")
    print(f"Try delete: {aop.delete_rdf(sid, rid, oid)}")

    # TEST
    eid = 'Q11606822'
    res = CLIENT.get(index=index_ent, id=eid)
    print(f"_id: {res['_id']} | _source: {res['_source']}")

    label = aop.get_label(eid)
    print(label)

    # TEST IF INDEX LOOKS OK
    res = CLIENT.search(index=index_rdf, query={'match_all': {}})
    print(f"_id: {res['hits']['hits'][0]['_id']} | _source: {res['hits']['hits'][0]['_source']}")

    exit()

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