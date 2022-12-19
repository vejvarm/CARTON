from helpers import connect_to_elasticsearch
from action_executor.actions import ESActionOperator

CLIENT = connect_to_elasticsearch()
OP = ESActionOperator(CLIENT)

if __name__ == '__main__':
    # res = CLIENT.search(index=OP.index_rdf, size=10,
    #                     query={
    #                         'match_all': {}
    #                     })
    #
    # for hit in res['hits']['hits']:
    #     print(f"{hit['_id']}\t{hit['_source']['sid']}\t{hit['_source']['rid']}\t{hit['_source']['oid']}")
    #
    # print('REL index')
    # res = CLIENT.search(index=OP.index_rel, size=10,
    #                     query={
    #                         'match': {'label': 'publisher'}
    #                     })
    #
    # for hit in res['hits']['hits']:
    #     print(f"{hit['_id']}\t{hit['_source']['label']}")
    #
    # print('ENT index')
    # res = CLIENT.search(index=OP.index_ent, size=10,
    #                     query={
    #                         'match': {'label': 'cosmology'}
    #                     })
    #
    # for hit in res['hits']['hits']:
    #     print(f"{hit['_id']}\t{hit['_source']['label']}\t{hit['_source']['types']}")

    print('Specific RDFs')
    sid = 'Q1962105'
    rid = 'P921'  # 'P495'
    oid = 'Q338'
    _id = f'{sid}{rid}{oid}'
    # if OP.get_rdf(_id):
    print(f"{_id}\t{sid}\t{rid}\t{oid}")
