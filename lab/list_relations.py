import logging

from action_executor.actions import ESActionOperator
from helpers import connect_to_elasticsearch

from helpers import setup_logger
from constants import ElasticIndices

CLIENT = connect_to_elasticsearch()
LOGGER = setup_logger(__name__, loglevel=logging.INFO)

if __name__ == '__main__':
    index_rel = ElasticIndices.REL.value
    index_rdf = ElasticIndices.RDF_FULL.value

    res = CLIENT.search(index=index_rel, query={'match_all': {}}, size=1000)
    for i, hit in enumerate(res['hits']['hits']):
        print(f"{i:03d}\t{hit['_id']:5s}\t{hit['_source']['label']}")
    # print(f"_id: {res['hits']['hits'][0]['_id']} | _source: {res['hits']['hits'][0]['_source']}")

    aop = ESActionOperator(CLIENT)

    res = aop.find("Q", "P50")
    print(res)