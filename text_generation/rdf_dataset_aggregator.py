# This file is for generating a dataset of RDF entries, aggregated by having the same subject entity
# for the purpose of D2T generation from the extracted RDF files
from helpers import connect_to_elasticsearch
from action_executor.actions import ESActionOperator

# Plan:
# first, we connect to elasticsearch
if __name__ == '__main__':
    client = connect_to_elasticsearch()
    aop = ESActionOperator(client)
    all_subjs = client.search(index=aop.index_rdf,
                              query={'match_all': {}},
                              size=10)
    print(all_subjs)
    # TODO: maybe use aggregations?
    # https://www.elastic.co/guide/en/elasticsearch/reference/current/search-aggregations-pipeline.html

    # HOHOHOOOOO
    sid_agg_with_oid = client.search(
        query={
            "bool": {
                "must_not": [
                    {
                        "match": {
                            "oid": ""
                        }
                    }
                ]
            }
        },
        size=0,
        aggs={
        "group_by_sid": {
            "terms": {
                "field": "sid",
                "size": 1000,  # so we need to take a lot (how do we not time out?)
                "order": {
                    "_count": "desc"
                },
            },
            "aggs": {
                "hits": {
                    "top_hits": {
                        "size": 3,
                    }
                },
                "top_hits_filter": {
                    "bucket_selector": {
                        "buckets_path": {
                            "total_hits": "_count"
                        },
                        "script": "params.total_hits <= 500"
                    }
                }
            }
        }
    })

    print(sid_agg_with_oid['aggregations']['group_by_sid']['buckets'])
    key = sid_agg_with_oid['aggregations']['group_by_sid']['buckets'][0]['key']
    for dct in sid_agg_with_oid['aggregations']['group_by_sid']['buckets']:
        key = dct['key']
        doc_count = dct['doc_count']
        hits = dct['hits']
        print(f"key: {key} ({doc_count})")
        for hit in hits['hits']['hits']:
            print(hit["_source"])
        print("\n\n")

    exit()

    # for subj in all_subjs:
    #     res = client.search(index=aop.index_rdf,
    #                              query={aop._match('sid', subj)})
# then, for each subject entity (from index_ent), we extract all RDFs with the same subject entity and we concatenate
# them into list
# save it all into folder structure as follows:
#   create split folders for
#       train - 1triples, 2triples, 3triples ...
#       test - 1triples, 2triples, 3triples ...
#       dev  - 1triples, ...
# each folder contains json file with this structure:
#     {"data":
#             [
#                 ['subject1_label | relation_label | object_label',
#                  'subject1_label | ... | ...',
#                  'subject1_label | ... | ...'],
#                 ['subject2_label | relation_label | object_label',
#                  '...',
#                  '...'],
#                 ...
#             ]
#     }
