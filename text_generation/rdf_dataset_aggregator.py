# This file is for generating a dataset of RDF entries, aggregated by having the same subject entity
# for the purpose of D2T generation from the extracted RDF files
import json
import pathlib
from random import shuffle

from helpers import connect_to_elasticsearch
from constants import ROOT_PATH
from action_executor.actions import ESActionOperator

BUCKET_SAVE_FREQUENCY = 100000  # subjects/bucket
DATA_DUMP_ROOT = ROOT_PATH.joinpath("text_generation").joinpath("data")
DUMP_PATH = {"buckets": DATA_DUMP_ROOT.joinpath("buckets"),
             "train": DATA_DUMP_ROOT.joinpath("train"),
             "test": DATA_DUMP_ROOT.joinpath("test"),
             "dev": DATA_DUMP_ROOT.joinpath("dev")}
for pth in DUMP_PATH.values():
    pth.mkdir(exist_ok=True, parents=True)

SPLIT = {"train": 0.8,
          "test": 0.1,
          "dev": 0.1}


def process_and_save_buckets(buckets, dataset_name: str):
    """ Helper function to process and save the triples"""
    data = []

    for bucket in buckets:
        triples = []

        # Retrieve and process the triples
        for hit in bucket["hits"]["hits"]["hits"]:
            source = hit["_source"]
            triple = f"{source['sid']} | {source['rid']} | {source['oid']}"
            triples.append(triple)

        if triples:
            data.append(triples)

            # Save the triples in the appropriate directory
            num_triples = len(triples)
            file_path = DUMP_PATH[dataset_name].joinpath(f"{num_triples}triples").joinpath(f"{bucket['key']['sid']}.json")
            file_path.parent.mkdir(exist_ok=True, parents=True)

            json.dump({"data": [triples]}, file_path.open("w"))

    return data


# Plan:
# first, we connect to elasticsearch
if __name__ == '__main__':
    client = connect_to_elasticsearch()
    aop = ESActionOperator(client)
    all_subjs = client.search(index=aop.index_rdf,
                              query={'match_all': {}},
                              size=10)

    sid_agg_with_oid = client.search(
        index=aop.index_rdf,
        size=0,
        body={
            "query": {
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
            "aggs": {
                "group_by_sid": {
                    "composite": {
                        "size": 1000,  # Fetch up to 1000 buckets per request
                        "sources": [
                            {
                                "sid": {
                                    "terms": {
                                        "field": "sid"
                                    }
                                }
                            }
                        ]
                    },
                    "aggs": {
                        "hits": {
                            "top_hits": {
                                "size": 3
                            }
                        }
                    }
                }
            }
        }
    )

    all_buckets = []
    dump_num = 0
    while True:
        response = sid_agg_with_oid
        sid_buckets = response['aggregations']['group_by_sid']['buckets']
        after_key = response['aggregations']['group_by_sid'].get('after_key')

        # Append the buckets to the all_buckets list
        all_buckets.extend(sid_buckets)

        # Process sid_buckets or do something with them
        # ...

        if after_key:
            sid_agg_with_oid = client.search(
                index=aop.index_rdf,
                size=0,
                query={"bool": {
                            "must_not": [
                                {
                                    "match": {
                                        "oid": ""
                                    }
                                }
                            ]
                        }
                    },
                aggs={
                        "group_by_sid": {
                            "composite": {
                                "size": 1000,
                                "sources": [
                                    {
                                        "sid": {
                                            "terms": {
                                                "field": "sid"
                                            }
                                        }
                                    }
                                ],
                                "after": after_key  # Continue from the last bucket
                            },
                            "aggs": {
                                "hits": {
                                    "top_hits": {
                                        "size": 3
                                    }
                                }
                            }
                        }
                    }
                )

            if len(all_buckets) >= BUCKET_SAVE_FREQUENCY:
                # split buckets
                shuffle(all_buckets)
                num_buckets = len(all_buckets)
                train_buckets = all_buckets[:int(num_buckets * SPLIT['train'])]
                test_buckets = all_buckets[
                               int(num_buckets * SPLIT['train']): int(num_buckets * (SPLIT['train'] + SPLIT['test']))]
                dev_buckets = all_buckets[int(num_buckets * (SPLIT['train'] + SPLIT['test'])):]

                # dump splits
                train_data = process_and_save_buckets(train_buckets, 'train')
                test_data = process_and_save_buckets(test_buckets, 'test')
                dev_data = process_and_save_buckets(dev_buckets, 'dev')

                # dump full unprocessed bucket
                json.dump(all_buckets, DUMP_PATH['buckets'].joinpath(f"all_buckets-{dump_num}.json").open("w"))
                print(len(all_buckets))
                print(f"{len(train_data)}, {len(test_data)}, {len(dev_data)}")

                # empty all_buckets
                dump_num += 1
                all_buckets = []
        else:
            # split buckets
            shuffle(all_buckets)
            num_buckets = len(all_buckets)
            train_buckets = all_buckets[:int(num_buckets * SPLIT['train'])]
            test_buckets = all_buckets[
                           int(num_buckets * SPLIT['train']): int(num_buckets * (SPLIT['train'] + SPLIT['test']))]
            dev_buckets = all_buckets[int(num_buckets * (SPLIT['train'] + SPLIT['test'])):]

            # dump splits
            train_data = process_and_save_buckets(train_buckets, 'train')
            test_data = process_and_save_buckets(test_buckets, 'test')
            dev_data = process_and_save_buckets(dev_buckets, 'dev')

            # dump full unprocessed bucket
            json.dump(all_buckets, DUMP_PATH['buckets'].joinpath(f"all_buckets-{dump_num}.json").open("w"))
            print(f"all: {len(all_buckets)}")
            print(f"train: {len(train_data)}, test: {len(test_data)}, dev: {len(dev_data)}")

            # empty all_buckets
            dump_num += 1
            break

    print(dump_num)