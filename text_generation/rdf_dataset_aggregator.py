# This file is for generating a dataset of RDF entries, aggregated by having the same subject entity
# for the purpose of D2T generation from the extracted RDF files
import json
import pathlib
from random import shuffle

import logging
import elasticsearch
from wikidata.client import Client as WDClient
from wikidata.entity import EntityId
from helpers import connect_to_elasticsearch, setup_logger
from constants import ROOT_PATH, ElasticIndices
from action_executor.actions import ESActionOperator

BUCKET_SAVE_FREQUENCY = 100000  # subjects/bucket
TEMPLATES_ROOT = ROOT_PATH.joinpath("text_generation/templates")
DATA_DUMP_ROOT = pathlib.Path("/media/freya/kubuntu-data/datasets/text_generation_with_labels")
# DATA_DUMP_ROOT = ROOT_PATH.joinpath("text_generation").joinpath("aggregation_outputs")
LOGFILE_PATH = DATA_DUMP_ROOT.joinpath("aggregation-results.log")
DUMP_PATH = {"buckets": DATA_DUMP_ROOT.joinpath("buckets"),
             "train": DATA_DUMP_ROOT.joinpath("train"),
             "test": DATA_DUMP_ROOT.joinpath("test"),
             "dev": DATA_DUMP_ROOT.joinpath("dev")}
for pth in DUMP_PATH.values():
    pth.mkdir(exist_ok=True, parents=True)

LOGGER = setup_logger(__name__, handlers=[logging.FileHandler(LOGFILE_PATH, 'w'), logging.StreamHandler()])

SPLIT = {"train": 0.8,
          "test": 0.1,
          "dev": 0.1}


def get_label(erid: str, index: ElasticIndices, esclient: elasticsearch.Elasticsearch) -> str:
    try:
        return esclient.get(index=index.value, id=erid)['_source']['label']
    except elasticsearch.NotFoundError:
        LOGGER.warning(f"{erid} not found in {index.value}. Returning original id.")
        return erid


def get_predicate_label_from_wd(predicate_id: str or EntityId, wd_client=WDClient()):
    """ fetch predicate label from WikiData online database based on predicate_id (e.g. "P18" -> "image")"""
    # Get the predicate entity
    pid = EntityId(predicate_id)
    predicate_entity = wd_client.get(pid, load=True)
    return predicate_entity.label


def process_and_save_buckets(buckets, dataset_name: str, esclient: elasticsearch.Elasticsearch, replace_with_labels=False):
    """ Helper function to process and save the triples"""
    data = []

    for bucket in buckets:
        triples = []

        # Retrieve and process the triples
        for hit in bucket["hits"]["hits"]["hits"]:
            source = hit["_source"]
            if replace_with_labels:
                sid = get_label(source['sid'], ElasticIndices.ENT_FULL, esclient)
                rid = get_label(source['rid'], ElasticIndices.REL, esclient)
                oid = get_label(source['oid'], ElasticIndices.ENT_FULL, esclient)
            else:
                sid, rid, oid = source['sid'], source['rid'], source['oid']
            triple = f"{sid} | {rid} | {oid}"

            triples.append(triple)

        if triples:
            data.append(triples)

            # Save the triples in the appropriate directory
            num_triples = len(triples)
            file_path = DUMP_PATH[dataset_name].joinpath(f"{num_triples}triples").joinpath(f"{bucket['key']['sid']}.json")
            file_path.parent.mkdir(exist_ok=True, parents=True)

            json.dump({"data": [triples]}, file_path.open("w"))

    return data


def rdf_query(client, aop, included_pids, buckets_per_query, after_key: dict):
    return client.search(
        index=aop.index_rdf,
        size=0,
        query={
            "bool": {
                "must": [
                    {
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
                    {
                        "terms": {
                            "rid": included_pids
                        }
                    }
                ]
            }
        },
        aggs={
            "group_by_sid": {
                "composite": {
                    "size": buckets_per_query,  # Fetch up to 1000 buckets per request
                    "sources": [
                        {
                            "sid": {
                                "terms": {
                                    "field": "sid"
                                }
                            }
                        }
                    ],
                    "after": after_key
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


# Plan:
# first, we connect to elasticsearch
if __name__ == '__main__':
    buckets_per_query = 1000
    rel_dict = json.load(TEMPLATES_ROOT.joinpath("index_rel_dict.json").open("r"))  # {rid: rlabel, ...}
    included_pids = list(rel_dict.keys())
    repl_with_labels = True
    client = connect_to_elasticsearch()
    aop = ESActionOperator(client)
    response = rdf_query(client, aop, included_pids, buckets_per_query, after_key={'sid': ""})

    all_buckets = []
    dump_num = 0
    while True:
        sid_buckets = response['aggregations']['group_by_sid']['buckets']
        after_key = response['aggregations']['group_by_sid'].get('after_key')

        # Append the buckets to the all_buckets list
        all_buckets.extend(sid_buckets)

        # Process sid_buckets or do something with them
        # ...

        if after_key:
            response = rdf_query(client, aop, included_pids, buckets_per_query, after_key=after_key)

            if len(all_buckets) >= BUCKET_SAVE_FREQUENCY:
                # split buckets
                shuffle(all_buckets)
                num_buckets = len(all_buckets)
                train_buckets = all_buckets[:int(num_buckets * SPLIT['train'])]
                test_buckets = all_buckets[
                               int(num_buckets * SPLIT['train']): int(num_buckets * (SPLIT['train'] + SPLIT['test']))]
                dev_buckets = all_buckets[int(num_buckets * (SPLIT['train'] + SPLIT['test'])):]

                # dump splits
                train_data = process_and_save_buckets(train_buckets, 'train', client, repl_with_labels)
                test_data = process_and_save_buckets(test_buckets, 'test', client, repl_with_labels)
                dev_data = process_and_save_buckets(dev_buckets, 'dev', client, repl_with_labels)

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
            train_data = process_and_save_buckets(train_buckets, 'train', client, repl_with_labels)
            test_data = process_and_save_buckets(test_buckets, 'test', client, repl_with_labels)
            dev_data = process_and_save_buckets(dev_buckets, 'dev', client, repl_with_labels)

            # dump full unprocessed bucket
            json.dump(all_buckets, DUMP_PATH['buckets'].joinpath(f"all_buckets-{dump_num}.json").open("w"))
            print(f"all: {len(all_buckets)}")
            print(f"train: {len(train_data)}, test: {len(test_data)}, dev: {len(dev_data)}")

            # empty all_buckets
            dump_num += 1
            break

    print(dump_num)