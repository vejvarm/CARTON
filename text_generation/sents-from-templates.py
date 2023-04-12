import json
import logging
from typing import Dict, List
from ordered_set import OrderedSet
from pathlib import Path

from constants import ROOT_PATH, ElasticIndices
from helpers import connect_to_elasticsearch
from action_executor.actions import ESActionOperator

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.WARNING)

CLIENT = connect_to_elasticsearch()
AOP = ESActionOperator(CLIENT)


def lookup_labels(tripple_ids: OrderedSet[str], es_index: ElasticIndices=ElasticIndices.REL) -> OrderedSet:
    return AOP._get_by_ids(tripple_ids, es_index.value)


def transform_triples_to_json(triples_list: list, templates_dict: Dict) -> List[Dict]:
    transformed_data = []
    missing_relation_labels = set()
    n_missing_err = 0
    # print(triples_list)
    for triple_dict in triples_list:
        print(triple_dict.triples)
        for data_triple in triple_dict.triples:
            relation_id = data_triple.pred
            label = lookup_labels(OrderedSet([relation_id]))[relation_id][0]
            print(label)
            try:
                template = templates_dict[label][0]
            except KeyError:
                missing_relation_labels.add(relation_id)
                n_missing_err += 1
                continue
            ent_labels = lookup_labels(OrderedSet([data_triple.subj, data_triple.obj]), es_index=ElasticIndices.ENT_FULL)
            text = template.replace('<subject>', ent_labels[data_triple.subj][0]).replace('<object>', ent_labels[data_triple.obj][0])
            transformed_data.append({
                'sents': [text],
                'text': text,
                'rdfs': triple_dict.triples
            })
    LOGGER.warning(f"n_errs: {n_missing_err} | relations with missing labels: {missing_relation_labels}")
    return transformed_data


def create_json_files(templates: Dict, triples: Dict, out_folder: str or Path) -> None:

    for subset in ['train', 'dev', 'test']:
        output_data = transform_triples_to_json(triples[subset], templates)
        output = {'data': output_data}
        with out_folder.joinpath(f"{subset}.json").open('w') as outfile:
            json.dump(output, outfile, indent=2)


if __name__ == "__main__":
    from text_generation.dataclasses import WikiData
    wk = WikiData()

    splits = ["dev"]
    root_folder = ROOT_PATH.joinpath("text_generation")
    path_to_data = root_folder.joinpath("data/d2t/wikidata/data")
    path_to_templates = root_folder.joinpath("templates/templates-wikidata.json")
    output_folder = root_folder.joinpath("outputs")

    output_folder.mkdir(exist_ok=True)

    wk.load_from_dir(path_to_data, path_to_templates, splits)

    # print(wk.templates)
    # print(wk.data['dev'])

# Call the function with the required arguments
    create_json_files(wk.templates, wk.data, output_folder)
