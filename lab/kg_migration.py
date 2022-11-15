import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ujson
import logging
from knowledge_graph.knowledge_graph import KGMigrator, check_consistency


logging.basicConfig(level=logging.INFO)


def make_index_jsons():
    kg_migrator = KGMigrator()
    # DONE 1: INDEX_ENT ... eid -> {label: str, types: list[str]}
    kg_migrator.construct_index_ent_dict(merge=True, update_entries=True, dump_to='index_ent_dict.json')
    kg_migrator.construct_index_ent_dict(merge=True, update_entries=False, dump_to='index_ent_dict_merge_but_no_update.json')
    kg_migrator.construct_index_ent_dict(merge=False, update_entries=False, dump_to='index_ent_dict_no_merge.json')
    print('INDEX_ENT: DONE')

    # DONE 2: INDEX_REL ... rid -> relation label
    # NOTE: already exists as default file (filtered_property_wikidata4.json), the migrator just copies the file
    kg_migrator.construct_index_rel_dict(dump_to='index_rel_dict.json')
    print('INDEX_REL: DONE')

    # DONE 3: INDEX_RDF
    kg_migrator.construct_index_rdf_dict(dump_to='index_rdf_dict.json')

    kg_root = "/home/vejvar-martin-nj/git/CARTONwNER/knowledge_graph/"
    rdf_new = ujson.loads(open(f"{kg_root}index_rdf_dict.json").read())

    rdf_orig = {**ujson.loads(open(f'{kg_root}wikidata_short_1.json').read()),
                **ujson.loads(open(f'{kg_root}/wikidata_short_2.json').read())}

    check_consistency(rdf_new, rdf_orig, exact_match=True)
    check_consistency(rdf_new, rdf_orig, exact_match=False)


if __name__ == '__main__':
    # make_index_jsons()
    pass

    # ANCHOR: For filling ES indices, refer to fill_es.py