import copy
import logging
import os
import ujson
import time
from pathlib import Path

from tqdm import tqdm

ROOT_PATH = Path(os.path.dirname(__file__))

LOGGER = logging.getLogger(__name__)


class MiniKG:
    def __init__(self, wikidata_path=f'{ROOT_PATH}'):
        tic = time.perf_counter()

        # id -> entity label
        self.id_entity = ujson.loads(open(f'{wikidata_path}/items_wikidata_n.json').read())
        LOGGER.info(f'Loaded id_entity {time.perf_counter()-tic:0.2f}s')

        # entity -> type
        self.entity_type = ujson.loads(open(f'{wikidata_path}/entity_type.json').read()) # dict[e] -> type
        LOGGER.info(f'Loaded entity_type {time.perf_counter()-tic:0.2f}s')

        # labels
        self.labels = {
            'entity': self.id_entity,  # dict[e] -> label
        }


class KnowledgeGraph:
    def __init__(self, wikidata_path=f'{ROOT_PATH}'):
        tic = time.perf_counter()

        # id -> entity label
        self.id_entity = ujson.loads(open(f'{wikidata_path}/items_wikidata_n.json').read())
        LOGGER.info(f'Loaded id_entity {time.perf_counter()-tic:0.2f}s')

        # id -> relation label
        self.id_relation = ujson.loads(open(f'{wikidata_path}/filtered_property_wikidata4.json').read())
        LOGGER.info(f'Loaded id_relation {time.perf_counter()-tic:0.2f}s')

        # entity -> type
        self.entity_type = ujson.loads(open(f'{wikidata_path}/entity_type.json').read()) # dict[e] -> type
        LOGGER.info(f'Loaded entity_type {time.perf_counter()-tic:0.2f}s')

        # type -> relation -> type
        self.type_triples = ujson.loads(open(f'{wikidata_path}/wikidata_type_dict.json').read())
        LOGGER.info(f'Loaded type_triples {time.perf_counter()-tic:0.2f}s')

        # subject -> relation -> object
        self.subject_triples_1 = ujson.loads(open(f'{wikidata_path}/wikidata_short_1.json').read())
        self.subject_triples_2 = ujson.loads(open(f'{wikidata_path}/wikidata_short_2.json').read())
        self.subject_triples = {**self.subject_triples_1, **self.subject_triples_2}
        LOGGER.info(f'Loaded subject_triples {time.perf_counter()-tic:0.2f}s')

        # object -> relation -> subject
        self.object_triples = ujson.loads(open(f'{wikidata_path}/comp_wikidata_rev.json').read())
        LOGGER.info(f'Loaded object_triples {time.perf_counter()-tic:0.2f}s')

        # relation -> subject -> object | relation -> object -> subject
        self.relation_subject_object = ujson.loads(open(f'{wikidata_path}/relation_subject_object.json').read())
        self.relation_object_subject = ujson.loads(open(f'{wikidata_path}/relation_object_subject.json').read())
        LOGGER.info(f'Loaded relation_triples {time.perf_counter()-tic:0.2f}s')

        # labels
        self.labels = {
            'entity': self.id_entity, # dict[e] -> label
            'relation': self.id_relation # dict[r] -> label
        }

        # triples
        self.triples = {
            'subject': self.subject_triples,  # dict[s][r] -> [o1, o2, o3]
            'object': self.object_triples,  # dict[o][r] -> [s1, s2, s3]
            'relation': {
                'subject': self.relation_subject_object,  # dict[r][s] -> [o1, o2, o3]
                'object': self.relation_object_subject  # dict[r][o] -> [s1, s2, s3]
            },
            'type': self.type_triples  # dict[t][r] -> [t1, t2, t3]
        }


class KGMigrator:
    def __init__(self, wikidata_path=f'{ROOT_PATH}'):
        self.wikidata_path = wikidata_path

    def construct_index_ent_dict(self, merge=True, update_entries=True, dump_to=None):
        """ Index for storing labels and types of entities retrievable by entitiy id
             - final structure:
                {eid[str]: {label: lb[str], type: tp[str]}}

        """
        tic = time.perf_counter()
        # # ...eid -> entity label
        id_entity = ujson.loads(open(f'{self.wikidata_path}/items_wikidata_n.json').read())
        LOGGER.info(f'Loaded id_entity {time.perf_counter() - tic:0.2f}s')

        # ... eid -> list[type id]
        entity_type = ujson.loads(open(f'{self.wikidata_path}/entity_type.json').read())  # dict[e] -> type
        LOGGER.info(f'Loaded entity_type {time.perf_counter() - tic:0.2f}s')

        LOGGER.info(f"id_entity len: {len(id_entity)}")
        LOGGER.info(f"entity_type len: {len(entity_type)}")

        if merge:
            # ... eid -> type id
            child_par = ujson.loads(open(f'{self.wikidata_path}/child_par_dict_immed.json').read())  # dict[e] -> type
            LOGGER.info(f'Loaded child_par_dict_immed {time.perf_counter() - tic:0.2f}s')

            LOGGER.info(f"Merging entity_type with child_par_dict_immed")
            added = 0
            updated = 0
            for eid in tqdm(child_par.keys()):
                if eid not in entity_type.keys():
                    entity_type[eid] = [child_par[eid]]
                    added += 1
                else:
                    if update_entries:
                        LOGGER.debug(f"BEFORE: {entity_type[eid]}")
                        len_before = len(entity_type[eid])
                        entity_type[eid].append(child_par[eid])
                        entity_type[eid] = list(set(entity_type[eid]))
                        LOGGER.debug(f"AFTER: {entity_type[eid]}")
                        if len(entity_type[eid]) > len_before:
                            updated += 1

            LOGGER.info(f"entity_type len: {len(entity_type)} (after merge)")
            LOGGER.info(f"\t added: {added}")
            LOGGER.info(f"\t updated: {updated}")

        all_entity_ids = set(id_entity.keys()).union(entity_type.keys())
        index_ent_dict = dict()
        missing_types = set()
        missing_labels = set()

        for eid in tqdm(all_entity_ids):
            if eid in entity_type.keys() and eid in id_entity.keys():  # both label and type are present
                index_ent_dict[eid] = {'label': id_entity[eid], 'types': entity_type[eid]}
            elif eid in entity_type.keys():
                index_ent_dict[eid] = {'label': None, 'types': entity_type[eid]}
                LOGGER.debug(f"entity id [{eid}] is missing from 'items_wikidata_n.json' -> 'label' field was set to None")
                missing_labels.update([eid])
            elif eid in id_entity.keys():
                index_ent_dict[eid] = {'label': id_entity[eid], 'types': []}
                LOGGER.debug(f"entity id [{eid}] is missing from 'entity_type.json' -> 'types' field was set to []")
                missing_types.update([eid])
            else:   # both not present - should be impossible under these circumstances, but better be safe
                index_ent_dict[eid] = {'label': None, 'types': []}
                LOGGER.warning(f"Somehow entity id [{eid}] is missing from both json files. How did this happen?")
                missing_labels.update([eid])
                missing_types.update([eid])

        LOGGER.info(f'missing_labels: {len(missing_labels)}\nmissing_types: {len(missing_types)}')

        if dump_to is not None and isinstance(dump_to, str):
            ujson.dump(index_ent_dict, open(f'{ROOT_PATH}/{dump_to}', 'w'), indent=4, escape_forward_slashes=False)

        return index_ent_dict

    def construct_index_rel_dict(self, dump_to=None):
        tic = time.perf_counter()

        index_rel_dict = ujson.loads(open(f'{self.wikidata_path}/filtered_property_wikidata4.json').read())
        LOGGER.info(f'Loaded id_relation {time.perf_counter()-tic:0.2f}s')

        if dump_to is not None and isinstance(dump_to, str):
            ujson.dump(index_rel_dict, open(f'{ROOT_PATH}/{dump_to}', 'w'), indent=4, escape_forward_slashes=False)

    # NOTE: this is useless ... because the subject_triples already has everything from ors, rso and ros
    def construct_index_rdf_dict(self, dump_to=None):
        tic = time.perf_counter()

        # subject -> relation -> object  # DONE
        subject_triples = {**ujson.loads(open(f'{self.wikidata_path}/wikidata_short_1.json').read()),
                           **ujson.loads(open(f'{self.wikidata_path}/wikidata_short_2.json').read())}
        LOGGER.info(f'Loaded subject_triples {time.perf_counter()-tic:0.2f}s')

        # object -> relation -> subject  # DONE
        object_triples = ujson.loads(open(f'{self.wikidata_path}/comp_wikidata_rev.json').read())
        LOGGER.info(f'Loaded object_triples {time.perf_counter()-tic:0.2f}s')

        # relation -> subject -> object | relation -> object -> subject  # DONE
        relation_subject_object = ujson.loads(open(f'{self.wikidata_path}/relation_subject_object.json').read())
        relation_object_subject = ujson.loads(open(f'{self.wikidata_path}/relation_object_subject.json').read())
        LOGGER.info(f'Loaded relation_triples {time.perf_counter()-tic:0.2f}s')

        # 3.a: transform all triples into unique s - r - o RDFs
        # 3.b: check if every entity in RDF has label and type
        # 3.c: add rdf entries to index

        # subject_triples are good as they are ... check against the subjects in those
        LOGGER.info(f'Loaded all necessary KG files.')
        LOGGER.info(f'RDF count: {len(subject_triples)}')
        LOGGER.info('Starting object-rel-subject triple migration:')
        sub_rel_objs_added = 0
        rel_objs_added = 0
        objs_added = 0
        for oid, relsub_dict in tqdm(object_triples.items()):
            for rid, sid_list in relsub_dict.items():
                # exists already?
                for sid in sid_list:
                    if sid in subject_triples.keys():
                        if rid in subject_triples[sid].keys():
                            if oid in subject_triples[sid][rid]:
                                pass  # this rdf already exists, so don't do anything
                            else:  # object not present in object list for this s - r -
                                subject_triples[sid][rid].append(oid)
                                objs_added += 1
                        else: # subject exists but relation doesnt for this one
                            subject_triples[sid][rid] = [oid, ]
                            rel_objs_added += 1
                    else:  # subject doesn't exist yet
                        subject_triples[sid] = {rid: [oid, ]}
                        sub_rel_objs_added += 1

        LOGGER.info(f'\t results: +sro_count: {sub_rel_objs_added} | +ro_count {rel_objs_added} | +o_count {objs_added}')
        LOGGER.info(f'RDF count: {len(subject_triples)}')

        LOGGER.info('Starting rel-subject-object triple migration:')
        sub_rel_objs_added = 0
        rel_objs_added = 0
        objs_added = 0
        for rid, subobj_dict in tqdm(relation_subject_object.items()):
            for sid, oid_list in subobj_dict.items():
                for oid in oid_list:
                    # exists already?
                    if sid in subject_triples.keys():
                        if rid in subject_triples[sid].keys():
                                if oid in subject_triples[sid][rid]:
                                    pass  # this rdf already exists, so don't do anything
                                else:  # object not present in object list for this s - r -
                                    subject_triples[sid][rid].append(oid)
                                    objs_added += 1
                        else: # subject exists but relation doesnt for this one
                            subject_triples[sid][rid] = [oid, ]
                            rel_objs_added += 1
                    else:  # subject doesn't exist yet
                        subject_triples[sid] = {rid: [oid, ]}
                        sub_rel_objs_added += 1

        LOGGER.info(f'\t results: +sro_count: {sub_rel_objs_added} | +ro_count {rel_objs_added} | +o_count {objs_added}')
        LOGGER.info(f'RDF count: {len(subject_triples)}')

        LOGGER.info('Starting rel-object-subject triple migration:')
        sub_rel_objs_added = 0
        rel_objs_added = 0
        objs_added = 0
        for rid, objsub_dict in tqdm(relation_object_subject.items()):
            for oid, sid_list in objsub_dict.items():
                for sid in sid_list:
                    # exists already?
                    if sid in subject_triples.keys():
                        if rid in subject_triples[sid].keys():
                            if oid in subject_triples[sid][rid]:
                                pass  # this rdf already exists, so don't do anything
                            else:  # object not present in object list for this s - r -
                                subject_triples[sid][rid].append(oid)
                                objs_added += 1
                        else: # subject exists but relation doesnt for this one
                            subject_triples[sid][rid] = [oid, ]
                            rel_objs_added += 1
                    else:  # subject doesn't exist yet
                        subject_triples[sid] = {rid: [oid, ]}
                        sub_rel_objs_added += 1

        LOGGER.info(
            f'\t results: +sro_count: {sub_rel_objs_added} | +ro_count {rel_objs_added} | +o_count {objs_added}')
        LOGGER.info(f'RDF count: {len(subject_triples)}')

        index_rdf_dict = subject_triples

        if dump_to is not None and isinstance(dump_to, str):
            ujson.dump(index_rdf_dict, open(f'{ROOT_PATH}/{dump_to}', 'w'), indent=4, escape_forward_slashes=False)


def check_consistency(rdf_new, rdf_orig, exact_match=False):
    '''

    :param self:
    :param rdf_new: new dictionary
    :param rdf_orig: original dictionary
    :return:
    '''

    if exact_match:
        assert rdf_new == rdf_orig

    k_set_new = set(rdf_new.keys())
    k_set_orig = set(rdf_orig.keys())

    if exact_match:
        if k_set_new == k_set_orig:
            LOGGER.info('It worked! new and orig is the same')
            return
        else:
            LOGGER.warning('keys in new set are not same as keys in old set.')
            return
    else:
        if k_set_orig.issubset(k_set_new):
            LOGGER.info('New set contains all of original set keys.')
        else:
            LOGGER.warning('New set is missing some keys from the original.')
            return

    LOGGER.info(f"num_keys in rdf_new: {len(k_set_new)}")
    LOGGER.info(f"num_keys in rdf_orig: {len(k_set_orig)}")
    assert len(k_set_new) >= len(k_set_orig), f'rdf_orig appears to have more entries than rdf_new.'

    # check if all existing values stayed the same
    for k_orig, p_dict_orig in rdf_orig.items():
        try:
            p_dict_new = rdf_new[k_orig]
            assert p_dict_orig.keys() in p_dict_new.keys(), f'rdf_new[k_orig] is not the same as rdf_orig[k_orig]'
        except KeyError:
            LOGGER.info(f"Key [{k_orig}] doesn't exist in rdf_new.")
            return

        for p_orig, obj_list_orig in p_dict_orig.keys():
            assert p_orig in p_dict_new.keys(), f"Predicate {p_orig} is not in the original dictionary {p_dict_new}"

            obj_set_new = set(p_dict_new[p_orig])
            obj_set_orig = set(p_dict_orig[p_orig])

            LOGGER.debug(f"new: {obj_set_new} | orig: {obj_set_orig}")

            assert obj_set_orig.issubset(obj_set_new), f"Objects in new set don't contain all objects from old set. (new: {obj_set_new}, orig: {obj_set_orig})"