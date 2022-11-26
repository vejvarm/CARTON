import time
import logging
import transaction
import os
from pathlib import Path

from BTrees import OOBTree
from BTrees.OOBTree import TreeSet

import ujson
import ZODB
from tqdm import tqdm

from args import KGType
from constants import args

from helpers import setup_logger, connect_to_elasticsearch

ROOT_PATH = Path(os.path.dirname(__file__))
LOGGER = setup_logger(__name__, logging.INFO)


class KGLoader:
    kg_paths = {KGType.MEMORY: ROOT_PATH,
                KGType.ZODB: f"{ROOT_PATH}/Wikidata.fs",
                KGType.ELASTICSEARCH: None}

    @classmethod
    def load_kg(cls, kg_type: str = args.kg_type):
        """Load one of implemented KG types"""
        if kg_type not in [tp.value for tp in KGType]:
            raise NotImplementedError("Specified KG kg_type is not a valid. Aborting")

        if kg_type == KGType.MEMORY.value:
            return KGWikidataJSON(cls.kg_paths[KGType.MEMORY])
        if kg_type == KGType.ZODB.value:
            return KGZODB(cls.kg_paths[KGType.ZODB], run_adapter=True)
        if kg_type == KGType.ELASTICSEARCH.value:
            return connect_to_elasticsearch()


class KGZODB:

    def __init__(self, path_to_db_file: str, initialise=False, run_adapter=False):
        self.dbase = None
        self.conn = None
        self.root = None
        self.path_to_db_file = path_to_db_file

        # for adapting to CARTON-like KG implementation
        self.entity_type = {}
        self.labels = {}
        self.triples = {}

        self.open(path_to_db_file)

        if initialise:
            self.initialise_structure()

        if run_adapter:
            self.kg_adapter()

    def open(self, path_to_db_file=None):
        if self.conn is None:
            self.dbase = ZODB.DB(path_to_db_file)
            self.conn = self.dbase.open()
            self.root = self.conn.root
        elif self.conn.opened:
            print(f'Connection {self.conn.opened} already opened. Skipping Command.')
        elif not self.dbase.databases:
            if path_to_db_file is None:
                print(f'Database was closed. Reinitialising from last path_to_db_file: {self.path_to_db_file}.')
                self.__init__(self.path_to_db_file)
            else:
                print(f'Database was closed. Reinitialising from new path: {path_to_db_file}')
                self.__init__(path_to_db_file)
        else:  # conneciton is closed but not None
            self.conn.open()
            print(f'Database open but connection closed. Connection {self.conn.opened} reopened.')

    def kg_adapter(self):
        self.entity_type = self.root.entity_type

        self.labels = {
            'entity': self.root.id_entity,   # dict[e] -> label
            'relation': self.root.id_relation,   # dict[r] -> label
            'inverse': self.root.inverse_entity if 'inverse_entity' in self.root() else None  # dict[label] -> entity
        }

        self.triples = {
            'subject': self.root.subject_triples,  # dict[s][r] -> [o1, o2, o3]
            'object': self.root.object_triples,  # dict[o][r] -> [s1, s2, s3]
            'relation': {
                'subject': self.root.relation_subject_object,  # dict[r][s] -> [o1, o2, o3]
                'object': self.root.relation_object_subject  # dict[r][o] -> [s1, s2, s3]
            },
            'type': self.root.type_triples  # dict[t][r] -> [t1, t2, t3]
        }

        if self.labels['inverse'] is None:
            print('invert labels')
            self.invert_labels()
            self.labels['inverse'] = self.root.inverse_entity

    def invert_labels(self):
        if 'inverse_entity' not in self.root():
            self.root.inverse_entity = OOBTree.BTree()
            print('Inverse entity tree not initialised. Initialising new one.')

        for k, v in self.root.id_entity.items():
            self.root.inverse_entity[v] = k  # TODO: deal with duplicate labels

        assert len(self.root.inverse_entity) > 0
        print('Inverting successful.')

    # UPDATING KG
    def check_label_existance(self, sr: str, lab: str):
        """ Check for entity/relation label existance in KG

        param sr: subject or relation
        param lab: label for the subjet or relation
        """
        if sr[0] == "Q":
            try:
                entry = self.root.id_entity[sr]
                print(f"Entity {sr} already exists with label '{entry}'. Should I change it to '{lab}'?")
            except KeyError:
                print(f"Entity {sr} doesn't exist yet. Should I assign it with label '{lab}'?")
                # self.root.id_entity[sr] = lab
        elif sr[0] == "P":
            try:
                entry = self.root.id_relation[sr]
                print(f"Relation {sr} already exists with label '{entry}'. Should I change it to '{lab}'?")
            except KeyError:
                print(f"Relation {sr} doesn't exist yet. Should I assign it with label '{lab}'?")

    def add_label(self, sr: str, lab: str = None):
        """ Update entity/relation label mapping in KG

        !warning, destructive method, run check_label_existance first

        param sr: subject or relation
        param lab: label for the entity or relation (can be None == marked for labeling)
        """

        if sr[0] == 'Q':
            if lab is None and sr in self.root.id_entity.keys():
                print(f"entity {sr} already exists with label '{self.root.id_entity[sr]}'.")
            else:
                self.root.id_entity[sr] = lab
        elif sr[0] == 'P':
            if lab is None and sr in self.root.id_relation.keys():
                print(f"relation {sr} already exists with label '{self.root.id_relation[sr]}'.")
            else:
                self.root.id_relation[sr] = lab
        else:
            raise KeyError("First letter of ID must be either 'Q' (entity) or 'P' (relation).")

    def insert(self, s: str, r: str, o: list[str], replace=False):
        """
        param s: subject
        param r: relation
        param o: objects (len: 1 to n)
        """
        self.update_sub_rel_ob(s, r, o, replace)
        self.update_rel_sub_ob(r, s, o, replace)
        for ob in o:
            self.update_rel_ob_sub(r, ob, [s], replace)
            self.update_ob_rel_sub(ob, r, [s], replace)

        # initialise new entities in label maps (with lab=None)
        for e in [s, r, *o]:
            self.add_label(e)  # if entity already exists it will not change

        print("All entries succesfully updated")

    def _del_entity(self, e: str, mapping: OOBTree.BTree | OOBTree.TreeSet):

        try:
            return mapping.pop(e)
            # check for subjects, properties and objects to delete from other KG structures
        except KeyError:
            print(f"entity '{e}' is not present in {mapping}")
            return None

    def remove_subject(self, s):
        """Completely removes subject from KG and all RDF entries it has within the KG"""

        deleted_map = self._del_entity(s, self.root.subject_triples)

        if deleted_map is not None:
            for rel, object_list in deleted_map.items():
                try:
                    objects = self.root.relation_subject_object[rel].pop(s)
                    print(f"Removed {objects} from rel_sub_ob")
                except KeyError:
                    print('No objects deleted from rel_sub_ob triple')
                for ob in object_list:
                    try:
                        self.root.relation_object_subject[rel][ob].remove(s)
                        print(f"Removed {s} from rel_ob_sub")
                        if not self.root.relation_object_subject[rel][ob]:
                            print(f"popped empty object array {self.root.relation_object_subject[rel].pop(ob)}")
                    except KeyError:
                        print('No subjects deleted from rel_ob_sub triple')
                        continue
                    try:
                        subject = self.root.object_triples[ob][rel].remove(s)
                        print(f"Removed {subject} from rel_ob_sub")
                    except KeyError:
                        print('No subjects deleted from ob_rel_sub triple')

    def update_sub_rel_ob(self, s, r, o: list[str], replace=False):
        if s in self.root.subject_triples.keys():
            if replace or r not in self.root.subject_triples[s].keys():
                self.root.subject_triples[s][r] = o  # add new objets entry for given s/r pair
            elif r in self.root.subject_triples[s].keys():
                self.root.subject_triples[s][r].extend(o)  # add more objects to existing entry
            else:
                raise NotImplemented(f"Operation not implemented.")
        else:  # subject is not yet in KG
            self.root.subject_triples[s] = OOBTree.BTree({r: o})  # add entirely new rdf entry to KG

    def update_ob_rel_sub(self, o, r, s: list[str], replace=False):
        if o in self.root.object_triples.keys():
            if replace or r not in self.root.object_triples[o].keys():
                self.root.object_triples[o][r] = s  # add new objets entry for given o/r pair
            elif r in self.root.object_triples[o].keys():
                self.root.object_triples[o][r].extend(s)  # add more objects to existing entry
            else:
                raise NotImplemented(f"Operation not implemented.")
        else:  # subject is not yet in KG
            self.root.object_triples[o] = OOBTree.BTree({r: s})  # add entirely new rdf entry to KG

    def update_rel_sub_ob(self, r, s, o: list[str], replace=False):
        try:
            if s in self.root.relation_subject_object[r].keys():
                if replace:
                    self.root.relation_subject_object[r][s] = o        # replace existing entries
                else:
                    self.root.relation_subject_object[r][s].extend(o)  # extend existing entries
            else:
                self.root.relation_subject_object[r][s] = o
        except KeyError:
            self.root.relation_subject_object[r] = OOBTree.BTree({s: o})  # add new entry entirely

    def update_rel_ob_sub(self, r, o, s: list[str], replace=False):
        try:
            if o in self.root.relation_object_subject[r].keys():
                if replace:
                    self.root.relation_object_subject[r][o] = s        # replace existing entries
                else:
                    self.root.relation_object_subject[r][o].extend(s)  # extend existing entries
            else:
                self.root.relation_object_subject[r][o] = s
        except KeyError:
            self.root.relation_object_subject[r] = OOBTree.BTree({o: s})  # add new entry entirely

    def update_entry(self, ):
        pass  # ANCHOR Which way to update

    @staticmethod
    def _fill_oobtree(input_dict: dict, db_tree: OOBTree):
        total_entries = len(input_dict)
        for i, (key, val) in enumerate(input_dict.items()):
            progress = i * 100 // total_entries
            if isinstance(val, dict):
                # we expect val to be btree map of list entries
                persist_map = OOBTree.BTree()
                for k, v in val.items():
                    # fill tree map with the dictionary values
                    if isinstance(v, list):
                        persist_map[k] = TreeSet(v)  # fill new map with BTree Set structure
                    else:
                        print(f"this is not a list: {v}")
                db_tree[key] = persist_map
            else:
                db_tree[key] = val

            if progress > (i-1)*100//total_entries:
                print(f"Progress: {progress}%")

    @staticmethod
    def commit():
        transaction.commit()

    @staticmethod
    def savepoint():
        transaction.savepoint(True)

    def close(self, commit=True):
        if commit:
            self.commit()
        self.conn.close()
        self.dbase.close()

    def initialise_structure(self):
        """ Initialize structure of the database"""
        # root.sub_pred_ob = OOBTree.BTree()
        # root.ob_pred_sub = OOBTree.BTree() # ANCHOR 1
        # labels
        self.root.id_entity = OOBTree.BTree()
        self.root.id_relation = OOBTree.BTree()
        self.root.inverse_entity = OOBTree.BTree()
        # triples
        self.root.subject_triples = OOBTree.BTree()
        self.root.object_triples = OOBTree.BTree()
        self.root.relation_subject_object = OOBTree.BTree()
        self.root.relation_object_subject = OOBTree.BTree()
        self.root.type_triples = OOBTree.BTree()
        self.root.entity_type = OOBTree.BTree()

    def fill_from_kg(self, kg):

        # fill LABELS
        print("Filling Label maps...")
        self._fill_oobtree(kg.id_entity, self.root.id_entity)
        print("\tid_entity filled.")
        self._fill_oobtree(kg.id_relation, self.root.id_relation)
        print("\tid_relation filled.")
        # fill inverse index label/entity_id BTree
        self.invert_labels()

        # fill TRIPLES
        print("Filling Triples maps...")
        self._fill_oobtree(kg.subject_triples, self.root.subject_triples)
        print("\tsubject_triples filled.")
        self._fill_oobtree(kg.object_triples, self.root.object_triples)
        print("\tobject_triples filled.")
        self._fill_oobtree(kg.relation_subject_object, self.root.relation_subject_object)
        print("\trelation_subject_object filled.")
        self._fill_oobtree(kg.relation_object_subject, self.root.relation_object_subject)
        print("\trelation_object_subject filled.")
        self._fill_oobtree(kg.type_triples, self.root.type_triples)
        print("\ttype_triples filled.")

    def fill_from_dict(self, input_dict, tree_to_fill: OOBTree):
        self._fill_oobtree(input_dict, tree_to_fill)
        # self.savepoint()
        self.commit()
        print(f"Tree {tree_to_fill} filled and savepoint created.")

    def fill_from_json(self, path_to_json: str, tree_to_fill: OOBTree):
        loaded_dict = ujson.loads(open(path_to_json).read())
        self._fill_oobtree(loaded_dict, tree_to_fill)
        # self.savepoint()
        self.commit()
        print(f"Tree from path: {path_to_json} filled and savepoint created.")


class KGJSON:

    def __init__(self, root_folder: str):
        """ General class for loading KG information to Dictionary structures in memory from json files.

        :param root_folder: (str) path to folder where data is located
        """
        self.root_folder = root_folder

        # labels
        self.labels = {
            'entity': dict(),  # dict[e] -> label
            'relation': dict()  # dict[r] -> label
        }

        # triples
        self.triples = {
            'subject': dict(),  # dict[s][r] -> [o1, o2, o3]
            'object': dict(),  # dict[o][r] -> [s1, s2, s3]
            'relation': {
                'subject': dict(),  # dict[r][s] -> [o1, o2, o3]
                'object': dict()  # dict[r][o] -> [s1, s2, s3]
            },
            'type': dict()  # dict[t][r] -> [t1, t2, t3]
        }

        # id -> entity type
        self.entity_type = dict()  # dict[e] -> type

    def load_json(self, filename: str):
        tic = time.perf_counter()
        fullpath = f"{self.root_folder}/{filename}"
        try:
            LOGGER.info(f'Loaded {filename} in {time.perf_counter() - tic:0.2f}s')
            return ujson.loads(open(fullpath).read())
        except FileNotFoundError:
            LOGGER.warning(f"File at path {fullpath} doesn't exist. Returning empty dictionary.")
            return {}


class MiniKGWikidataJSON(KGJSON):
    def __init__(self, root_folder=f'{ROOT_PATH}'):
        super().__init__(root_folder)

        # labels
        self.labels['entity'] = self.load_json("items_wikidata_n.json")

        # entity -> type
        self.entity_type = self.load_json("entity_type.json")


class KGWikidataJSON(KGJSON):
    def __init__(self, root_folder=f'{ROOT_PATH}'):
        super().__init__(root_folder)

        # labels
        self.labels['entity'] = self.load_json("items_wikidata_n.json")
        self.labels['relation'] = self.load_json("filtered_property_wikidata4.json")

        # triples
        self.triples['subject'] = {**self.load_json("wikidata_short_1.json"), **self.load_json("wikidata_short_2.json")}
        self.triples['object'] = self.load_json("comp_wikidata_rev.json")
        self.triples['relation']['subject'] = self.load_json("relation_subject_object.json")
        self.triples['relation']['object'] = self.load_json("relation_object_subject.json")
        self.triples['type'] = self.load_json("wikidata_type_dict.json")

        # entity -> type
        self.entity_type = self.load_json("entity_type.json")


class KG2ElasticIndexMigrator:
    def __init__(self, root_folder=f'{ROOT_PATH}'):
        self.root_folder = root_folder

    def construct_index_ent_dict(self, merge=True, update_entries=True, dump_to=None):
        """ Index for storing labels and types of entities retrievable by entitiy id
             - final structure:
                {eid[str]: {label: lb[str], type: tp[str]}}

        """
        tic = time.perf_counter()
        # # ...eid -> entity label
        id_entity = ujson.loads(open(f'{self.root_folder}/items_wikidata_n.json').read())
        LOGGER.info(f'Loaded id_entity {time.perf_counter() - tic:0.2f}s')

        # ... eid -> list[type id]
        entity_type = ujson.loads(open(f'{self.root_folder}/entity_type.json').read())  # dict[e] -> type
        LOGGER.info(f'Loaded entity_type {time.perf_counter() - tic:0.2f}s')

        LOGGER.info(f"id_entity len: {len(id_entity)}")
        LOGGER.info(f"entity_type len: {len(entity_type)}")

        if merge:
            # ... eid -> type id
            child_par = ujson.loads(open(f'{self.root_folder}/child_par_dict_immed.json').read())  # dict[e] -> type
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

        index_rel_dict = ujson.loads(open(f'{self.root_folder}/filtered_property_wikidata4.json').read())
        LOGGER.info(f'Loaded id_relation {time.perf_counter()-tic:0.2f}s')

        if dump_to is not None and isinstance(dump_to, str):
            ujson.dump(index_rel_dict, open(f'{ROOT_PATH}/{dump_to}', 'w'), indent=4, escape_forward_slashes=False)

    # NOTE: this is useless ... because the subject_triples already has everything from ors, rso and ros
    def construct_index_rdf_dict(self, dump_to=None):
        tic = time.perf_counter()

        # subject -> relation -> object  # DONE
        subject_triples = {**ujson.loads(open(f'{self.root_folder}/wikidata_short_1.json').read()),
                           **ujson.loads(open(f'{self.root_folder}/wikidata_short_2.json').read())}
        LOGGER.info(f'Loaded subject_triples {time.perf_counter()-tic:0.2f}s')

        # object -> relation -> subject  # DONE
        # object_triples = ujson.loads(open(f'{self.root_folder}/comp_wikidata_rev.json').read())
        # LOGGER.info(f'Loaded object_triples {time.perf_counter()-tic:0.2f}s')

        # # relation -> subject -> object | relation -> object -> subject  # DONE
        # relation_subject_object = ujson.loads(open(f'{self.root_folder}/relation_subject_object.json').read())
        # relation_object_subject = ujson.loads(open(f'{self.root_folder}/relation_object_subject.json').read())
        # LOGGER.info(f'Loaded relation_triples {time.perf_counter()-tic:0.2f}s')

        # 3.a: transform all triples into unique s - r - o RDFs
        # 3.b: check if every entity in RDF has label and type
        # 3.c: add rdf entries to index

        # subject_triples are good as they are ... check against the subjects in those
        # LOGGER.info(f'Loaded all necessary KG files.')
        # LOGGER.info(f'RDF count: {len(subject_triples)}')
        # LOGGER.info('Starting object-rel-subject triple migration:')
        # sub_rel_objs_added = 0
        # rel_objs_added = 0
        # objs_added = 0
        # for oid, relsub_dict in tqdm(object_triples.items()):
        #     for rid, sid_list in relsub_dict.items():
        #         # exists already?
        #         for sid in sid_list:
        #             if sid in subject_triples.keys():
        #                 if rid in subject_triples[sid].keys():
        #                     if oid in subject_triples[sid][rid]:
        #                         pass  # this rdf already exists, so don't do anything
        #                     else:  # object not present in object list for this s - r -
        #                         subject_triples[sid][rid].append(oid)
        #                         objs_added += 1
        #                 else: # subject exists but relation doesnt for this one
        #                     subject_triples[sid][rid] = [oid, ]
        #                     rel_objs_added += 1
        #             else:  # subject doesn't exist yet
        #                 subject_triples[sid] = {rid: [oid, ]}
        #                 sub_rel_objs_added += 1
        #
        # LOGGER.info(f'\t results: +sro_count: {sub_rel_objs_added} | +ro_count {rel_objs_added} | +o_count {objs_added}')
        # LOGGER.info(f'RDF count: {len(subject_triples)}')
        #
        # LOGGER.info('Starting rel-subject-object triple migration:')
        # sub_rel_objs_added = 0
        # rel_objs_added = 0
        # objs_added = 0
        # for rid, subobj_dict in tqdm(relation_subject_object.items()):
        #     for sid, oid_list in subobj_dict.items():
        #         for oid in oid_list:
        #             # exists already?
        #             if sid in subject_triples.keys():
        #                 if rid in subject_triples[sid].keys():
        #                         if oid in subject_triples[sid][rid]:
        #                             pass  # this rdf already exists, so don't do anything
        #                         else:  # object not present in object list for this s - r -
        #                             subject_triples[sid][rid].append(oid)
        #                             objs_added += 1
        #                 else: # subject exists but relation doesnt for this one
        #                     subject_triples[sid][rid] = [oid, ]
        #                     rel_objs_added += 1
        #             else:  # subject doesn't exist yet
        #                 subject_triples[sid] = {rid: [oid, ]}
        #                 sub_rel_objs_added += 1
        #
        # LOGGER.info(f'\t results: +sro_count: {sub_rel_objs_added} | +ro_count {rel_objs_added} | +o_count {objs_added}')
        # LOGGER.info(f'RDF count: {len(subject_triples)}')
        #
        # LOGGER.info('Starting rel-object-subject triple migration:')
        # sub_rel_objs_added = 0
        # rel_objs_added = 0
        # objs_added = 0
        # for rid, objsub_dict in tqdm(relation_object_subject.items()):
        #     for oid, sid_list in objsub_dict.items():
        #         for sid in sid_list:
        #             # exists already?
        #             if sid in subject_triples.keys():
        #                 if rid in subject_triples[sid].keys():
        #                     if oid in subject_triples[sid][rid]:
        #                         pass  # this rdf already exists, so don't do anything
        #                     else:  # object not present in object list for this s - r -
        #                         subject_triples[sid][rid].append(oid)
        #                         objs_added += 1
        #                 else: # subject exists but relation doesnt for this one
        #                     subject_triples[sid][rid] = [oid, ]
        #                     rel_objs_added += 1
        #             else:  # subject doesn't exist yet
        #                 subject_triples[sid] = {rid: [oid, ]}
        #                 sub_rel_objs_added += 1
        #
        # LOGGER.info(
        #     f'\t results: +sro_count: {sub_rel_objs_added} | +ro_count {rel_objs_added} | +o_count {objs_added}')
        # LOGGER.info(f'RDF count: {len(subject_triples)}')

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
