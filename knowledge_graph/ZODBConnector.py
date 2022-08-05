import transaction
from BTrees import OOBTree
from BTrees.OOBTree import TreeSet

import ujson
import ZODB
# DONE: Change - dictionaries in KG into PersistentMappigs
#              - lists in KG into PersistentList


class BTreeDB:

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
            'entity': self.root.id_entity,  # dict[e] -> label
            'relation': self.root.id_relation  # dict[r] -> label
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

    def add_rdf(self, s: str, r: str, o: list[str], replace=False):
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
        # TODO: Commit?
        # TODO: Also update id_entity and id_relation

        # DONE: update all other KG objects with the same data

# TODO:    def update_triple(self, s: list[str] | str, r: list[str] | str, o: list[str] | str, map_to_update: OOBTree.BTree, replace=False):

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

    # DONE:
    #   id_entity -- DONE
    #   id_relation -- DONE
    #   subject_triples -- DONE
    #   object_triples -- DONE
    #   relation_subject_object -- DONE
    #   relation_object_subject -- DONE
    #   type_triples  -- IGNORE
    #   entity_type  -- IGNORE
    # TODO:
    #   Test adding ned RDFs to KG (add_rdf function)
    #   Test adding labels to entities/relations

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


if __name__ == "__main__":
    # open and initialise DB object from file
    path_to_db = "./zodb_kg/KGTest.fs"
    db = BTreeDB(path_to_db, initialise=True, run_adapter=True)

    db.labels['entity']['Q1234'] = 'nice name'
    db.labels['relation']['P1'] = 'nice relation'

    s = 'Q0001'
    r = 'P1'
    o = ['Q1001', 'Q1002', 'Q1003']
    db.add_rdf(s, r, o)

    s = 'Q1002'
    r = 'P2'
    o = ['Q1234', 'Q1005', 'Q1004']
    db.add_rdf(s, r, o)

    # db.close()
