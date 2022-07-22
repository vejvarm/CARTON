import transaction
from BTrees import OOBTree
from BTrees.OOBTree import TreeSet

import ujson
import ZODB
# DONE: Change - dictionaries in KG into PersistentMappigs
#              - lists in KG into PersistentList


class BTreeDB:

    def __init__(self, path_to_db_file: str):
        self.dbase = None
        self.conn = None
        self.root = None
        self.path_to_db_file = path_to_db_file

        # for adapting to CARTON-like KG implementation
        self.entity_type = {}
        self.labels = {}
        self.triples = {}

        self.open(path_to_db_file)

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
    path_to_db = "./dbfiles/KnowledgeGraph.fs"
    db = BTreeDB(path_to_db)
