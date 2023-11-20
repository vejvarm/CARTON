from __future__ import division

import json
import logging
import pathlib

import elasticsearch
from elasticsearch import Elasticsearch
from ordered_set import OrderedSet
from unidecode import unidecode
from random import randint
from wikidata.client import Client as WDClient
from wikidata.entity import EntityId
from urllib.error import HTTPError

from helpers import uppercase

from args import parse_and_get_args
args = parse_and_get_args()

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


class ActionOperator:

    def __init__(self, kg):
        if kg is not None:
            self.kg = kg

    def find(self, e, p):
        if isinstance(e, list):
            return self._find_set(e, p)

        if e is None or p is None:
            return None

        if e not in self.kg.triples['subject'] or p not in self.kg.triples['subject'][e]:
            return set()

        return set(self.kg.triples['subject'][e][p])

    def find_reverse(self, e, p):
        if isinstance(e, list):
            return self._find_reverse_set(e, p)

        if e is None or p is None:
            return None

        if e not in self.kg.triples['object'] or p not in self.kg.triples['object'][e]:
            return set()

        return set(self.kg.triples['object'][e][p])

    def _find_set(self, e_set, p):
        result_set = set()
        for e in e_set:
            result_set.update(self.find(e, p))

        return result_set

    def _find_reverse_set(self, e_set, p):
        result_set = set()
        for e in e_set:
            result_set.update(self.find_reverse(e, p))

        return result_set

    def filter_type(self, ent_set, typ):
        if type(ent_set) is not set or typ is None:
            return None

        result = set()

        for o in ent_set:
            if (o in self.kg.entity_type and typ in self.kg.entity_type[o]):
                result.add(o)

        return result

    def filter_multi_types(self, ent_set, t1, t2):
        typ_set = {t1, t2}
        if type(ent_set) is not set or type(typ_set) is not set:
            return None

        result = set()

        for o in ent_set:
            if (o in self.kg.entity_type and len(typ_set.intersection(set(self.kg.entity_type[o]))) > 0):
                result.add(o)

        return result

    def find_tuple_counts(self, r, t1, t2):
        if r is None or t1 is None or t2 is None:
            return None

        tuple_count = dict()

        for s in self.kg.triples['relation']['subject'][r]:
            if (s in self.kg.entity_type and t1 in self.kg.entity_type[s]):
                count = 0
                for o in self.kg.triples['relation']['subject'][r][s]:
                    if (o in self.kg.entity_type and t2 in self.kg.entity_type[o]):
                        count += 1

                tuple_count[s] = count

        return tuple_count

    def find_reverse_tuple_counts(self, r, t1, t2):
        if r is None or t1 is None or t2 is None:
            return None

        tuple_count = dict()

        for o in self.kg.triples['relation']['object'][r]:
            if (o in self.kg.entity_type and t1 in self.kg.entity_type[o]):
                count = 0
                for s in self.kg.triples['relation']['object'][r][o]:
                    if (s in self.kg.entity_type and t2 in self.kg.entity_type[s]):
                        count += 1

                tuple_count[o] = count

        return tuple_count

    # NOTE: type_dict == output from either find_tuple_counts or find_reverse_tuple_counts
    @staticmethod
    def greater(type_dict, value):
        return OrderedSet([k for k, v in type_dict.items() if v > value and v >= 0])

    @staticmethod
    def less(type_dict, value):
        return OrderedSet([k for k, v in type_dict.items() if v < value and v >= 0])

    @staticmethod
    def equal(type_dict, value):
        return OrderedSet([k for k, v in type_dict.items() if v == value and v >= 0])

    @staticmethod
    def approx(type_dict, value, interval=10):
        # ambiguous action
        # simply check for more than 0
        return OrderedSet([k for k, v in type_dict.items() if v > 0])

    @staticmethod
    def atmost(type_dict, max_value):
        return OrderedSet([k for k, v in type_dict.items() if v <= max_value and v >= 0])

    @staticmethod
    def atleast(type_dict, min_value):
        return OrderedSet([k for k, v in type_dict.items() if v >= min_value and v >= 0])

    @staticmethod
    def argmin(type_dict, value=0):
        min_value = min(type_dict.values())
        return OrderedSet([k for k, v in type_dict.items() if v == min_value])

    @staticmethod
    def argmax(type_dict, value=0):
        max_value = max(type_dict.values())
        return OrderedSet([k for k, v in type_dict.items() if v == max_value])

    @staticmethod
    def is_in(ent, set_ent):
        return OrderedSet(ent).issubset(set_ent)

    @staticmethod
    def count(in_set):
        return len(in_set)

    @staticmethod
    def union(*args):
        if all(isinstance(x, OrderedSet) for x in args):
            return args[0].union(*args)
        else:
            return {k: args[0].get(k, 0) + args[1].get(k, 0) for k in OrderedSet(args[0]) | OrderedSet(args[1])}

    @staticmethod
    def intersection(s1, s2):
        return s1.intersection(s2)

    @staticmethod
    def difference(s1, s2):
        return s1.difference(s2)


class ESActionOperator(ActionOperator):
    def __init__(self, client: Elasticsearch,
                 index_ent=args.elastic_index_ent_full,
                 index_rel=args.elastic_index_rel,
                 index_rdf=args.elastic_index_rdf_full,
                 ent_dict_path: str or pathlib.Path = args.ent_dict_path,
                 rel_dict_path: str or pathlib.Path = args.rel_dict_path,
                 wd_client: WDClient = WDClient(),
                 logger=LOGGER):
        super().__init__(None)
        self.client = client
        self.index_ent = index_ent
        self.index_rel = index_rel
        self.index_rdf = index_rdf
        self.wd_client = wd_client
        self.logger = logger

        self.ent_dict_path = pathlib.Path(ent_dict_path)
        if self.ent_dict_path.exists():
            self.ent_dict = json.load(self.ent_dict_path.open())
        else:
            self.ent_dict = None

        self.rel_dict_path = pathlib.Path(rel_dict_path)
        if self.rel_dict_path.exists():
            self.rel_dict = json.load(self.rel_dict_path.open())
        else:
            self.rel_dict = None

    @staticmethod
    def _match(field: str, term: str):
        return {'match': {field: term}}

    def _terms(self, field: str, terms: list[str]):
        if isinstance(terms, str):
            self.logger.warning(f"in self._terms: 'terms' argument should be a list, instead we got {type(terms)}.")
            terms = [terms]

        return {'terms': {field: [t.lower() for t in terms]}}

    @uppercase
    def _get_by_ids(self, obj_set: OrderedSet[str], es_index: str):
        """

        # special cases:
        #   entity doesn't exist: {..., (None, None), ...}
        #   types list for entity is empty: {..., ('label', []), ...}

        :param obj_set: (OrderedSet) objects to get from entity index
        :param es_index: (str) index on which to run this operation
        :return:
        """
        res_dict = dict()
        if not obj_set or '' in obj_set:
            self.logger.info(f"in self._get_by_ids: obj_set is {obj_set}, returning empty dictionary")
            return res_dict

        res = self.client.mget(index=es_index,
                               ids=list(obj_set))

        self.logger.debug(f"res in self._get_by_ids: {res}")

        for hit in res['docs']:
            _id = hit['_id']
            if hit['found']:
                label = None
                types = None
                if 'label' in hit['_source'].keys():
                    label = hit['_source']['label']
                if 'types' in hit['_source'].keys():
                    types = hit['_source']['types']
                res_dict[_id] = (label, types)
            else:
                self.logger.info(f'in _get_by_ids: Entity with id "{_id}" was NOT found in label&type documents.')
                res_dict[_id] = (None, None)

        self.logger.debug(f"res_dict in self._get_by_ids: {res_dict}")

        return res_dict

    @uppercase
    def get_rdf(self, _id: str) -> dict:
        """Get RDF with given _id, which follows {sid}{rid}{oid} structure, if it exists"""
        index = self.index_rdf
        if not self.client.exists(index=index, id=_id):
            self.logger.info(f"get_rdf in ESActionOperator: rdf with _id {_id} doesn't exist in {index}.")
            return {}

        return self.client.get(index=index, id=_id)['_source']

    def update_ent_json(self):
        f"""
        Should call this if you want to OVERWRITE the {self.ent_dict_path} with the current state of self.ent_dict
        """
        self.logger.info(f"updating ent_dict file at `{self.ent_dict_path}`")
        json.dump(self.ent_dict, self.ent_dict_path.open("w", encoding='utf8'), indent=4)

    def update_rel_json(self):
        """
        Should call this if you want to OVERWRITE the self.rel_dict_path with the current state of self.rel_dict
        """
        self.logger.info(f"updating rel_dict file at `{self.rel_dict_path}`")
        json.dump(self.rel_dict, self.rel_dict_path.open("w", encoding='utf8'), indent=4)

    @uppercase
    def _get_english_label_from_wikidata(self, gid: str or EntityId, wd_client: WDClient, logger: logging.Logger) -> str:
        if not str(gid).startswith(("P", "Q")):
            raise NameError("id must start with 'P' (for predicate/relation) or 'Q' (entity)")

        try:
            # try fetching the entity from WD database
            entity = wd_client.get(EntityId(gid), load=True)
        except HTTPError as err:
            logger.warning(f"{err} (returning original id: {gid})")
            return gid

        # get the entity/predicate label in english
        label = entity.label['en']

        # update the dictionaries
        if str(gid).startswith("P"):
            self.rel_dict.update({gid: label})
        if str(gid).startswith("Q"):
            self.ent_dict.update({gid: label})

        return label

    @uppercase
    def _get_label(self, gid: str, id2label_dict: dict, index: str):
        """Generic get label for given gid in given index if it exists"""
        if id2label_dict is not None:
            try:
                return id2label_dict[gid]
            except KeyError:
                print("d", end="")
                pass

        if self.client is not None and self.client.exists(index=index, id=gid):
            return self.client.get(index=index, id=gid)['_source']['label']

        print("e", end="")
        return self._get_english_label_from_wikidata(gid, self.wd_client, self.logger)

    # @uppercase
    # def get_entity_label(self, eid: str):
    #     """Get entity label for given entity (eid) if it exists"""
    #     index = self.index_ent
    #     return self._get_label(eid, index)

    @uppercase
    def get_entity_label(self, eid: str):
        """Get entity label for given entity (eid) if it exists"""
        if not str(eid).startswith("Q"):
            raise NameError("id of entity must start with 'Q'")
        index = self.index_ent
        ent_dict = self.ent_dict
        return self._get_label(eid, ent_dict, index)

    @uppercase
    def get_relation_label(self, rid: str):
        """Get relation label for given relation (rid) if it exists"""
        if not str(rid).startswith("P"):
            raise NameError("id of predicate (relation) must start with 'P'")
        index = self.index_rel
        rel_dict = self.rel_dict
        return self._get_label(rid, rel_dict, index)

    @uppercase
    def get_types(self, eid: str):
        """Get list of types for given entity (eid) if it exists."""
        index = self.index_ent
        if not self.client.exists(index=index, id=eid):
            self.logger.info(f"get_types in ESActionOperator: entity with {eid} doesn't exist in {index}.")
            return []

        return self.client.get(index=index, id=eid)['_source']['types']

    @uppercase
    def find(self, e: OrderedSet[str] or list[str] or str, rid: str):
        if e is None or rid is None:
            return None

        if isinstance(e, str):
            e = [e]

        if isinstance(e, OrderedSet):
            e = list(e)

        res = self.client.search(index=self.index_rdf,
                                 query={
                                     'bool': {
                                         'must': [
                                             self._terms('sid', e),
                                             self._match('rid', rid)
                                         ]
                                     }
                                 })

        self.logger.debug(f"res in self.find: {res}")

        if res['hits']['total']['value'] <= 0:
            return OrderedSet()

        result_set = OrderedSet()
        for hit in res['hits']['hits']:
            result_set.update([hit['_source']['oid']])

        return result_set

    @uppercase
    def find_reverse(self, e: OrderedSet[str] or list[str] or str, rid: str):
        if e is None or rid is None:
            return None

        if isinstance(e, str):
            e = [e]

        if isinstance(e, OrderedSet):
            e = list(e)

        res = self.client.search(index=self.index_rdf,
                                 query={
                                     'bool': {
                                         'must': [
                                             self._match('rid', rid),
                                             self._terms('oid', e)
                                         ]
                                     }
                                 })

        self.logger.debug(f"res in self.find_reverse: {res}")

        if res['hits']['total']['value'] <= 0:
            return OrderedSet()

        result_set = OrderedSet()
        for hit in res['hits']['hits']:
            result_set.update([hit['_source']['sid']])

        return result_set

    @uppercase
    def filter_type(self, ent_set: OrderedSet, typ: str):
        result = OrderedSet()
        if not ent_set:
            self.logger.info(f"in filter_type: ent_set is empty, returning empty set")
            return result

        if type(ent_set) is not OrderedSet:
            self.logger.warning(
                f"in filter_type: ent_set ({ent_set}) was type {type(ent_set)}, this might result in ordering problems.")

        if typ is None:
            self.logger.info(f"in filter_type: typ is None, returning original set")
            return ent_set

        lab_tp_dict = self._get_by_ids(ent_set, self.index_ent)

        for oid in ent_set:
            if oid not in lab_tp_dict.keys():
                continue

            if lab_tp_dict[oid] == (None, None):
                continue  # TODO: maybe implement list of missing entities?

            if typ == '' and lab_tp_dict[oid][-1] == []:
                self.logger.debug(f"in filter_type: typ is '', filtering entity with type==[]")
                result.add(oid)
            elif typ in lab_tp_dict[oid][-1]:
                result.add(oid)

        return result

    @uppercase
    def filter_multi_types(self, ent_set: OrderedSet, t1: str, t2: str):
        """ filter set of entities by two types (t1, t2)

        object is accepted if it has at least one type in common with (t1, t2) ( == OR operation)

        :param ent_set: (OrderedSet) set of entities to filter
        :param t1: (str) type one by which to filter entities in ent_set
        :param t2: (str) type two ...
        :return: (OrderedSet) of entities with at least one of the types (t1 or t2)
        """
        typ_set = OrderedSet([t1, t2])
        if type(ent_set) is not OrderedSet:
            self.logger.warning(
                f"in filter_multi_types: ent_set ({ent_set}) was type {type(ent_set)}, this might result in ordering problems.")
            ent_set = OrderedSet(ent_set)

        result = OrderedSet()
        lab_tp_dict = self._get_by_ids(ent_set, self.index_ent)

        for oid in ent_set:
            if oid not in lab_tp_dict.keys():
                continue

            if lab_tp_dict[oid] == (None, None):
                continue  # TODO: maybe implement list of missing entities?

            if oid in lab_tp_dict.keys() and len(typ_set.intersection(OrderedSet(lab_tp_dict[oid][-1]))) > 0:
                result.add(oid)

        return result

    def find_tuple_counts(self, r: str, t1: str = None, t2: str = None):
        """
        :param r: (str) property (relation) for which to count connections
        :param t1: (str|None) only count sro connections for subjects with this type (if None, don't filter)
        :param t2: (str|None) only count sro connections for objects with this type (if None, don't filter)

        :return type_dict: {'sub_id': int(count of connections of sub(t1) to objects (of t2) by property r)}
        """
        if r is None:
            return None

        res = self.client.search(index=self.index_rdf, query=self._match('rid', r), size=args.max_results)
        self.logger.debug(f'res in fild_tuple_counts: {res}')

        tuple_count = dict()
        subjects = OrderedSet()

        for hit in res['hits']['hits']:
            subjects.append(hit['_source']['sid'])

        self.logger.debug(f'subjects in fild_tuple_counts: {subjects}')
        subject_subset = self.filter_type(subjects, t1)
        self.logger.debug(f'subject_subset in fild_tuple_counts: {subject_subset}')

        for sub in subject_subset:
            obj_filtered = self.filter_type(self.find(sub, r), t2)

            tuple_count[sub] = len(obj_filtered)

        return tuple_count

    def find_reverse_tuple_counts(self, r: str, t1: str = None, t2: str = None):
        """
        :param r: (str) property (relation) for which to count connections
        :param t1: (str|None) only count ors connections for objects with this type (if None, don't filter)
        :param t2: (str|None) only count ors connections for subjects with this type (if None, don't filter)

        :return type_dict: {'obj_id': int(count of connections of obj(t1) to subjects (of t2) by property r)}
        """
        if r is None:
            return None

        res = self.client.search(index=self.index_rdf, query=self._match('rid', r), size=args.max_results)
        self.logger.debug(f'res in fild_reverse_tuple_counts: {res}')

        tuple_count = dict()
        objects = OrderedSet()

        for hit in res['hits']['hits']:
            objects.append(hit['_source']['oid'])

        self.logger.debug(f'objects in fild_tuple_counts: {objects}')
        object_subset = self.filter_type(objects, t1)
        self.logger.debug(f'object_subset in fild_tuple_counts: {object_subset}')

        for obj in object_subset:
            sub_filtered = self.filter_type(self.find_reverse(obj, r), t2)

            tuple_count[obj] = len(sub_filtered)

        return tuple_count

    def insert(self, sid: str, rid: str, oid: str):
        """ Add a single entry into the knowledge graph

        If exists: do nothing

        :param sid: (str) id of subject
        :param rid: (str) id of relation
        :param oid: (str) id of object (NOTE: can be '')
        :return: OrderedSet[sid, oid]
        """
        _id = f'{sid}{rid}{oid}'.upper()
        if self.client.exists(index=self.index_rdf, id=_id):
            self.logger.info(f'insert in actions: entry with id {_id} already exists in {self.index_rdf}')
            return None

        if not self.client.exists(index=self.index_ent, id=sid):
            self.logger.warning(f"insert in actions: entry with id {sid} doesn't exists in {self.index_ent}! (Problem in NER module?)")
            # NOTE: This shouldn't be possile to happen, as we create new entities in NER module

        self.client.index(index=self.index_rdf, id=_id, document={'sid': sid, 'rid': rid, 'oid': oid})
        self.logger.info(f'insert in actions: entry with id {_id} was added to {self.index_rdf}')

        return OrderedSet([sid, oid])

    def insert_reverse(self, oid, rid, sid):
        # NOTE we don't need this, the system should be able to learn the difference between oid and sid and
        #   just use the insert function
        pass

    @uppercase
    def update_labels(self, ent_set: OrderedSet[str], label_list: list[str], overwrite=False):
        """

        :param ent_set: (OrderedSet[str]) ids of entities which should be affected
        :param label_list: (list[str]) new labels for each entity (same order as ent_set)
        :param overwrite: (bool) switch between overriding non empty labels (True) or not (False

        :return op_results: (list[str]) 'noop' if no change | 'updated' if changed
        """
        op_results = []
        for eid, labl in zip(ent_set, label_list):
            try:
                cur_label = self.client.get(index=self.index_ent, id=eid)['_source']['label']
                self.logger.debug(f'cur_label in update_labels: {cur_label}')
                if not cur_label or overwrite:
                    res = self.client.update(index=self.index_ent, id=eid, doc={'label': labl})
                    op_results.append(res['result'])
                else:
                    self.logger.info(f'in update_labels: entity {eid} already has label {cur_label} and overwrite==False. Skipping')
            except elasticsearch.NotFoundError:
                self.logger.info(f'set_labels in actions: entity with id {eid} not found in {self.index_ent}. Skipping.')
                op_results.append('noop')

        return op_results

    @uppercase
    def update_types(self, ent_set: OrderedSet[str], types_list: list[list[str]], overwrite=False):
        """

        :param ent_set: (OrderedSet[str]) set of entities to check/change types of
        :param types_list: (list[list[str]]) new type lists for each entity (same order as ent_set)
        :param overwrite: (bool) switch between overwriting non empty type lists (True) or not (False)

        :return op_results: (list[str]) 'noop' if no change | 'updated' if changed
        """
        op_results = []
        for eid, tp_list in zip(ent_set, types_list):
            if isinstance(tp_list, str):
                tp_list = [tp_list]
            try:
                if overwrite:  # overwrite existing list of types with new ones
                    res = self.client.update(index=self.index_ent, id=eid, doc={'types': tp_list})
                    op_results.append(res['result'])
                else:  # append to existing field
                    tp_set = set(self.client.get(index=self.index_ent, id=eid)['_source']['types'])
                    tp_set.update(tp_list)
                    res = self.client.update(index=self.index_ent, id=eid, doc={'types': list(tp_set)})
                    op_results.append(res['result'])
                    # TODO: Alternatively use append pipeline processor?
            except elasticsearch.NotFoundError:
                self.logger.info(f'update_types in actions: entity with id {eid} not found in {self.index_ent}. Skipping.')
                op_results.append('noop')

        return op_results

    def delete_rdf(self, sid: str, rid: str, oid: str):
        """ Remove single RDF entry from the KG

        If not exists: do nothing

        :param sid: (str) id of subject
        :param rid: (str) id of relation
        :param oid: (str) id of object (NOTE: can be '')
        :return: OrderedSet[sid, oid]
        """
        _id = f'{sid}{rid}{oid}'.upper()
        if not self.client.exists(index=self.index_rdf, id=_id):
            self.logger.info(f"delete_rdf in actions: entry with id {_id} doesn't exist in {self.index_rdf}. No action")
            return None

        self.client.delete(index=self.index_rdf, id=_id)
        self.logger.info(f'delete_rdf in actions: entry with id {_id} was removed from {self.index_rdf}')

        return OrderedSet([sid, oid])


def search_by_label(client: Elasticsearch, query: str, filter_type: str, res_size=1, index=args.elastic_index_ent_full):
    """ ElasticSearch implementation of inverse index Fuzzy search. Essentially searching for a document with specific label
    utilizing a bit of fuzziness to account for misspellings and typos.

    :param client: elasticsearch client
    :param str query: label to search for
    :param str filter_type: type_id which restricts the search to only entities of this type
    :param int res_size: maximum number of results
    :param str index: client index in which to search
    """
    res = client.search(index=index, size=res_size, query={'match': {'label': {'query': unidecode(query), 'fuzziness': 'AUTO'}}})
    results = []
    for hit in res['hits']['hits']: results.append([hit['_id'], hit['_source']['types']])
    filtered_results = [res for res in results if filter_type in res[1]]
    return [res[0] for res in filtered_results] if filtered_results else [res[0] for res in results]


def _gen_unique_id(client: Elasticsearch, eid_range: tuple[int, int]):
    # generate new id randomly until we generate unique id
    while True:
        eid = f'Q{randint(eid_range[0], eid_range[1])}'

        if not client.exists(index=args.elastic_index_ent_full, id=eid):
            break

    return eid


def create_entity(client: Elasticsearch, eid: str=None, label: str=None, types: list[str] = tuple(), production=False,
                  eid_range=(1000000, 9999999), logger=LOGGER):
    """ create new entity in args.elastic_index_ent_full

    :param client: Elasticsearch client object
    :param eid: (str) entity id, if eid is None, generate random unique eid
    :param label: (str) label of the newly created entity
    :param types: (list[str]) list of types of the newly create entity
    :param production: (bool) if production==True, eids are generated by random generator, else it is taken from dataset
    :param eid_range: (tuple[int]) minimum and maximum value for random eid generator
    :param logger: logging.Logger from parent
    :return eid: (str) newly generated entity id OR passthrough of eid from input (if eid is not None)
    """

    if label is None:
        label = ''

    if not types:
        types = []

    if eid is None:
        if production:
            # generate new id randomly until we generate unique id
            eid = _gen_unique_id(client, eid_range)
        else:
            # 'generate' new id corresponding to index_ent_full
            # NOTE: for traning purposes, this randomness is artificially made to reflect training set entity ids
            try:
                eid = search_by_label(client, label, '', res_size=1, index=args.elastic_index_ent_full)[0]
            except IndexError:
                logger.info(f"`{label}` not in KG, generating random eid: {eid}")
                eid = _gen_unique_id(client, eid_range)

    client.index(index=args.elastic_index_ent_full, id=eid, document={'label': label, 'types': types})
    logger.info(f'create_entity in actions: added new entry to {args.elastic_index_ent_full} with id: {eid} label: {label}, types: {types}')

    return eid
