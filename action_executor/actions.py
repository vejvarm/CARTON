from typing import Union
import logging

from ordered_set import OrderedSet
from constants import args

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.WARNING)


class ESActionOperator:
    def __init__(self, es_client,
                 index_ent=args.elastic_index_ent,
                 index_rel=args.elastic_index_rel,
                 index_rdf=args.elastic_index_rdf):  # TODO: switch to individual indices for label_and_type and rdf
        self.kg = None
        self.client = es_client
        self.index_ent = index_ent
        self.index_rel = index_rel
        self.index_rdf = index_rdf

    def _match(self, field: str, term: str):
        return {'match': {field: term}}

    def _terms(self, field: str, terms: list[str]):
        if isinstance(terms, str):
            LOGGER.warning(f"in self._terms: 'terms' argument should be a list, instead we got {type(terms)}.")
            terms = [terms]

        return {'terms': {field: [t.lower() for t in terms]}}

    def _get_tp_and_label(self, oid: str):
        pass

    def _get_by_ids(self, obj_set: OrderedSet['str'], es_index: str):
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
            LOGGER.info(f"in self._get_by_ids: obj_set is {obj_set}, returning empty dictionary")
            return res_dict

        res = self.client.mget(index=es_index,
                               ids=list(obj_set))

        LOGGER.debug(f"res in self._get_by_ids: {res}")

        for hit in res['docs']:
            _id = hit['_id']
            if hit['found']:
                label = hit['_source']['label']
                types = hit['_source']['types']
                res_dict[_id] = (label, types)
            else:
                LOGGER.info(f'in _get_by_ids: Entity with id "{_id}" was NOT found in label&type documents.')
                res_dict[_id] = (None, None)

        LOGGER.debug(f"res_dict in self._get_by_ids: {res_dict}")

        return res_dict

    def find(self, e: list[str], rid: str):
        if e is None or rid is None:
            return None

        if not isinstance(e, list):
            e = [e]

        res = self.client.search(index=self.index_rdf,
                                 query={
                                     'bool': {
                                         'must': [
                                             self._terms('sid', e),
                                             self._match('rid', rid)
                                         ]
                                     }
                                 })

        LOGGER.debug(f"res in self.find: {res}")

        if res['hits']['total']['value'] <= 0:
            return OrderedSet()

        result_set = OrderedSet()
        for hit in res['hits']['hits']:
            result_set.update([hit['_source']['oid']])

        return result_set

    def find_reverse(self, e: list[str], rid: str):
        if e is None or rid is None:
            return None

        if not isinstance(e, list):
            e = [e]

        res = self.client.search(index=self.index_rdf,
                                 query={
                                     'bool': {
                                         'must': [
                                             self._match('rid', rid),
                                             self._terms('oid', e)
                                         ]
                                     }
                                 })

        LOGGER.debug(f"res in self.find_reverse: {res}")

        if res['hits']['total']['value'] <= 0:
            return OrderedSet()

        result_set = OrderedSet()
        for hit in res['hits']['hits']:
            result_set.update([hit['_source']['sid']])

        return result_set

    def filter_type(self, ent_set: OrderedSet, typ: str):
        result = OrderedSet()
        if not ent_set:
            LOGGER.info(f"in filter_type: ent_set is empty, returning empty set")
            return result

        if type(ent_set) is not OrderedSet:
            LOGGER.warning(
                f"in filter_type: ent_set ({ent_set}) was type {type(ent_set)}, this might result in ordering problems.")

        if typ is None:
            LOGGER.info(f"in filter_type: typ is None, returning original set")
            return ent_set

        lab_tp_dict = self._get_by_ids(ent_set, self.index_ent)

        for oid in ent_set:
            if oid not in lab_tp_dict.keys():
                continue

            if lab_tp_dict[oid] == (None, None):
                continue  # TODO: maybe implement list of missing entities?

            if typ == '' and lab_tp_dict[oid][-1] == []:
                LOGGER.debug(f"in filter_type: typ is '', filtering entity with type==[]")
                result.add(oid)
            elif typ in lab_tp_dict[oid][-1]:
                result.add(oid)

        return result

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
            LOGGER.warning(
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
        LOGGER.debug(f'res in fild_tuple_counts: {res}')

        tuple_count = dict()
        subjects = OrderedSet()

        for hit in res['hits']['hits']:
            subjects.append(hit['_source']['sid'])

        LOGGER.debug(f'subjects in fild_tuple_counts: {subjects}')
        subject_subset = self.filter_type(subjects, t1)
        LOGGER.debug(f'subject_subset in fild_tuple_counts: {subject_subset}')

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
        LOGGER.debug(f'res in fild_reverse_tuple_counts: {res}')

        tuple_count = dict()
        objects = OrderedSet()

        for hit in res['hits']['hits']:
            objects.append(hit['_source']['oid'])

        LOGGER.debug(f'objects in fild_tuple_counts: {objects}')
        object_subset = self.filter_type(objects, t1)
        LOGGER.debug(f'object_subset in fild_tuple_counts: {object_subset}')

        for obj in object_subset:
            sub_filtered = self.filter_type(self.find_reverse(obj, r), t2)

            tuple_count[obj] = len(sub_filtered)

        return tuple_count

    # NOTE: type_dict == output from either find_tuple_counts or find_reverse_tuple_counts
    def greater(self, type_dict, value):
        return OrderedSet([k for k, v in type_dict.items() if v > value and v >= 0])

    def less(self, type_dict, value):
        return OrderedSet([k for k, v in type_dict.items() if v < value and v >= 0])

    def equal(self, type_dict, value):
        return OrderedSet([k for k, v in type_dict.items() if v == value and v >= 0])

    def approx(self, type_dict, value, interval=10):
        # ambiguous action
        # simply check for more than 0
        return OrderedSet([k for k, v in type_dict.items() if v > 0])

    def atmost(self, type_dict, max_value):
        return OrderedSet([k for k, v in type_dict.items() if v <= max_value and v >= 0])

    def atleast(self, type_dict, min_value):
        return OrderedSet([k for k, v in type_dict.items() if v >= min_value and v >= 0])

    def argmin(self, type_dict, value=0):
        min_value = min(type_dict.values())
        return OrderedSet([k for k, v in type_dict.items() if v == min_value])

    def argmax(self, type_dict, value=0):
        max_value = max(type_dict.values())
        return OrderedSet([k for k, v in type_dict.items() if v == max_value])

    def is_in(self, ent, set_ent):
        return OrderedSet(ent).issubset(set_ent)

    def count(self, in_set):
        return len(in_set)

    def union(self, *args):
        if all(isinstance(x, OrderedSet) for x in args):
            return args[0].union(*args)
        else:
            return {k: args[0].get(k, 0) + args[1].get(k, 0) for k in OrderedSet(args[0]) | OrderedSet(args[1])}

    def intersection(self, s1, s2):
        return s1.intersection(s2)

    def difference(self, s1, s2):
        return s1.difference(s2)

    def insert(self, s_label: str, r: str, o_labels: OrderedSet[str]):
        """
        :param s_label: from user input based on NER module
        :param r: based on relation Pointer Network
        :param o_labels: from user input based on NER module
        """
        sub_id = None
        obj_ids = []
        # TODO: during training non-existant entities will be "randomly" assigned the gold label!
        # TODO: type should be added from the NER module (don't just use type None, but loss must account for right types!)
        #   for both subjects and objects, because they all sould be present in the user input
        # TODO: during inference, we just need to assign random value (is the system gonna be able to account for this new value?) - it should because all we are searching for is labels of entities from user input and fuzzy linking them to the KG
        # check if s_label exists in KG: FUZZY INVERSE INDEX SEARCH


        # if s_label exists:
        #   check if o_label objects exist
        #   for olab in o_labels:
        #       oid = FUZZY INVERSE INDEX SEARCH
        #       if exists:
        #           TODO: get that object_id and just update the label (and type?)
        #           self.client.update(index=self.index_ent, id=oid, document={'label': olab})
        #       if not exists:
        #           TODO: create new object_id with type None

        # if s_label doesn't exist:
        # TODO: how to set id if it is a new subject?
        # TODO: use update instead of index?
        self.client.index(index=self.index_ent, id=sub_id, document={'label': s_label, 'type': None})

        for oid, olab in zip(obj_ids, o_labels):
            self.client.index(index=self.index_rdf, document={'sid': sub_id,
                                                              'rid': r,
                                                              'oid': list(obj_ids)})
            self.client.index(index=self.index_ent, id=oid, document={'label': olab, 'type': None})

        # what if objects don't exist?

        # TODO: return OrderedSet() of all entities !!!(subject first, folowed by all objects)

    def insert_reverse(self, o, r, s):
        pass

    def set_types(self, entity_set: OrderedSet[str], type_set: OrderedSet[str], override=False):
        """

        :param entity_set: set of entities to check/change types of
        :param type_set: if s exists in KG: keep type from KG (if override=False) | else: from NER module
        :param override: (bool) switch between overriding non None types (True) or not (False)
        """
        # TODO: check only for types which are None
        # TODO: DATASET: how are we gonna reflect this in the dataset gold actions???
        for e, t in zip(entity_set, type_set):
            self.client.update(index=self.index_ent, id=e, document={'type': t})
        # DONE: in order to preserve order, use dictionaries instead of sets

    def update_entity(self, e_set, ein, num):
        pass


class ActionOperator:
    def __init__(self, kg):
        self.kg = kg

    def find(self, e, p):
        if isinstance(e, list):
            return self.find_set(e, p)

        if e is None or p is None:
            return None

        if e not in self.kg.triples['subject'] or p not in self.kg.triples['subject'][e]:
            return set()

        return set(self.kg.triples['subject'][e][p])

    def find_reverse(self, e, p):
        if isinstance(e, list):
            return self.find_reverse_set(e, p)

        if e is None or p is None:
            return None

        if e not in self.kg.triples['object'] or p not in self.kg.triples['object'][e]:
            return set()

        return set(self.kg.triples['object'][e][p])

    def find_set(self, e_set, p):
        result_set = set()
        for e in e_set:
            result_set.update(self.find(e, p))

        return result_set

    def find_reverse_set(self, e_set, p):
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

    def greater(self, type_dict, value):
        return set([k for k, v in type_dict.items() if v > value and v >= 0])

    def less(self, type_dict, value):
        return set([k for k, v in type_dict.items() if v < value and v >= 0])

    def equal(self, type_dict, value):
        return set([k for k, v in type_dict.items() if v == value and v >= 0])

    def approx(self, type_dict, value, interval=10):
        # ambiguous action
        # simply check for more than 0
        return set([k for k, v in type_dict.items() if v > 0])

    def atmost(self, type_dict, max_value):
        return set([k for k, v in type_dict.items() if v <= max_value and v >= 0])

    def atleast(self, type_dict, min_value):
        return set([k for k, v in type_dict.items() if v >= min_value and v >= 0])

    def argmin(self, type_dict, value=0):
        min_value = min(type_dict.values())
        return set([k for k, v in type_dict.items() if v == min_value])

    def argmax(self, type_dict, value=0):
        max_value = max(type_dict.values())
        return set([k for k, v in type_dict.items() if v == max_value])

    def is_in(self, ent, set_ent):
        return set(ent).issubset(set_ent)

    def count(self, in_set):
        return len(in_set)

    def union(self, *args):
        if all(isinstance(x, set) for x in args):
            return args[0].union(*args)
        else:
            return {k: args[0].get(k, 0) + args[1].get(k, 0) for k in set(args[0]) | set(args[1])}

    def intersection(self, s1, s2):
        return s1.intersection(s2)

    def difference(self, s1, s2):
        return s1.difference(s2)
