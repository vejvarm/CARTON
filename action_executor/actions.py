from typing import Union
import logging

from ordered_set import OrderedSet
from constants import args

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)


class ESActionOperator:
    def __init__(self, es_client, index_ent=args.elastic_index_ent, index_rdf=args.elastic_index_rdf):  # TODO: switch to individual indices for label_and_type and rdf
        self.kg = None
        self.client = es_client
        self.index_ent = index_ent
        self.index_rdf = index_rdf

    def _match(self, field: str, term: str):
        return {'match': {field: term}}

    def _terms(self, field: str, terms: list[str]):
        if isinstance(terms, str):
            LOGGER.warning(f"in self._terms: 'terms' argument should be a list, instead we got {type(terms)}.")
            terms = [terms]

        return {'terms': {field: [t.lower() for t in terms]}}

    def _get_tp_and_label(self, o: str):
        pass

    def _get_by_ids(self, obj_set: OrderedSet['str'], es_index):
        res = self.client.mget(index=es_index,
                               ids=list(obj_set))
        res_dict = dict()

        LOGGER.debug(f"res in self._get_by_ids: {res}")

        for hit in res['docs']:
            _id = hit['_id']
            if hit['found']:
                label = hit['_source']['label']
                type = hit['_source']['type']
                res_dict[_id] = (label, type)
            else:
                LOGGER.warning(f'Entity with id "{_id}" was NOT found in label&type documents.')
                res_dict[_id] = (None, None)

        LOGGER.debug(f"res_dict in self._get_by_ids: {res_dict}")

        return res_dict

    def find(self, e: list[str], p: str):
        if e is None or p is None:
            return None

        if not isinstance(e, list):
            e = [e]

        res = self.client.search(index=self.index_rdf,
                                 query={
                                     'bool': {
                                         'must': [
                                             self._terms('id', e),
                                             self._match('p', p)
                                         ]
                                     }
                                 })

        LOGGER.debug(f"res in self.find: {res}")

        if res['hits']['total']['value'] <= 0:
            return OrderedSet()

        result_set = OrderedSet()
        for hit in res['hits']['hits']:
            result_set.update(hit['_source']['o'])

        return result_set

    def find_reverse(self, e: list[str], p: str):
        if e is None or p is None:
            return None

        if not isinstance(e, list):
            e = [e]

        res = self.client.search(index=self.index_rdf,
                                 query={
                                     'bool': {
                                         'must': [
                                             self._match('p', p),
                                             self._terms('o', e)
                                         ]
                                     }
                                 })

        LOGGER.debug(f"res in self.find_reverse: {res}")

        if res['hits']['total']['value'] <= 0:
            return OrderedSet()

        result_set = OrderedSet()
        for hit in res['hits']['hits']:
            result_set.update([hit['_source']['id']])

        return result_set

    def filter_type(self, ent_set: OrderedSet, typ: str):
        if type(ent_set) is not OrderedSet:
            LOGGER.warning(
                f"in filter_type: ent_set ({ent_set}) was type {type(ent_set)}, this might result in ordering problems.")

        if typ is None:
            LOGGER.info(f"in filter_type: typ is None, returning None")
            return None

        result = OrderedSet()
        lab_tp_dict = self._get_by_ids(ent_set, self.index_ent)

        for o in ent_set:
            if o in lab_tp_dict.keys() and typ == lab_tp_dict[o][-1]:
                result.add(o)

        return result

    def filter_multi_types(self, ent_set: OrderedSet, t1: str, t2: str):
        typ_set = OrderedSet([t1, t2])
        if type(ent_set) is not OrderedSet:
            LOGGER.warning(
                f"in filter_multi_types: ent_set ({ent_set}) was type {type(ent_set)}, this might result in ordering problems.")
            ent_set = OrderedSet(ent_set)

        result = OrderedSet()
        lab_tp_dict = self._get_by_ids(ent_set, self.index_ent)

        for o in ent_set:
            if o in lab_tp_dict.keys() and len(typ_set.intersection(OrderedSet(lab_tp_dict[o][-1:]))) > 0:
                result.add(o)

        return result

    def find_tuple_counts(self, r: str, t1: str, t2: str):
        """
        :param r: (str) property (relation) for which to count connections
        :param t1: (str) only count sro connections for subjects with this type
        :param t2: (str) only count sro connections for objects with this type

        :return type_dict: {'sub_id': int(count of connections of sub(t1) to objects (of t2) by property r)}
        """
        if r is None or t1 is None or t2 is None:
            return None

        res = self.client.search(index=self.index_rdf, query=self._match('p', r))

        tuple_count = dict()
        subjects = OrderedSet()

        for hit in res['hits']['hits']:
            subjects.append(hit['_source']['id'])

        subject_subset = self.filter_type(subjects, t1)

        for sub in subject_subset:
            obj_filtered = self.filter_type(self.find(sub, r), t2)

            tuple_count[sub] = len(obj_filtered)

        return tuple_count

    def find_reverse_tuple_counts(self, r: str, t1: str, t2: str):
        """
        :param r: (str) property (relation) for which to count connections
        :param t1: (str) only count ors connections for objects with this type
        :param t2: (str) only count ors connections for subjects with this type

        :return type_dict: {'obj_id': int(count of connections of obj(t1) to subjects (of t2) by property r)}
        """
        if r is None or t1 is None or t2 is None:
            return None

        res = self.client.search(index=self.index_rdf, query=self._match('p', r))

        tuple_count = dict()
        objects = OrderedSet()

        for hit in res['hits']['hits']:
            objects.update(hit['_source']['o'])

        object_subset = self.filter_type(objects, t1)

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
        self.client.index(index=self.index_rdf, document={'id': sub_id,
                                                      'p': r,
                                                      'o': list(obj_ids)})
        for oid, olab in zip(obj_ids, o_labels):
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
