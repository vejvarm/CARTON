"""
Simple Insert (Direct)
- Simple Insert|Single Entity - TEST
- Simple Insert - TEST
- Simple Insert|Mult. Entity|Indirect - TODO

Simple Insert (Coreferenced)
- Simple Insert|Mult. Entity - TODO
- Simple Insert|Single Entity|Indirect - TODO

Simple Insert (Ellipsis)
- only subject is changed, parent and predicate remains same - TODO
- Incomplete|object parent is changed, subject and predicate remain same - TODO
"""
class SimpleInsert:
    def __init__(self, operator):
        self.operator = operator

    def __call__(self, user, system):
        # Clarification inserts, will be handled from clarification class
        if 'description' not in user:
            return user, system

        # Simple Insert (Direct)
        if user['description'] == 'Simple Insert|Single Entity':
            return self.simple_insert_rdf(user, system)  # NOTE: Test it

        if user['description'] == 'Simple Insert':
            return self.simple_insert_rdf(user, system)  # NOTE: Test it

        if user['description'] == 'Simple Insert|Mult. Entity|Indirect':
            return self.simple_insert_multi_entities(user, system)  # TODO!

        # Simple Insert (Coreferenced)
        if user['description'] == 'Simple Insert|Mult. Entity':
            return self.simple_insert_multi_entities(user, system)  # TODO!

        if user['description'] == 'Simple Insert|Single Entity|Indirect':
            return self.simple_insert_rdf(user, system)  # NOTE: Test it

        # Simple Insert (Ellipsis)
        if user['description'] == 'only subject is changed, parent and predicate remains same':
            return self.simple_insert_ellipsis(user, system)

        if user['description'] == 'Incomplete|object parent is changed, subject and predicate remain same':
            return self.simple_insert_ellipsis(user, system)

        raise Exception(f'Description could not be found: {user["description"]}')

    def simple_insert_rdf(self, user, system):
        # parse input
        data = self.parse_simple_insert_multi_entities(user, system)

        system['is_spurious'] = False

        # extract values
        insert_operator = data['insert_operator']
        entities = data['entity']
        relation = data['relation']

        assert len(entities) == 2  # TODO: (#fix) simple inserts may have much more entities!
        # TODO: possible fix: take action_set into consideration when building gold actions!
        # Decide which entity is subject and which object
        e1, e2 = entities
        if self.operator.find(e1, relation) == e2:
            entities_out = [e1, e2]
        elif self.operator.find(e2, relation) == e1:
            entities_out = [e2, e1]
        else:
            system['is_spurious'] = True
            entities_out = [e1, e2]
            print(f"simple_insert_rdf in action_annotators: This RDF doesn't exist: {e1}:{relation}:{e2}. Setting system['is_spurious'] to True")

        system['gold_actions'] = [
            ['action', insert_operator.__name__],
            ['entity', entities_out[0]],
            ['relation', relation],
            ['entity', entities_out[1]],
        ]

        print(user)

        return user, system

    def simple_insert_multi_entities(self, user, system):
        # parse input
        data = self.parse_simple_insert_multi_entities(user, system)

        # extract values
        logical_operator = data['logical_operator']
        filter_operator = data['filter_operator']
        find_operator = data['find_operator']
        entities = data['entity']
        relation = data['relation']
        typ = data['type']
        gold = data['gold']

        # get results
        find_op_entities = {}
        for find in find_operator:
            filter_entities = []
            for entity in entities:
                ent = find(entity, relation)
                filter_ent = filter_operator(ent, typ)
                if filter_ent and filter_ent.issubset(gold):
                    filter_entities.append(filter_ent)
                    find_op_entities[entity] = find

            if filter_entities:
                result = logical_operator(*filter_entities)

                if gold == result:
                    break

        assert len(find_op_entities) <= len(entities)

        # For multiple entities we might have entities that do not affect the final result
        # We have to include them on the actions with normal find operator
        # We can skip for now
        if len(find_op_entities) < len(entities):
            for ent in entities:
                if ent not in find_op_entities:
                    find_op_entities[ent] = find_operator[0]

        assert len(find_op_entities) == len(entities)

        assert gold == result

        if user['description'] == 'Simple Insert|Mult. Entity':
            system['gold_actions'] = [
                # ['action', logical_operator.__name__],
                ['action', filter_operator.__name__],
                ['action', next(iter(find_op_entities.values())).__name__],
                ['entity', 'prev_answer'],
                ['relation', relation],
                ['type', typ]
            ]
        elif user['description'] == 'Simple Insert|Mult. Entity|Indirect':
            # This type of logical form introduces a big bias for the model.
            # We need to replace this with a more general logical form
            system['gold_actions'] = [
                ['action', logical_operator.__name__],
            ]
            for entity, find in find_op_entities.items():
                system['gold_actions'].extend([
                    ['action', filter_operator.__name__],
                    ['action', find.__name__],
                    ['entity', entity],
                    ['relation', relation],
                    ['type', typ]
                ])
        else:
            raise ValueError(f'Unknown user description: {user}')

        system['is_spurious'] = False if gold == result else True

        return user, system

    def simple_insert_ellipsis(self, user, system):
        # parse input
        data = self.parse_simple_insert_ellipsis(user, system)

        # extract values
        filter_operator = data['filter_operator']
        find_operator = data['find_operator']
        entity = data['entity']
        relation = data['relation']
        typ = data['type']
        gold = data['gold']

        # get results
        for find in find_operator:
            ent = find(entity, relation)
            result = filter_operator(ent, typ)

            if gold == result:
                find_operator = find
                break

        assert gold == result

        system['gold_actions'] = [
            ['action', filter_operator.__name__],
            ['action', find_operator.__name__],
            ['entity', entity],
            ['relation', relation],
            ['type', typ]
        ]

        system['is_spurious'] = False if gold == result else True

        return user, system

    def parse_simple_insert_multi_entities(self, user, system):
        assert len(user['entities_in_utterance']) >= 1
        assert len(user['relations']) == 1
        assert len(user['type_list']) == 1

        return {
            'logical_operator': self.operator.union,
            'filter_operator': self.operator.filter_type,
            'insert_operator': self.operator.insert,
            'entity': user['entities_in_utterance'],
            'relation': user['relations'][0],
            'type': user['type_list'],
            'gold': set(system['all_entities'])
        }

    def parse_simple_insert_ellipsis(self, user, system):
        assert len(system['active_set']) == 1

        active_set = system['active_set'][0][1:-1].split(',')
        if active_set[0].startswith('c'):
            entity = active_set[2]
            relation = active_set[1]
            typ = active_set[0][2:-1]
        elif active_set[0].startswith('Q'):
            entity = active_set[0]
            relation = active_set[1]
            typ = active_set[2][2:-1]
        else:
            raise Exception(f'Wrong active set: {user}')

        return {
            'logical_operator': self.operator.union,
            'filter_operator': self.operator.filter_type,
            'find_operator': [self.operator.find, self.operator.find_reverse],
            'entity': entity,
            'relation': relation,
            'type': typ,
            'gold': set(system['all_entities'])
        }
