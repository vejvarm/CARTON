"""
Simple Insert (Direct)
- Simple Insert|Single Entity - TODO
- Simple Insert - TODO
- Simple Insert|Mult. Entity|Indirect - TODO

Simple Insert (Coreferenced)
- Simple Insert|Mult. Entity - TODO
- Simple Insert|Single Entity|Indirect - TODO

Simple Insert (Ellipsis)
- only subject is changed, parent and predicate remains same - TODO
- Incomplete|object parent is changed, subject and predicate remain same - TODO
"""
from annotate_csqa.ner_annotators.ner_base import NERBase


class SimpleInsert(NERBase):
    def __init__(self, client, preprocessed_data, tokenizer):
        super().__init__(client, preprocessed_data, tokenizer)

    def __call__(self, user, system):
        # Clarification inserts, will be handled from clarification class
        if 'description' not in user:
            return user, system

        # Simple Insert (Direct)
        if user['description'] == 'Simple Insert|Single Entity':
            return self.new_direct_question(user, system)  # NOTE : probably no changes needed

        if user['description'] == 'Simple Insert':
            return self.new_direct_question(user, system)  # NOTE : probably no changes needed

        if user['description'] == 'Simple Insert|Mult. Entity|Indirect':
            return self.new_direct_question(user, system)  # NOTE : probably no changes needed

        if user['description'] == 'Simple Insert|Mult. Entity':
            return self.new_direct_question(user, system)
            # return self.indirect_question(user, system)   # NOTE: no need to change

        # Indirect
        if user['description'] == 'Simple Insert|Single Entity|Indirect':
            return self.indirect_question(user, system)   # NOTE: no need to change

        # Simple Insert (Ellipsis)
        if user['description'] == 'only subject is changed, parent and predicate remains same':
            return self.ellipsis_question(user, system)

        if user['description'] == 'Incomplete|object parent is changed, subject and predicate remain same':
            return self.ellipsis_question(user, system, key_word='which')

        raise Exception(f'Description could not be found: {user["description"]}')
