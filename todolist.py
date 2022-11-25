# DONE 0: utils.search_by_label: test reimplementation of inverse index (entity label) search to work with new index layout!
#  ・ tested in es_search_and_indexing_tests.py
# DONE 1: should we implement inverse index search in action_executor.actions.py?
# DONE 2: in actions.py: finish implementing insert, insert_reverse, set_labels, set_types
# NOPE 3: in train.py.train() implement ner module functionality and evaluation
# NOPE 4: how to deal with initial incompetence of the NER module while adding entities? Warmup with adding disabled, but reward gained?
# TODO 4: update CSQA dataset with INSERT actions
#  .a) figure out which fields of the original set need to be tweaked
#  .b) tweak action_annotators and ner_annotators for INSERT Question types
#  .c) use QA2D-T5 for transcribing question utterances to statement utterances
# TODO 5:
#  .b just simulate 50/50 chance that entity already exists (50% - search from index_ent 50% - search from index_ent_full)
#   +action for VALIDATE EXISTANCE needed?
# NOPE 6: implement resetting of the index at the beginning of training (NOTE not necessary if we implement the TODO 5.b)
# TODO 8: how to evaluate the final performance?

# TODO in lab/qa2d.py
#       1. replace all entities with their ids
#       2. run QA2D-t5 model on the utterances to get the statements
#       3. embed those to the new dataset

# BONUSES:
# DONE B1: unify ES CLIENT inits to utils.py
# TODO B2: polish kg_migration of index_rdf.json
# TODO B3: analyze ES disk usage (https://www.elastic.co/guide/en/elasticsearch/reference/master/indices-disk-usage.html)