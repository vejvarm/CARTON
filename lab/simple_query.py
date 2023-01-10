import logging

from action_executor.actions import ESActionOperator
from helpers import connect_to_elasticsearch, setup_logger

CLIENT = connect_to_elasticsearch()
LOGGER = setup_logger(__name__, loglevel=logging.INFO)

if __name__ == '__main__':
    aop = ESActionOperator(CLIENT)

    e_john_king = 'Q18671928'
    p_father = 'P22'

    # TEST ONE WAY DIRECTIONALITY
    # FIND SUBJECTS by OBJECT+PRED
    subjects = aop.find_reverse([e_john_king], p_father)
    print(subjects)
    # should return:
    # wd:Q18576048	Dorothy Durie
    # wd:Q16853663	Robert King
    # wd:Q5343941	Edward King

    objects = aop.find([e_john_king], p_father)
    print(objects)
    # should return:
    # {}
