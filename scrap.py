from time import perf_counter
from knowledge_graph.ZODBConnector import BTreeDB
from knowledge_graph.knowledge_graph import MiniKG

from utils import rapidfuzz_query


def time_querry(fun):

    def helper(*args, **kwargs):
        tic = perf_counter()
        res = fun(*args, **kwargs)
        print(f"t: {perf_counter() - tic} | {res}")

    return helper


if __name__ == '__main__':
    kg = BTreeDB('./knowledge_graph/Wikidata.fs', run_adapter=True)

    kg_memory = MiniKG()

    timed_rp_querry = time_querry(rapidfuzz_query)

    tic = perf_counter()
    timed_rp_querry('Stargate', 'Q1417412', kg_memory)
    timed_rp_querry('Borat', 'Q1417412', kg_memory)
    timed_rp_querry('Boat', 'Q1417412', kg_memory)
    timed_rp_querry('Pretty little liers', 'Q1417412', kg_memory)

    print(f'Memory performance: {perf_counter() - tic}')

    timed_rp_querry('Stargate', 'Q1417412', kg)

    tic = perf_counter()
    timed_rp_querry('Stargate', 'Q1417412', kg)
    timed_rp_querry('Borat', 'Q1417412', kg)
    timed_rp_querry('Boat', 'Q1417412', kg)
    timed_rp_querry('Pretty little liers', 'Q1417412', kg)

    print(f'ZODB performance: {perf_counter() - tic}')

