def debug_get_entity_label(eid: str):
    from action_executor.actions import ESActionOperator
    from helpers import connect_to_elasticsearch
    from action_executor.actions import LOGGER
    try:
        client = connect_to_elasticsearch()
        eso = ESActionOperator(client)
        print(f"{eso.get_entity_label(eid)} [eso.get_entity_label(eid)]")
        print(f"{eso._get_english_label_from_wikidata(eid, eso.wd_client, LOGGER)} [eso._get_english_label_from_wikidata]")
        return 1
    except Exception as err:
        print(repr(err))
        return 0
