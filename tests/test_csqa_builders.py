import unittest
import json
import logging
import re
from pathlib import Path

from text_generation.label_replacement import LabelReplacer
from text_generation.qa2d import get_model
from lab.expand_csqa import CSQAInsertBuilder
from action_executor.actions import ESActionOperator
from helpers import connect_to_elasticsearch, setup_logger
from constants import ROOT_PATH, QA2DModelChoices, RepresentEntityLabelAs
from args import parse_and_get_args
args = parse_and_get_args()

CLIENT = connect_to_elasticsearch()
LOGGER = setup_logger(__name__, loglevel=logging.INFO)


class TestCSQAInsertBuilder(unittest.TestCase):

    def setUp(self):
        self.op = ESActionOperator(CLIENT)
        self.transformer = get_model(QA2DModelChoices.T5_WHYN)
        self.labeler = LabelReplacer(self.op)
        self.builder = CSQAInsertBuilder(self.op, self.transformer, self.labeler)

    def test_transform_fields(self):
        user_turn = {
            "ques_type_id": 1,
            "question-type": "Simple Question (Direct)",
            "description": "Simple Question",
            "entities_in_utterance": ["Q961955"],
            "relations": ["P17"],
            "type_list": ["Q15617994"],
            "speaker": "USER",
            "utterance": "Which administrative territory is Saint-Gauzens a part of ?",
            "turn_position": 0
        }

        system_turn = {
            "all_entities": ["Q142"],
            "speaker": "SYSTEM",
            "entities_in_utterance": ["Q142"],
            "utterance": "France",
            "active_set": ["(Q961955,P17,c(Q15617994))"]
        }

        user_new, system_new = self.builder.transform_fields(user_turn, system_turn)

        # Test if the question-type and description are updated correctly
        self.assertEqual(user_new["question-type"], "Simple Input (Direct)")
        self.assertEqual(user_new["description"], "Simple Input")

        # Test if the user's entities_in_utterance field is updated correctly
        self.assertEqual(user_new["entities_in_utterance"], ["Q961955", "Q142"])

        # Test if the system's all_entities field is updated correctly
        self.assertEqual(system_new["all_entities"], ["Q961955", "Q142"])

        # Test if active_set is transformed correctly
        self.assertEqual(system_new["active_set"], ["i(Q961955,P17,Q142)"])

        # Test if the table_format is created correctly
        expected_table_format = [
            [
            "name",
                [
                "Saint-Gauzens"
                ]
             ],
            [
            "country",
                [
                "France"
                ]
            ]
        ]
        self.assertEqual(system_new["table_format"], expected_table_format)

    def test_transform_fields_failure(self):
        user_turn = {
            # Missing required fields or incorrect data types
            "ques_type_id": "1",
            "question-type": 123,
            "description": {},
            "entities_in_utterance": [],
            "relations": [],
            "type_list": [],
            "speaker": "USER",
            "utterance": "",
            "turn_position": 0
        }

        system_turn = {
            # Missing required fields or incorrect data types
            "all_entities": {},
            "speaker": "SYSTEM",
            "entities_in_utterance": {},
            "utterance": "",
            "active_set": []
        }

        with self.assertRaises(Exception):
            user_new, system_new = self.builder.transform_fields(user_turn, system_turn)


if __name__ == '__main__':
    unittest.main()
