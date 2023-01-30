import json
import logging
import re
from pathlib import Path

import pandas as pd
import torch
from unidecode import unidecode
from typing import Protocol
from abc import ABC, abstractmethod

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

from action_executor.actions import ESActionOperator
from helpers import connect_to_elasticsearch, setup_logger
from constants import ROOT_PATH, QA2DModelChoices, RepresentEntityLabelAs
from args import parse_and_get_args
args = parse_and_get_args()

CLIENT = connect_to_elasticsearch()
LOGGER = setup_logger(__name__, loglevel=logging.INFO)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Preprocessor(ABC):
    SEP: str

    @classmethod
    @abstractmethod
    def combine_qa(cls, question: str, answer: str) -> str:
        raise NotImplementedError


class QA2DPreprocessor(Preprocessor):
    SEP = ". "

    @classmethod
    def combine_qa(cls, question: str, answer: str) -> str:
        q = question.replace("?", "").strip()
        a = answer.strip()
        return f'{q}{cls.SEP}{a}'


class B3Preprocessor(Preprocessor):
    SEP = "</s>"

    @classmethod
    def combine_qa(cls, question: str, answer: str) -> str:
        q = question.replace(" ?", "?").strip()
        a = answer.strip()
        return f"{q} {cls.SEP} {a}"


class GhasPreprocessor(Preprocessor):
    SEP = " "

    @classmethod
    def combine_qa(cls, question: str, answer: str) -> str:
        q = question.replace(" ?", "?").strip()
        a = answer.strip()
        return f"q: {q}{cls.SEP}a: {a}"


class QA2DModel:

    def __init__(self, model_type: QA2DModelChoices):
        self.model_type = model_type
        self.tokenizer, self.model = self._get_tokenizer_and_model(model_type)
        self.preprocessor = self._get_preprocessor(model_type)

    @staticmethod
    def _get_tokenizer_and_model(model_type: QA2DModelChoices):
        tokenizer = AutoTokenizer.from_pretrained(model_type.value)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_type.value).to(device)

        return tokenizer, model

    @staticmethod
    def _get_preprocessor(model_type: QA2DModelChoices):
        if model_type in (QA2DModelChoices.T5_SMALL, QA2DModelChoices.T5_BASE):
            return QA2DPreprocessor
        elif model_type == QA2DModelChoices.T5_3B:
            return B3Preprocessor
        elif model_type == QA2DModelChoices.T5_WHYN:
            return GhasPreprocessor
        else:
            raise NotImplementedError(
                f'Chosen model type ({model_type}) is not supported. Refer to QA2DModelChoices class.')

    def preprocess_and_combine(self, question: str, answer: str) -> str:
        return self.preprocessor.combine_qa(question, answer)

    def infer_one(self, qa_string: str) -> str:
        input_ids = self.tokenizer(qa_string, return_tensors="pt").input_ids.to(device)
        LOGGER.debug(f"input_ids in infer_one: ({input_ids.shape}) {input_ids}")

        outputs = self.model.generate(input_ids)
        LOGGER.debug(f"outputs in infer_one: ({outputs.shape}) {outputs}")

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


class GhasQA2DModel(QA2DModel):

    def infer_one(self, qa_string: str, max_length=150) -> str:
        input_ids = self.tokenizer.encode(qa_string, return_tensors="pt", add_special_tokens=True).to(device)
        LOGGER.debug(f"input_ids in infer_one: ({input_ids.shape}) {input_ids}")

        outputs = self.model.generate(input_ids=input_ids, num_beams=2, max_length=max_length, early_stopping=True)
        LOGGER.debug(f"outputs in infer_one: ({outputs.shape}) {outputs}")

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)


def get_model(model_choice: QA2DModelChoices) -> QA2DModel or GhasQA2DModel:
    if model_choice not in QA2DModelChoices:
        raise NotImplementedError()

    if model_choice == QA2DModelChoices.T5_WHYN:
        return GhasQA2DModel(model_choice)
    else:
        return QA2DModel(model_choice)