from __future__ import division
import re
import time
import json
import logging
import torch.nn as nn

import helpers
from action_executor.actions import search_by_label, create_entity
from collections import OrderedDict
from transformers import BertTokenizer
from elasticsearch import Elasticsearch

from rapidfuzz import process
from rapidfuzz.distance.Levenshtein import distance

# import constants
from constants import *

# import CSQA ZODB KG

# set LOGGER
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, optimizer, model_size=args.emb_dim, factor=args.factor, warmup=args.warmup):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()

# meter class for storing results
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Predictor(object):
    """Predictor class"""
    def __init__(self, model, vocabs):
        self.model = model
        self.vocabs = vocabs

    def predict(self, input):
        """Perform prediction on given input example"""
        self.model.eval()
        model_out = {}

        # prepare input
        tokenized_sentence = [START_TOKEN] + [t.lower() for t in input] + [CTX_TOKEN]
        numericalized = [self.vocabs[INPUT].stoi[token] if token in self.vocabs[INPUT].stoi else self.vocabs[INPUT].stoi[UNK_TOKEN] for token in tokenized_sentence]
        src_tensor = torch.LongTensor(numericalized).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            # get ner, coref predictions
            encoder_step = self.model._predict_encoder(src_tensor)
            # TODO: encoder_step contains [encoder_out, ner_out, coref_out]
            encoder_out = encoder_step[ENCODER_OUT]  # FIXME compatibility with CARTON
            encoder_ctx = encoder_out[:, -1, :]  # TODO: check this
            ner_out = encoder_step[NER].argmax(1).tolist()
            coref_out = encoder_step[COREF].argmax(1).tolist()

            # get logical form, predicate and type prediction
            lf_out = [self.vocabs[LOGICAL_FORM].stoi[START_TOKEN]]
            pd_out = [self.vocabs[PREDICATE_POINTER].stoi[NA_TOKEN]]
            tp_out = [self.vocabs[TYPE_POINTER].stoi[NA_TOKEN]]

            for _ in range(self.model.decoder.max_positions):
                lf_tensor = torch.LongTensor(lf_out).unsqueeze(0).to(DEVICE)

                decoder_step = self.model._predict_decoder(src_tensor, lf_tensor, encoder_out)
                decoder_out = decoder_step[DECODER_OUT]
                decoder_h = decoder_step[DECODER_H]
                stacked_pointer_out = self.model.stptr_net(encoder_ctx, decoder_h)  # [bs*v, n_kg]

                # TODO: what is the shape of this?, How do we infer the KG entries from this?
                pred_lf = decoder_out.argmax(1)[-1].item()
                pred_pd = stacked_pointer_out[PREDICATE_POINTER].argmax(1)[-1].item()  # argmax(1) [bs*v, n_kg] -> [bs*v], [-1] [bs*v] -> last entry
                pred_tp = stacked_pointer_out[TYPE_POINTER].argmax(1)[-1].item()

                if pred_lf == self.vocabs[LOGICAL_FORM].stoi[END_TOKEN]:
                    break

                lf_out.append(pred_lf)
                pd_out.append(pred_pd)
                tp_out.append(pred_tp)

        # translate top predictions into vocab tokens
        model_out[LOGICAL_FORM] = [self.vocabs[LOGICAL_FORM].itos[i] for i in lf_out][1:]
        model_out[NER] = [self.vocabs[NER].itos[i] for i in ner_out][1:-1]
        model_out[COREF] = [self.vocabs[COREF].itos[i] for i in coref_out][1:-1]
        model_out[PREDICATE_POINTER] = [self.vocabs[PREDICATE_POINTER].itos[i] for i in pd_out][1:]
        model_out[TYPE_POINTER] = [self.vocabs[TYPE_POINTER].itos[i] for i in tp_out][1:]

        return model_out

class AccuracyMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.correct = 0
        self.wrong = 0
        self.accuracy = 0

    def update(self, gold, result):
        if gold == result:
            self.correct += 1
        else:
            self.wrong += 1

        self.accuracy = self.correct / (self.correct + self.wrong)

class Scorer(object):
    """Scorer class"""
    def __init__(self):
        self.tasks = [TOTAL, LOGICAL_FORM, NER, COREF, PREDICATE_POINTER, TYPE_POINTER]
        self.results = {
            OVERALL: {task:AccuracyMeter() for task in self.tasks},
            CLARIFICATION: {task:AccuracyMeter() for task in self.tasks},
            COMPARATIVE: {task:AccuracyMeter() for task in self.tasks},
            LOGICAL: {task:AccuracyMeter() for task in self.tasks},
            QUANTITATIVE: {task:AccuracyMeter() for task in self.tasks},
            SIMPLE_COREFERENCED: {task:AccuracyMeter() for task in self.tasks},
            SIMPLE_DIRECT: {task:AccuracyMeter() for task in self.tasks},
            SIMPLE_ELLIPSIS: {task:AccuracyMeter() for task in self.tasks},
            # -------------------------------------------
            VERIFICATION: {task:AccuracyMeter() for task in self.tasks},
            QUANTITATIVE_COUNT: {task:AccuracyMeter() for task in self.tasks},
            COMPARATIVE_COUNT: {task:AccuracyMeter() for task in self.tasks},
        }
        self.data_dict = []

    def data_score(self, data, helper, predictor):
        """Score complete list of data"""
        for i, (example, q_type)  in enumerate(zip(data, helper['question_type'])):
            # prepare references
            ref_lf = [t.lower() for t in example.logical_form]
            ref_ner = example.ner
            ref_coref = example.coref
            ref_pd = example.predicate_pointer
            ref_tp = example.type_pointer
            ref_en = helper[ENTITY][LABEL][example.id[0]]

            # get model hypothesis
            hypothesis = predictor.predict(example.input)

            # check correctness
            correct_lf = 1 if ref_lf == hypothesis[LOGICAL_FORM] else 0
            correct_ner = 1 if ref_ner == hypothesis[NER] else 0
            correct_coref = 1 if ref_coref == hypothesis[COREF] else 0
            correct_pd = 1 if ref_pd == hypothesis[PREDICATE_POINTER] else 0
            correct_tp = 1 if ref_tp == hypothesis[TYPE_POINTER] else 0

            # save results
            gold = 1
            res = 1 if correct_lf and correct_ner and correct_coref and correct_pd and correct_tp else 0
            # Question type
            self.results[q_type][TOTAL].update(gold, res)
            self.results[q_type][LOGICAL_FORM].update(ref_lf, hypothesis[LOGICAL_FORM])
            self.results[q_type][NER].update(ref_ner, hypothesis[NER])
            self.results[q_type][COREF].update(ref_coref, hypothesis[COREF])
            self.results[q_type][PREDICATE_POINTER].update(ref_pd, hypothesis[PREDICATE_POINTER])
            self.results[q_type][TYPE_POINTER].update(ref_tp, hypothesis[TYPE_POINTER])
            # Overall
            self.results[OVERALL][TOTAL].update(gold, res)
            self.results[OVERALL][LOGICAL_FORM].update(ref_lf, hypothesis[LOGICAL_FORM])
            self.results[OVERALL][NER].update(ref_ner, hypothesis[NER])
            self.results[OVERALL][COREF].update(ref_coref, hypothesis[COREF])
            self.results[OVERALL][PREDICATE_POINTER].update(ref_pd, hypothesis[PREDICATE_POINTER])
            self.results[OVERALL][TYPE_POINTER].update(ref_tp, hypothesis[TYPE_POINTER])

            # save data
            self.data_dict.append({
                INPUT: example.input,
                LOGICAL_FORM: hypothesis[LOGICAL_FORM],
                f'{LOGICAL_FORM}_gold': ref_lf,
                NER: hypothesis[NER],
                f'{NER}_gold': ref_ner,
                COREF: hypothesis[COREF],
                f'{COREF}_gold': ref_coref,
                PREDICATE_POINTER: hypothesis[PREDICATE_POINTER],
                f'{PREDICATE_POINTER}_gold': ref_pd,
                TYPE_POINTER: hypothesis[TYPE_POINTER],
                f'{TYPE_POINTER}_gold': ref_tp,
                # ------------------------------------
                f'{LOGICAL_FORM}_correct': correct_lf,
                f'{NER}_correct': correct_ner,
                f'{COREF}_correct': correct_coref,
                f'{PREDICATE_POINTER}_correct': correct_pd,
                f'{TYPE_POINTER}_correct': correct_tp,
                IS_CORRECT: res,
                QUESTION_TYPE: q_type
            })

            if (i+1) % 500 == 0:
                LOGGER.info(f'* {OVERALL} Data Results {i+1}:')
                for task, task_result in self.results[OVERALL].items():
                    LOGGER.info(f'\t\t{task}: {task_result.accuracy:.4f}')

    def write_results(self):
        save_dict = json.dumps(self.data_dict, indent=4)
        save_dict_no_space_1 = re.sub(r'": \[\s+', '": [', save_dict)
        save_dict_no_space_2 = re.sub(r'",\s+', '", ', save_dict_no_space_1)
        save_dict_no_space_3 = re.sub(r'"\s+\]', '"]', save_dict_no_space_2)
        with open(f'{ROOT_PATH}/{args.path_error_analysis}/error_analysis.json', 'w', encoding='utf-8') as json_file:
            json_file.write(save_dict_no_space_3)

    def reset(self):
        """Reset object properties"""
        self.results = []
        self.instances = 0


class Inference(object):
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained(BERT_BASE_UNCASED)
        self.inference_actions = []
        self.es = Elasticsearch(args.elastic_host, ca_certs=args.elastic_certs,
                                basic_auth=(args.elastic_user, args.elastic_password['freya']),
                                retry_on_timeout=True)  # for inverse index search
        # self.kg = BTreeDB(args.kg_path, run_adapter=True)  # ANCHOR: ZODB implementation

    def construct_actions(self, inference_data, predictor):
        LOGGER.info(f'Constructing actions for: {args.question_type}')
        self.inference_actions = []  # clear inference actions from previous run
        tic = time.perf_counter()
        # based on model outpus create a final logical form to execute
        question_type_inference_data = [data for data in inference_data if args.question_type in data[QUESTION_TYPE]]
        for i, sample in enumerate(question_type_inference_data):  # ANCHOR: u wot m8? ... how is having CONTEXT_ENTITIES not cheating?
            predictions = predictor.predict(sample[CONTEXT_QUESTION])  # NOTE: detokenized predictions!
            actions = []
            logical_form_prediction = predictions[LOGICAL_FORM]
            ent_count_pos = 0  # counts how many ENTITY actions we encountered in the LF so far
            for j, action in enumerate(logical_form_prediction):
                # TODO: if action == 'insert', change behaviour of ENTITY fills
                #   in that case
                if action not in [ENTITY, RELATION, TYPE, VALUE, PREV_ANSWER]:
                    actions.append([ACTION, action])
                elif action == ENTITY:  # ANCHOR: this is where we deal with filling the right entities to LF 'ENTITY' action
                    # get predictions
                    context_question = sample[CONTEXT_QUESTION]
                    ner_prediction = predictions[NER]
                    coref_prediction = predictions[COREF]
                    # get their indices
                    ner_indices = OrderedDict({k: tag.split('-')[-1] for k, tag in enumerate(ner_prediction) if
                                               tag.startswith(B) or tag.startswith(I)})  # idx: type_id
                    coref_indices = OrderedDict({k: tag for k, tag in enumerate(coref_prediction) if tag not in ['NA']})
                    # create a ner dictionary with index as key and entity as value
                    ner_idx_ent = self.create_ner_idx_ent_dict(ner_indices, context_question)  # {int: list[str]} ... {1: ['Q1', 'Q2'], 2: ['Q1', 'Q2'], 4: ['UNK'], 5: ['UNK']}
                    if str(ent_count_pos) not in list(coref_indices.values()):
                        if args.question_type in [CLARIFICATION, QUANTITATIVE_COUNT] and len(
                                list(coref_indices.values())) == ent_count_pos:  # simple constraint for clarification and quantitative count
                            for l, (cidx, ctag) in enumerate(coref_indices.items()):  # cidx = position in input ... ctag = desired position in LF
                                if ctag == str(ent_count_pos - 1):
                                    if cidx in ner_idx_ent:
                                        actions.append([ENTITY, ner_idx_ent[cidx][0]])  # NOTE: this is where we permute! BEWARE: we only take the first entity from list?!
                                        break
                                    else:
                                        print(f'Coref index {cidx} not in ner entities!')
                                        actions.append([ENTITY, ENTITY])
                                        break
                            try:
                                actions.append([ENTITY, ner_idx_ent.popitem()[1][0]])
                            except:
                                print('No coref indices!')
                                actions.append([ENTITY, ENTITY])
                        elif args.question_type in [VERIFICATION, SIMPLE_DIRECT,
                                                    CLARIFICATION] and ent_count_pos == 0 and not coref_indices:  # simple constraint for verification and simple question (direct)
                            try:
                                actions.append([ENTITY, ner_idx_ent.popitem()[1][0]])
                            except:
                                print('No coref indices!')
                                actions.append([ENTITY, ENTITY])
                        else:
                            # TODO here things get hard, we will need to use all ner entites and see if it works
                            print('No coref indices!')
                            actions.append([ENTITY, ENTITY])
                    else:
                        for l, (cidx, ctag) in enumerate(coref_indices.items()):
                            if ctag == str(ent_count_pos):
                                if cidx in ner_idx_ent:
                                    actions.append([ENTITY, ner_idx_ent[cidx][0]])
                                    break
                                else:
                                    print(f'Coref index {cidx} not in ner entities!')
                                    actions.append([ENTITY, ENTITY])
                                    break
                    # update entity position counter
                    ent_count_pos += 1
                elif action == RELATION:
                    predicate_prediction = predictions[PREDICATE_POINTER]
                    actions.append([RELATION, predicate_prediction[j]])
                elif action == TYPE:
                    type_prediction = predictions[TYPE_POINTER]
                    actions.append([TYPE, type_prediction[j]])
                elif action == VALUE:
                    try:
                        actions.append([VALUE, self.get_value(sample[QUESTION])])
                    except Exception as ex:
                        print(ex)
                        actions.append([VALUE, '0'])
                elif action == PREV_ANSWER:
                    actions.append([ENTITY, PREV_ANSWER])

            self.inference_actions.append({
                QUESTION_TYPE: sample[QUESTION_TYPE],
                QUESTION: sample[QUESTION],
                ANSWER: sample[ANSWER],
                ACTIONS: actions,
                RESULTS: sample[RESULTS],
                PREV_RESULTS: sample[PREV_RESULTS],
                GOLD_ACTIONS: sample[GOLD_ACTIONS] if GOLD_ACTIONS in sample else [],
                IS_CORRECT: 1 if GOLD_ACTIONS in sample and sample[GOLD_ACTIONS] == actions else 0
            })

            if (i+1) % 100 == 0:
                toc = time.perf_counter()
                print(f'==> Finished action construction {((i+1)/len(question_type_inference_data))*100:.2f}% -- {toc - tic:0.2f}s')

        self.write_inference_actions()

    def create_ner_idx_ent_dict(self, ner_indices, context_question):
        """

        :param ner_indices: (OrderedDict[int: str]) {pos_idx: type_id} positions and types of entity entries
        :param context_question: (list[str]) word list of current and previous (context) input from the user
        :return ner_idx_ent: (OrderedDict[int: list[str]]) dictionary of candidate entities and their positions in context_question
            eg: {1: ['Q1'], 2: ['Q1'], 5: ['Q2', 'Q3'], 6: ['Q2', 'Q3']}  # can be ['UNK']
        """
        ent_idx = []
        ner_idx_ent = OrderedDict()

        for index, span_type in ner_indices.items():  # index is just word order in the context question
            if not ent_idx or index-1 == ent_idx[-1][0]:  # NOTE: index-1 == ent_idx[-1][0] one entity will have continuous sequence
                # populate ent_idx with all parts of one entity
                ent_idx.append([index, span_type]) # check whether token start with ## then include previous token also from context_question
                # [[0, 'Q123']]
                # [[0, 'Q123'], [1, 'Q123']]
                # ...
                # until index jumps over to higher value than +1
            else:  # if ent_idx and index-1 != ent_idx[-1][0]:
                # after ent_idx is populated, do search for this entity
                # get ent tokens from input context
                ent_tokens = [context_question[idx] for idx, _ in ent_idx]
                # get string from tokens using tokenizer
                ent_label = self.tokenizer.convert_tokens_to_string(ent_tokens).replace('##', '')  # NOTE: this is label of one entity
                # get elastic search results
                es_results = search_by_label(self.es, ent_label, ent_idx[0][1])  # use type from B tag only (rest is redundant)
                if not es_results:
                    # if no entity was found, generate new entity!
                    es_results = [create_entity(self.es, label=ent_label, types=[ent_idx[0][1]])]
                # add indices to dict
                for idx, _ in ent_idx:
                    ner_idx_ent[idx] = es_results
                # clean ent_idx
                ent_idx = [[index, span_type]]
        if ent_idx:  # NOTE: for the last entry to be considered as well
            # get ent tokens from input context
            ent_tokens = [context_question[idx] for idx, _ in ent_idx]
            # get string from tokens using tokenizer
            ent_label = self.tokenizer.convert_tokens_to_string(ent_tokens).replace('##', '')  # NOTE: this is label of one entity
            # get elastic search results
            es_results = search_by_label(self.es, ent_label, ent_idx[0][1])  # use type from B tag only (rest is redundant)
            if not es_results:
                # if no entity was found, generate new entity!
                es_results = [create_entity(self.es, label=ent_label, types=[ent_idx[0][1]])]
            # add indices to dict
            for idx, _ in ent_idx:
                ner_idx_ent[idx] = es_results
        return ner_idx_ent

    def get_value(self, question):
        if 'min' in question.split():
            value = '0'
        elif 'max' in question.split():
            value = '0'
        elif 'exactly' in question.split():
            value = re.search(r'\d+', question.split('exactly')[1]).group()
        elif 'approximately' in question.split():
            value = re.search(r'\d+', question.split('approximately')[1]).group()
        elif 'around' in question.split():
            value = re.search(r'\d+', question.split('around')[1]).group()
        elif 'atmost' in question.split():
            value = re.search(r'\d+', question.split('atmost')[1]).group()
        elif 'atleast' in question.split():
            value = re.search(r'\d+', question.split('atleast')[1]).group()
        else:
            print(f'Could not extract value from question: {question}')
            value = '0'

        return value

    def write_inference_actions(self):
        with open(f'{ROOT_PATH}/{args.path_inference}/{args.model_path.rsplit("/", 1)[-1].rsplit(".", 2)[0]}_{args.question_type}.json', 'w', encoding='utf-8') as json_file:
            json_file.write(json.dumps(self.inference_actions, indent=4))


def rapidfuzz_query(query, filter_type, kg, res_size=50):
    """
    Fuzzy querry on entity labels and find maximum 'res_size' candidates for relevant entity ids based on Levenshtein distance
    Filter resulting entity_ids by type

    :return: list of filtered entity_ids or unfiltered entity_ids (if filtered is empty)

    """
    max_dist = helpers.get_edit_distance(query)
    res = process.extract(query, kg.labels['entity'], scorer=distance, score_cutoff=max_dist, limit=res_size)
    unfiltered_res = []
    filtered_res = []
    for hit in res:
        ent_id = hit[2]
        # try to filter by types (if type exists in database)
        try:
            ent_type_list = kg.entity_type[ent_id]  # filter by types
            if filter_type in ent_type_list:
                filtered_res.append(ent_id)
        except KeyError:
            print('x', end='')
        unfiltered_res.append(ent_id)

    return filtered_res if filtered_res else unfiltered_res


def save_checkpoint(state):
    filename = f'{ROOT_PATH}/{args.snapshots}/{MODEL_NAME}_e{state[EPOCH]}_v{state[CURR_VAL]:.4f}_{args.task}.pth.tar'
    torch.save(state, filename)


class SingleTaskLoss(nn.Module):
    '''Single Task Loss'''
    def __init__(self, ignore_index):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, output, target):
        return self.criterion(output, target)


class MultiTaskLoss(nn.Module):
    '''Multi Task Learning Loss'''
    def __init__(self, ignore_index):
        super().__init__()
        self.lf_loss = SingleTaskLoss(ignore_index)
        self.ner_loss = SingleTaskLoss(ignore_index)
        self.coref_loss = SingleTaskLoss(ignore_index)
        self.pred_pointer = SingleTaskLoss(ignore_index)
        self.type_pointer = SingleTaskLoss(ignore_index)

        self.mml_emp = torch.Tensor([True, True, True, True, True])
        self.log_vars = torch.nn.Parameter(torch.zeros(len(self.mml_emp)))

    def forward(self, output, target):
        # weighted loss
        task_losses = torch.stack((
            self.lf_loss(output[LOGICAL_FORM], target[LOGICAL_FORM]),
            self.ner_loss(output[NER], target[NER]),
            self.coref_loss(output[COREF], target[COREF]),
            self.pred_pointer(output[PREDICATE_POINTER], target[PREDICATE_POINTER]),
            self.type_pointer(output[TYPE_POINTER], target[TYPE_POINTER]),
        ))

        dtype = task_losses.dtype
        stds = (torch.exp(self.log_vars)**(1/2)).to(DEVICE).to(dtype)
        weights = 1 / ((self.mml_emp.to(DEVICE).to(dtype)+1)*(stds**2))

        losses = weights * task_losses + torch.log(stds)

        return {
            LOGICAL_FORM: losses[0],
            NER: losses[1],
            COREF: losses[2],
            PREDICATE_POINTER: losses[3],
            TYPE_POINTER: losses[4],
            MULTITASK: losses.mean()
        }[args.task]

def init_weights(model):
    # initialize model parameters with Glorot / fan_avg
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

# ANCHOR LASAGNE parameter initialisation
def Embedding(num_embeddings, embedding_dim, padding_idx):
    """Embedding layer"""
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.uniform_(m.weight, -0.1, 0.1)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m

def Linear(in_features, out_features, bias=True):
    """Linear layer"""
    m = nn.Linear(in_features, out_features, bias=bias)
    m.weight.data.uniform_(-0.1, 0.1)
    if bias:
        m.bias.data.uniform_(-0.1, 0.1)
    return m

def LSTM(input_size, hidden_size, **kwargs):
    """LSTM layer"""
    m = nn.LSTM(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name or 'bias' in name:
            param.data.uniform_(-0.1, 0.1)
    return m