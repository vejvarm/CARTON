import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import json
import argparse
from glob import glob
from pathlib import Path
from knowledge_graph.knowledge_graph import KnowledgeGraph
from executor import ActionExecutor
from meters import AccuracyMeter, F1scoreMeter
ROOT_PATH = Path(os.path.dirname(__file__)).parent

# add arguments to parser
parser = argparse.ArgumentParser(description='Execute actions')
parser.add_argument('--file_path', default='/data/final/csqa/process/test.json', help='json file with actions')
parser.add_argument('--question_type', default='Clarification', help='type of questions')
parser.add_argument('--max_results', default=1000, help='maximum number of results')
args = parser.parse_args()

# load kg
kg = KnowledgeGraph()

# load action executor
action_executor = ActionExecutor(kg)

# define question type meters
question_types_meters = {
    'Clarification': F1scoreMeter(),
    'Comparative Reasoning (All)': F1scoreMeter(),
    'Logical Reasoning (All)': F1scoreMeter(),
    'Quantitative Reasoning (All)': F1scoreMeter(),
    'Simple Question (Coreferenced)': F1scoreMeter(),
    'Simple Question (Direct)': F1scoreMeter(),
    'Simple Question (Ellipsis)': F1scoreMeter(),
    # -------------------------------------------
    'Verification (Boolean) (All)': AccuracyMeter(),
    'Quantitative Reasoning (Count) (All)': AccuracyMeter(),
    'Comparative Reasoning (Count) (All)': AccuracyMeter()
}


def run_question_type(file_path=args.file_path):
    count_no_answer = 0
    count_wrong_type = 0
    count_total = 0

    # load data
    data_path = f'{str(ROOT_PATH)}{file_path}'
    with open(data_path) as json_file:
        data = json.load(json_file)

    tic = time.perf_counter()
    for i, d in enumerate(data):
        print(d['question_type'])
        if d['question_type'] != args.question_type:
            continue

        count_total += 1
        try:
            if d['actions'] is not None:
                all_actions = [action[1] for action in d['actions']]
                if 'entity' not in all_actions:
                    result = action_executor(d['actions'], d['prev_results'], d['question_type'])
                else:
                    count_no_answer += 1
                    result = set([])
            else:
                count_no_answer += 1
                result = set([])
        except Exception as ex:
            print(d['question'])
            print(d['actions'])
            print(ex)
            count_no_answer += 1
            result = set([])

        try:
            if d['question_type'] == 'Verification (Boolean) (All)':
                answer = True if d['answer'] == 'YES' else False
                question_types_meters[d['question_type']].update(answer, result)
            else:
                if d['question_type'] in ['Quantitative Reasoning (Count) (All)',
                                          'Comparative Reasoning (Count) (All)']:
                    if d['answer'].isnumeric():
                        question_types_meters[d['question_type']].update(int(d['answer']), len(result))
                    else:
                        question_types_meters[d['question_type']].update(len(d['results']), len(result))
                else:
                    if not isinstance(result, set):  # TODO: does this break results?
                        result = {result}
                        count_wrong_type += 1
                        print(result)
                    if result != set(d['results']) and len(result) > args.max_results:
                        new_result = result.intersection(set(d['results']))
                        for res in result:
                            if res not in result: new_result.add(res)
                            if len(new_result) == args.max_results: break
                        result = new_result.copy()
                    gold = set(d['results'])
                    question_types_meters[d['question_type']].update(gold, result)
        except Exception as ex:
            print(f"Q: {d['question']}")
            print(f"A: {d['actions']}")
            print(f"result: {result}")
            print(f"gold result: {d['results']}")
            raise ValueError(ex)

        toc = time.perf_counter()
        print(f'==> Finished {((i + 1) / len(data)) * 100:.2f}% -- {toc - tic:0.2f}s')

    # print results
    result_list = [str(args.question_type),
                   f'\nNA actions: {count_no_answer}\n',
                   f'Wrong type: {count_wrong_type}\n',
                   f'Total samples: {count_total}\n',
                   ]
    print(args.question_type)
    print(f'NA actions: {count_no_answer}')
    print(f'Wrong type: {count_wrong_type}')
    print(f'Total samples: {count_total}')
    if args.question_type in ['Verification (Boolean) (All)', 'Quantitative Reasoning (Count) (All)',
                              'Comparative Reasoning (Count) (All)']:
        # print(f'Accuracy: {question_types_meters[args.question_type].accuracy}')
        result_list.append(f'Accuracy: {question_types_meters[args.question_type].accuracy}\n')
    else:
        result_list.append(f'Precision: {question_types_meters[args.question_type].precision}')
        print(f'Precision: {question_types_meters[args.question_type].precision}\n')
        result_list.append(f'Recall: {question_types_meters[args.question_type].recall}')
        print(f'Recall: {question_types_meters[args.question_type].recall}\n')
        result_list.append(f'F1-score: {question_types_meters[args.question_type].f1_score}')
        print(f'F1-score: {question_types_meters[args.question_type].f1_score}\n')

    with open(f"{str(ROOT_PATH)}{os.path.splitext(file_path)[0]}.log", "w") as result_log:
        result_log.writelines(result_list)


if args.question_type == 'all':
    for key in question_types_meters.keys():
        args.question_type = key
        relative_path = f"{args.file_path}{args.question_type}.json"
        print(f"relative path: {relative_path}")
        run_question_type(file_path=relative_path)
else:
    run_question_type()

