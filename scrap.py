from dataset import CSQADataset

from constants import *

if __name__ == '__main__':
    dataset = CSQADataset()
    inference_data = dataset.get_inference_data()

    question_type = 'Simple Question (Coreferenced)'
    question_type_inference_data = [data for data in inference_data if question_type in data[QUESTION_TYPE]]

    for i, sample in enumerate(question_type_inference_data):
        print(f'Q: {sample[CONTEXT_QUESTION]}\nE: {sample[CONTEXT_ENTITIES]}')
        break