#!/bin/bash

# CARTONNER trained on CSQA dataset for 10 epochs
python inference_for_ner_recall.py --name "00_csqa10_on_csqa" --batch-size 25 --model-path experiments/models/CARTONNER_csqa15_e10_v0.0154_multitask.pth.tar --data-path data/final/csqa --cuda-device 1

python inference_for_ner_recall.py --name "00_csqa10_on_merged" --batch-size 25 --model-path experiments/models/CARTONNER_csqa15_e10_v0.0154_multitask.pth.tar --data-path data/csqa-merged --cuda-device 1

python inference_for_ner_recall.py --name "00_csqa10_on_d2t" --batch-size 25 --model-path experiments/models/CARTONNER_csqa15_e10_v0.0154_multitask.pth.tar --data-path data/d2t-sampled --cuda-device 1

# CARTONNER trained on MERGED dataset for 10 epochs
python inference_for_ner_recall.py --name "01_merged10_on_csqa" --batch-size 25 --model-path experiments/models/CARTONNER_merged15_e10_v0.0273_multitask.pth.tar --data-path data/final/csqa --cuda-device 1

python inference_for_ner_recall.py --name "01_merged10_on_merged" --batch-size 25 --model-path experiments/models/CARTONNER_merged15_e10_v0.0273_multitask.pth.tar --data-path data/csqa-merged --cuda-device 1

python inference_for_ner_recall.py --name "01_merged10_on_d2t" --batch-size 25 --model-path experiments/models/CARTONNER_merged15_e10_v0.0273_multitask.pth.tar --data-path data/d2t-sampled --cuda-device 1


# CARTONNER trained on CSQA dataset for 15 epochs
python inference_for_ner_recall.py --name "02_csqa15_on_csqa" --batch-size 25 --model-path experiments/models/CARTONNER_csqa15_e15_v0.0148_multitask.pth.tar --data-path data/final/csqa --cuda-device 1

python inference_for_ner_recall.py --name "02_csqa15_on_merged" --batch-size 25 --model-path experiments/models/CARTONNER_csqa15_e15_v0.0148_multitask.pth.tar --data-path data/csqa-merged --cuda-device 1

python inference_for_ner_recall.py --name "02_csqa15_on_d2t" --batch-size 25 --model-path experiments/models/CARTONNER_csqa15_e15_v0.0148_multitask.pth.tar --data-path data/d2t-sampled --cuda-device 1


# CARTONNER trained on MERGED dataset for 15 epochs
python inference_for_ner_recall.py --name "03_merged15_on_csqa" --batch-size 25 --model-path experiments/models/CARTONNER_merged_e15_v0.0146_multitask.pth.tar --data-path data/final/csqa --cuda-device 1

python inference_for_ner_recall.py --name "03_merged15_on_merged" --batch-size 25 --model-path experiments/models/CARTONNER_merged_e15_v0.0146_multitask.pth.tar --data-path data/csqa-merged --cuda-device 1

python inference_for_ner_recall.py --name "03_merged15_on_d2t" --batch-size 25 --model-path experiments/models/CARTONNER_merged_e15_v0.0146_multitask.pth.tar --data-path data/d2t-sampled --cuda-device 1