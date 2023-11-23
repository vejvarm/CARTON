#!/bin/bash

# CARTONNER trained on CSQA dataset
python inference.py --name "00_csqa_on_csqa" --batch-size 40 --model-path experiments/models/CARTONNER_csqa_e10_v0.0102_multitask.pth.tar --data-path data/final/csqa --cache-path .cache/csqa/

python inference.py --name "00_csqa_on_merged" --batch-size 40 --model-path experiments/models/CARTONNER_csqa_e10_v0.0102_multitask.pth.tar --data-path data/csqa-merged --cache-path .cache/merged/

# CARTONNER trained on MERGED dataset for 10 epochs
python inference.py --name "01_merged10_on_csqa" --batch-size 40 --model-path experiments/models/CARTONNER_merged_e10_v0.0153_multitask.pth.tar --data-path data/final/csqa --cache-path .cache/csqa/

python inference.py --name "01_merged10_on_merged" --batch-size 40 --model-path experiments/models/CARTONNER_merged_e10_v0.0153_multitask.pth.tar --data-path data/csqa-merged --cache-path .cache/merged/

# CARTONNER trained on MERGED dataset for 15 epochs
python inference.py --name "02_merged15_on_csqa" --batch-size 40 --model-path experiments/models/CARTONNER_merged_e15_v0.0146_multitask.pth.tar --data-path data/final/csqa --cache-path .cache/csqa/

python inference.py --name "02_merged15_on_merged" --batch-size 40 --model-path experiments/models/CARTONNER_merged_e15_v0.0146_multitask.pth.tar --data-path data/csqa-merged --cache-path .cache/merged/