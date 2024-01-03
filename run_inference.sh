#!/bin/bash

# CARTONNER trained on CSQA dataset for 10 epochs
python inference.py --name "00_csqa10_on_csqa" --batch-size 50 --model-path experiments/models/CARTONNER_csqa15_e10_v0.0145_multitask.pth.tar --data-path data/final/csqa --cuda-device 0

python inference.py --name "00_csqa10_on_merged" --batch-size 50 --model-path experiments/models/CARTONNER_csqa15_e10_v0.0145_multitask.pth.tar --data-path data/csqa-merged --cuda-device 0

python inference.py --name "00_csqa10_on_d2t" --batch-size 50 --model-path experiments/models/CARTONNER_csqa15_e10_v0.0145_multitask.pth.tar --data-path data/d2t-sampled --cuda-device 0

# CARTONNER trained on MERGED dataset for 10 epochs
python inference.py --name "01_merged10_on_csqa" --batch-size 50 --model-path experiments/models/CARTONNER_merged15_e10_v0.0258_multitask.pth.tar --data-path data/final/csqa --cuda-device 0

python inference.py --name "01_merged10_on_merged" --batch-size 50 --model-path experiments/models/CARTONNER_merged15_e10_v0.0258_multitask.pth.tar --data-path data/csqa-merged --cuda-device 0

python inference.py --name "01_merged10_on_d2t" --batch-size 50 --model-path experiments/models/CARTONNER_merged15_e10_v0.0258_multitask.pth.tar --data-path data/d2t-sampled --cuda-device 0


# CARTONNER trained on CSQA dataset for 15 epochs
python inference.py --name "02_csqa15_on_csqa" --batch-size 50 --model-path experiments/models/CARTONNER_csqa15_e15_v0.0141_multitask.pth.tar --data-path data/final/csqa --cuda-device 0

python inference.py --name "02_csqa15_on_merged" --batch-size 50 --model-path experiments/models/CARTONNER_csqa15_e15_v0.0141_multitask.pth.tar --data-path data/csqa-merged --cuda-device 0

python inference.py --name "02_csqa15_on_d2t" --batch-size 50 --model-path experiments/models/CARTONNER_csqa15_e15_v0.0141_multitask.pth.tar --data-path data/d2t-sampled --cuda-device 0


# CARTONNER trained on MERGED dataset for 15 epochs
python inference.py --name "03_merged15_on_csqa" --batch-size 50 --model-path experiments/models/CARTONNER_merged15_e15_v0.0258_multitask.pth.tar --data-path data/final/csqa --cuda-device 0

python inference.py --name "03_merged15_on_merged" --batch-size 50 --model-path experiments/models/CARTONNER_merged15_e15_v0.0258_multitask.pth.tar --data-path data/csqa-merged --cuda-device 0

python inference.py --name "03_merged15_on_d2t" --batch-size 50 --model-path experiments/models/CARTONNER_merged15_e15_v0.0258_multitask.pth.tar --data-path data/d2t-sampled --cuda-device 0