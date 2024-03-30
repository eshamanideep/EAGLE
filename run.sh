#!/bin/bash
export PYTHONPATH=$PWD/EAGLE
ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=4 EAGLE/application/uieagle.py --base-model-path mixtral/model.pth --ea-model-path ea-mixtral/model.pth --model-type mixtral --compile

