#!/bin/bash
export PYTHONPATH=$PWD/EAGLE
ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=2 EAGLE/application/uieagle.py --base-model-path vicuna/model.pth --ea-model-path ea-vicuna/model.pth --model-type vicuna --compile