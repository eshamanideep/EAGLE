#!/bin/bash
export PYTHONPATH=$PWD/EAGLE
python EAGLE/application/uieagle.py --base-model-path vicuna/model.pth --ea-model-path ea-vicuna/model.pth --model-type vicuna --compile