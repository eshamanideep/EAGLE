#!/bin/bash
huggingface-cli download lmsys/vicuna-7b-v1.3 --local-dir vicuna &
huggingface-cli download yuhuili/EAGLE-Vicuna-7B-v1.3 --local-dir ea-vicuna &

wait
python EAGLE/convert/convert_hf_checkpoint.py --checkpoint_dir vicuna
python EAGLE/convert/convert_hf_checkpoint_EAGLE.py --checkpoint_dir ea-vicuna --base_dir vicuna