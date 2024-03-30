#!/bin/bash
huggingface-cli download lmsys/vicuna-7b-v1.3 --local-dir vicuna &
huggingface-cli download mistralai/Mixtral-8x7b-Instruct-v0.1 --local-dir mixtral --exclude *.safetensors &
huggingface-cli download yuhuili/EAGLE-mixtral-instruct-8x7B --local-dir ea-mixtral &
huggingface-cli download yuhuili/EAGLE-Vicuna-7B-v1.3 --local-dir ea-vicuna &

wait
python EAGLE/convert/convert_hf_checkpoint.py --checkpoint_dir vicuna
python EAGLE/convert/convert_hf_checkpoint_EAGLE.py --checkpoint_dir ea-vicuna --base_dir vicuna
python EAGLE/convert/convert_mixtral.py --checkpoint_dir mixtral
python EAGLE/convert/convert_hf_checkpoint_EAGLE.py --checkpoint_dir ea-mixtral --base_dir mixtral