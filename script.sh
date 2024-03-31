#!/bin/bash
huggingface-cli download mistralai/Mixtral-8x7b-Instruct-v0.1 --local-dir mixtral --exclude *.safetensors &
huggingface-cli download yuhuili/EAGLE-mixtral-instruct-8x7B --local-dir ea-mixtral &
huggingface-cli download mistralai/Mixtral-8x7b-Instruct-v0.1 model-00019-of-00019.safetensors --local-dir mixtral &

wait
python EAGLE/convert/convert_mixtral.py --checkpoint_dir mixtral
python EAGLE/convert/convert_hf_checkpoint_EAGLE.py --checkpoint_dir ea-mixtral --base_dir mixtral