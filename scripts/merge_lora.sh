#!/bin/bash

export PYTHONPATH=src:$PYTHONPATH

python src/merge_lora_weights.py \
    --model-path output/lora_vision_test \
    --model-base Phi-3-vision-128k-instruct  \
    --save-model-path output/vision_merge \
    --safe-serialization