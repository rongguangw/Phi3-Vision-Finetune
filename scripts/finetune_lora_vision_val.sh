#!/bin/bash

export PYTHONPATH=src:$PYTHONPATH

deepspeed src/training/train.py \
    --lora_enable True \
    --vision_lora True \
    --lora_namespan_exclude "['lm_head']" \
    --lora_rank 32 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --num_lora_modules -1 \
    --deepspeed scripts/zero3.json \
    --model_id Phi-3-vision-128k-instruct \
    --data_path data/chartvqa_images.json \
    --image_folder data/chartvqa_images \
    --tune_img_projector True \
    --freeze_vision_tower False \
    --bf16 True \
    --output_dir output/lora_vision_test \
    --num_train_epochs 5 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to wandb \
    --lazy_preprocess True \
    --dataloader_num_workers 0 \
    --eval_strategy steps \
    --per_device_eval_batch_size 4 \
    --eval_accumulation_steps 1 \
    --eval_steps 10 \
    --data_path_val data/local_images.json \
    --image_folder_val data/local_images