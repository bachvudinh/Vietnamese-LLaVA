#!/bin/bash

python train_llava.py \
    --cache_dir ./.cache/ \
    --base_model mistral \
    --lora_enable True --lora_r 32 --lora_alpha 64 \
    --freeze_backbone False \
    --tune_mm_mlp_adapter False \
    --model_name_or_path Viet-Mistral/Vistral-7B-Chat \
    --version v1 \
    --data_path ./dataset/train_llava_improved.json \
    --image_folder ./dataset/ \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter ./llava-vistral-7b-pretrain/checkpoint-144000/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --image_aspect_ratio square \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir ./llava-vistral-7b-IT-lora-1/ \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 4 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 1024 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to wandb
