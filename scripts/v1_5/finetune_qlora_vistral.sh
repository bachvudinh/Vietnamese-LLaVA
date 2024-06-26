#!/bin/bash

deepspeed llava/train/train_mem.py \
    --deepspeed  ./scripts/zero2.json \
    --lora_enable True --lora_r 128 --lora_alpha 256 \
    --bits 4 \
    --freeze_backbone False \
    --tune_mm_mlp_adapter False \
    --model_name_or_path Viet-Mistral/Vistral-7B-Chat \
    --version vistral-it \
    --data_path ./playground/data/train_llava.json \
    --image_folder ./finetune_data/ \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter ./llava-vistral-7b-pretrain/checkpoint-144000/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir ./llava-vistral-7b-IT-lora/ \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 4 \
    --learning_rate 4e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 1024 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
