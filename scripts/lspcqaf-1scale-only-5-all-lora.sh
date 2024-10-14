#!/bin/bash
LOAD='pit-qmm-base'
point_backbone_ckpt=checkpoints/point_bert_v1.2.pt
# 
# \
# for i in $(seq 5 5)
# do
# echo "Split $i"
DATA_FILE=/work/09030/shngt/ls6/pcq-lmm/point_qa_datagen/lspcqa_full_instruct_discrete.json
deepspeed --master_port 25801 pit_qmm/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --lora_enable True --visual_abstractor_lr 2e-5\
    --model_name_or_path $LOAD \
    --version v1 \
    --data_path $DATA_FILE \
    --image_folder /scratch/09030/shngt/lspcqa_full_views_b_bg \
    --point_backbone_ckpt $point_backbone_ckpt \
    --fix_pointnet False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /scratch/09030/shngt/pit-qmm-ckpts/lspcqaf-1-scale-only-5-all \
    --num_train_epochs 5 \
    --per_device_train_batch_size 13 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 50 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --tune_visual_abstractor True \
    --freeze_vision_model False \
    --dataloader_num_workers 28 \
    --lazy_preprocess True \
    --pc_data_path /scratch/09030/shngt/lspcqa_full/all \
    --use_two_scale_pc False \
    --use_fp_pc False \
    --freeze_vision_model True \
    --fix_pointnet True
# done
