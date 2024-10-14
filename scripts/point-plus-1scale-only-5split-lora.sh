#!/bin/bash
LOAD='pit-qmm-base'
point_backbone_ckpt=checkpoints/point_bert_v1.2.pt
# deepspeed --master_port 25801
#--deepspeed ./scripts/zero3.json \
for i in $(seq 2 5)
do
    echo "Split $i"
    DATA_FILE=playground/data/ft/point/train_split_$i.json
    deepspeed --master_port 25801 pit_qmm/train/train_mem.py \
        --deepspeed ./scripts/zero3.json \
        --lora_enable True --visual_abstractor_lr 2e-5\
        --model_name_or_path $LOAD \
        --version v1 \
        --data_path $DATA_FILE \
        --image_folder /work/09030/shngt/ls6/lspcqa_views \
        --point_backbone_ckpt $point_backbone_ckpt \
        --fix_pointnet False \
        --image_aspect_ratio pad \
        --group_by_modality_length True \
        --bf16 True \
        --output_dir ./pit-qmm-point-1scale-only-lora-$i \
        --num_train_epochs 30 \
        --per_device_train_batch_size 12 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 1 \
        --evaluation_strategy "no" \
        --save_strategy "steps" \
        --save_steps 800 \
        --save_total_limit 3 \
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
        --report_to wandb \
        --pc_data_path /scratch/09030/shngt/samples_with_MOS \
        --use_two_scale_pc False \
        --use_fp_pc False
done
