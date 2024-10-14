#!/bin/bash
LOAD='pit-qmm-base'
point_backbone_ckpt=checkpoints/point_bert_v1.2.pt

# --bf16 True \

for i in $(seq 1 5)
do
    echo "Split $i"
    DATA_FILE=playground/data/ft/wpc/train_split_$i.json
    deepspeed --master_port 25801 pit_qmm/train/train_mem.py \
        --deepspeed ./scripts/zero3.json \
        --lora_enable True --visual_abstractor_lr 2e-5\
        --model_name_or_path $LOAD \
        --version v1 \
        --data_path $DATA_FILE \
        --image_folder /scratch/09030/shngt/wpc_projections \
        --point_backbone_ckpt $point_backbone_ckpt \
        --fix_pointnet False \
        --image_aspect_ratio pad \
        --group_by_modality_length True \
        --output_dir ./pit-qmm-wpc-2scale-mm-olora-$i \
        --num_train_epochs 30 \
        --per_device_train_batch_size 12 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 1 \
        --evaluation_strategy "no" \
        --bf16 True \
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
        --dataloader_num_workers 28 \
        --lazy_preprocess True \
        --pc_data_path /scratch/09030/shngt/the_WPC_database/distorted_PCs \
        --use_two_scale_pc True \
        --use_fp_pc True \
        --freeze_vision_model True \
        --fix_pointnet True
done
