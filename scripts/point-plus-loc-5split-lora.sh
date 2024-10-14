#!/bin/bash
LOAD='q-future/one-align'
point_backbone_ckpt=checkpoints/point_bert_v1.2.pt
# deepspeed --master_port 25801
#--deepspeed ./scripts/zero3.json \
for i in $(seq 3 3)
do
    echo "Split $i"
    DATA_FILE=playground/data/ft/point_loc/train_split_$i.json
    python pit_qmm/train/train_mem.py \
        --lora_enable True --visual_abstractor_lr 2e-5\
        --model_name_or_path $LOAD \
        --version v1 \
        --data_path $DATA_FILE \
        --image_folder /proj/esv-summer-interns/home/eguhpas/point_qa_datagen/lspcqa_syn_views \
        --point_backbone_ckpt $point_backbone_ckpt \
        --fix_pointnet False \
        --image_aspect_ratio pad \
        --group_by_modality_length True \
        --bf16 True \
        --output_dir ./pit-qmm-point-plus-loc-lora-$i \
        --num_train_epochs 20 \
        --per_device_train_batch_size 15 \
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
        --report_to wandb
done