#!/bin/bash
LOAD='pit-qmm-base'
point_backbone_ckpt=checkpoints/point_bert_v1.2.pt
# 
# \
iters=(1 5)
for i in $iters # $(seq 1 5)
do
    echo "Split $i"
    DATA_FILE=playground/data/ft/wpc_desc/train_split_$i.json
    TEST_DATA_FILE=playground/data/ft/wpc_desc/test_split_$i.json
    deepspeed --master_port 25801 pit_qmm/train/train_mem.py \
        --deepspeed ./scripts/zero3.json \
        --lora_enable True --visual_abstractor_lr 2e-5\
        --model_name_or_path $LOAD \
        --version v1 \
        --data_path $DATA_FILE \
        --image_folder /home1/09030/shngt/work/pit-qmm/point_qa_datagen/wpc_views \
        --point_backbone_ckpt $point_backbone_ckpt \
        --fix_pointnet False \
        --image_aspect_ratio pad \
        --group_by_modality_length True \
        --bf16 True \
        --output_dir /scratch/09030/shngt/pit-qmm-ckpts/wpc-1scale-only-desc-50-lora-$i \
        --num_train_epochs 50 \
        --per_device_train_batch_size 6 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 1 \
        --evaluation_strategy "no" \
        --save_strategy "steps" \
        --save_steps 100 \
        --save_total_limit 10 \
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
        --pc_data_path /scratch/09030/shngt/the_WPC_database/distorted_PCs \
        --use_two_scale_pc False \
        --use_fp_pc False # \
        # --eval_data_path $TEST_DATA_FILE
done
