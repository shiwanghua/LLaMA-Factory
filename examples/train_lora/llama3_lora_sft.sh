#!/bin/bash

set -x

MODEL_PATH=meta-llama/Meta-Llama-3-8B-Instruct

llamafactory-cli train \
    --model_name_or_path ${MODEL_PATH} \
    --trust_remote_code \
    --stage sft \
    --do_train \
    --finetuning_type lora \
    --lora_rank 8 \
    --lora_target all \
    --dataset identity,alpaca_en_demo \
    --template llama3 \
    --cutoff_len 2048 \
    --max_samples 1000 \
    --overwrite_cache \
    --preprocessing_num_workers 16 \
    --dataloader_num_workers 4 \
    --output_dir saves/llama3-8b/lora/sft \
    --logging_steps 10 \
    --save_steps 500 \
    --plot_loss \
    --overwrite_output_dir \
    --save_only_model false \
    --report_to none \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --num_train_epochs 3.0 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --bf16 \
    --ddp_timeout 180000000



# DISABLE_VERSION_CHECK=1 CUDA_VISIBLE_DEVICES=0  python  ./LLaMA-Factory/src/train.py \
#         --stage sft --do_train \
#         --model_name_or_path /home/valiantsec/cjr/models/merged/merged_kit_qwen_model_3/  \
#         --dataset 'instruction,multiround,functioncall' \
#         --dataset_dir /home/valiantsec/cjr/data/V2_2  \
#         --template qwen \
#         --finetuning_type lora  \
#         --lora_rank 128  \
#         --lora_alpha 32  \
#         --lora_dropout 0  \
#         --lora_target all \
#         --output_dir ./sft/qwen_unsloth_sft_v4_3  \
#         --overwrite_cache  \
#         --overwrite_output_dir \
#         --cutoff_len 16384  \
#         --flash_attn fa2  \
#         --preprocessing_num_workers 16  \
#         --per_device_train_batch_size 1   \
#         --per_device_eval_batch_size 1  \
#         --gradient_accumulation_steps 1  \
#         --lr_scheduler_type cosine \
#         --logging_steps 10  \
#         --warmup_ratio 0   \
#         --save_steps 1000   \
#         --max_grad_norm 1.0  \
#         --eval_strategy no  \
#         --learning_rate 5e-6   \
#         --num_train_epochs 2.0  \
#         --val_size 0.05  \
#         --ddp_timeout 180000000 \
#         --plot_loss \
#         --bf16 --report_to none --use_rslora --use_unsloth \
#         --optim adamw_8bit --quantization_bit 4  --do_eval False

# DISABLE_VERSION_CHECK=1 CUDA_VISIBLE_DEVICES=0,1,2,3 python ./LLaMA-Factory/src/export_model.py \
#     --model_name_or_path /home/valiantsec/cjr/models/merged/mergekit_qwen_v4_pt42k_20250425p_v1 \
#     --adapter_name_or_path /home/valiantsec/cjr/sft/qwen32bCoder_sft_functioncall_only \
#     --template qwen\
#     --finetuning_type lora \
#     --export_dir  /home/valiantsec/cjr/models/merged/mergelora_qwen_v4_sft_20250512p_v1 \
#     --export_size 9 \
#     --export_device auto \
#     --export_legacy_format False