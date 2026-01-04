


# DISABLE_VERSION_CHECK=1  CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file ./examples/train_lora/fsdp_config.yaml ./src/train.py \
#  --stage dpo --do_train  --model_name_or_path /home/ubisec/swh/train_models/DS-R1-0528-Qwen3-8B_cpp_completion_20250609_lora-sft_model13 \
#  --dataset  'c_cpp_completion_dpo_train_data_20250609_1600' --dataset_dir  data/ --template qwen3  --finetuning_type lora \
#  --lora_rank 16  --lora_alpha 32 --lora_dropout 0.05 --lora_target all  \
#  --output_dir /home/ubisec/swh/train_models/DS-R1-0528-Qwen3-8B_cpp_completion_20250609_lora-dpo_adapter8 --use_dora False \
#  --pref_loss orpo --pref_beta 0.1  --overwrite_cache --overwrite_output_dir --cutoff_len 8194 --enable_liger_kernel  \
#  --flash_attn fa2 --preprocessing_num_workers 16 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 \
#  --gradient_accumulation_steps 1 --lr_scheduler_type cosine  --logging_steps 10  --warmup_ratio 0.1  --save_steps 1000  --eval_steps 500  --eval_strategy steps  \
#  --load_best_model_at_end    --learning_rate 5e-6 --loraplus_lr_ratio 12.0 --num_train_epochs 6.0  --val_size 0.05 --ddp_timeout 180000000  \
#  --plot_loss  --bf16 --optim adamw_torch >> log/DS-R1-0528-Qwen3-8B_cpp_completion_20250609_lora-dpo-8_20250718.log

# llamafactory-cli export examples/merge_lora/qwen3_lora_dpo.yaml >> log/merge_250716.log


# DISABLE_VERSION_CHECK=1  CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file ./examples/train_lora/fsdp_config.yaml ./src/train.py \
#  --stage dpo --do_train  --model_name_or_path /home/ubisec/swh/train_models/selekt_stage1_instruction_train13/checkpoint-123 \
#  --dataset  'c_cpp_completion_dpo_train_data_20250609_1600' --dataset_dir  data/ --template qwen3  --finetuning_type lora \
#  --lora_rank 16  --lora_alpha 32 --lora_dropout 0.05 --lora_target all  \
#  --output_dir /home/ubisec/swh/train_models/DS-R1-0528-Qwen3-8B_cpp_completion_20250609_lora-dpo_adapter9 --use_dora False \
#  --pref_loss orpo --pref_beta 0.1  --overwrite_cache --overwrite_output_dir --cutoff_len 8194 --enable_liger_kernel  \
#  --flash_attn fa2 --preprocessing_num_workers 16 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 \
#  --gradient_accumulation_steps 1 --lr_scheduler_type cosine  --logging_steps 10  --warmup_ratio 0.1  --save_steps 1000  --eval_steps 500  --eval_strategy steps  \
#  --load_best_model_at_end    --learning_rate 5e-6 --loraplus_lr_ratio 12.0 --num_train_epochs 6.0  --val_size 0.05 --ddp_timeout 180000000  \
#  --plot_loss  --bf16 --optim adamw_torch >> log/DS-R1-0528-Qwen3-8B_cpp_completion_20250609_lora-dpo-9_20250718.log


# OOM
# DISABLE_VERSION_CHECK=1  CUDA_VISIBLE_DEVICES=0,2 accelerate launch --main_process_port 0 --config_file ./examples/train_lora/fsdp_config.yaml ./src/train.py \
#  --stage dpo --do_train  --model_name_or_path /home/ubisec/swh/models/deepseek-ai-DeepSeek-R1-0528-Qwen3-8B \
#  --dataset  'c_cpp_completion_dpo_train_data_20250609_1600_random800' --dataset_dir  data/ --template qwen3  --finetuning_type lora \
#  --lora_rank 64  --lora_alpha 16 --lora_dropout 0.05 --lora_target all  \
#  --output_dir /home/ubisec/swh/train_models/DS-R1-0528-Qwen3-8B_cpp_completion_20250609_lora-llamafactory-orpo-adapter-12 --use_dora False \
#  --pref_loss orpo --pref_beta 0.05  --overwrite_cache --overwrite_output_dir --cutoff_len 6144 --enable_liger_kernel  \
#  --flash_attn fa2 --preprocessing_num_workers 16 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 \
#  --gradient_accumulation_steps 1 --lr_scheduler_type cosine  --logging_steps 10  --warmup_ratio 0.1  --save_steps 1000  --eval_steps 500  --eval_strategy steps  \
#  --load_best_model_at_end  --learning_rate 8e-5 --loraplus_lr_ratio 12.0 --num_train_epochs 6.0  --val_size 0.05 --ddp_timeout 180000000  --plot_loss  --bf16 --optim rmsprop \
#  > log/DS-R1-0528-Qwen3-8B_cpp_completion_20250609_lora-llamaf-orpo-12_20250823.log

# DISABLE_VERSION_CHECK=1  CUDA_VISIBLE_DEVICES=0,2 accelerate launch --main_process_port 0 --config_file ./examples/train_lora/fsdp_config.yaml ./src/train.py \
#  --stage dpo --do_train  --model_name_or_path /home/ubisec/swh/models/deepseek-ai-DeepSeek-R1-0528-Qwen3-8B \
#  --dataset  'c_cpp_completion_dpo_train_data_20250609_1600_random800' --dataset_dir  data/ --template qwen3  --finetuning_type lora \
#  --lora_rank 64  --lora_alpha 16 --lora_dropout 0.05 --lora_target all  \
#  --output_dir /home/ubisec/swh/train_models/DS-R1-0528-Qwen3-8B_cpp_completion_20250609_lora-llamafactory-orpo-adapter-12 --use_dora False \
#  --pref_loss orpo --pref_beta 0.05  --overwrite_cache --overwrite_output_dir --cutoff_len 3072 --enable_liger_kernel  \
#  --flash_attn fa2 --preprocessing_num_workers 16 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 \
#  --gradient_accumulation_steps 1 --lr_scheduler_type cosine  --logging_steps 10  --warmup_ratio 0.1  --save_steps 1000  --eval_steps 500  --eval_strategy steps  \
#  --load_best_model_at_end  --learning_rate 8e-5 --loraplus_lr_ratio 12.0 --num_train_epochs 6.0  --val_size 0.05 --ddp_timeout 180000000  --plot_loss  --bf16 --optim rmsprop \
#   > log/DS-R1-0528-Qwen3-8B_cpp_completion_20250609_lora-llamaf-orpo-12_20250823.log

# DISABLE_VERSION_CHECK=1  CUDA_VISIBLE_DEVICES=0,2 accelerate launch --main_process_port 0 --config_file ./examples/train_lora/fsdp_config.yaml ./src/train.py \
#  --stage dpo --do_train  --model_name_or_path /home/ubisec/swh/models/deepseek-ai-DeepSeek-R1-0528-Qwen3-8B \
#  --dataset  'c_cpp_completion_dpo_train_data_20250609_1600_random800' --dataset_dir  data/ --template qwen3  --finetuning_type lora \
#  --lora_rank 64  --lora_alpha 16 --lora_dropout 0.05 --lora_target all  \
#  --output_dir /home/ubisec/swh/train_models/DS-R1-0528-Qwen3-8B_cpp_completion_20250609_lora-llamafactory-orpo-adapter-13 --use_dora False \
#  --pref_loss orpo --pref_beta 0.05  --overwrite_cache --overwrite_output_dir --cutoff_len 4096 --enable_liger_kernel  \
#  --flash_attn fa2 --preprocessing_num_workers 16 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 \
#  --gradient_accumulation_steps 1 --lr_scheduler_type cosine  --logging_steps 10  --warmup_ratio 0.1  --save_steps 1000  --eval_steps 500  --eval_strategy steps  \
#  --load_best_model_at_end  --learning_rate 8e-5 --loraplus_lr_ratio 12.0 --num_train_epochs 6.0  --val_size 0.05 --ddp_timeout 180000000  --plot_loss  --bf16 --optim rmsprop \
#   > log/DS-R1-0528-Qwen3-8B_cpp_completion_20250609_lora-llamaf-orpo-13_20250823.log

DISABLE_VERSION_CHECK=1  CUDA_VISIBLE_DEVICES=0,2 accelerate launch --main_process_port 0 --config_file ./examples/train_lora/fsdp_config.yaml ./src/train.py \
 --stage dpo --do_train  --model_name_or_path /home/ubisec/swh/models/deepseek-ai-DeepSeek-R1-0528-Qwen3-8B \
 --dataset  'c_cpp_completion_dpo_train_data_20250609_1600_random800' --dataset_dir  data/ --template qwen3  --finetuning_type lora \
 --lora_rank 128  --lora_alpha 32 --lora_dropout 0.05 --lora_target all  \
 --output_dir /home/ubisec/swh/train_models/DS-R1-0528-Qwen3-8B_cpp_completion_20250609_lora-llamafactory-orpo-adapter-14 --use_dora False \
 --pref_loss orpo --pref_beta 0.05  --overwrite_cache --overwrite_output_dir --cutoff_len 8192 --enable_liger_kernel  \
 --flash_attn fa2 --preprocessing_num_workers 16 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 \
 --gradient_accumulation_steps 1 --lr_scheduler_type cosine  --logging_steps 10  --warmup_ratio 0.1  --save_steps 500 \
 --learning_rate 8e-5 --loraplus_lr_ratio 12.0 --num_train_epochs 12.0  --ddp_timeout 180000000  --plot_loss  --bf16 --optim rmsprop \
 --quantization_bit 4 --quantization_method bnb --double_quantization true \
  > log/DS-R1-0528-Qwen3-8B_cpp_completion_20250609_lora-llamaf-orpo-14_20250823.log
  #  --val_size 0.0  --eval_steps 500  --eval_strategy steps  --load_best_model_at_end

llamafactory-cli export examples/merge_lora/qwen3_lora_llama-factory-orpo.yaml >> log/merge_250823.log

CUDA_VISIBLE_DEVICES=0  llamafactory-cli train examples/train_lora/qwen3_lora_dpo_orpo.yaml > log/DS-R1-0528-Qwen3-8B_cpp_completion_20250609_lora-llamaf-orpo-14_20250823.log

# conda activate swh-vllm

# cd ../AssessModel

# python llm_pre_assess.py >> data/R1_0528_Qwen3_8B-lora-dpo_20250702/cpp_train8_4000_250719.cpp

# python llm_pre_assess_250719.py >> data/R1_0528_Qwen3_8B-lora-dpo_20250702/cpp_train9_4000_250719.cpp

# python cal_rates.py