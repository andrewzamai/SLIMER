set -x

port=$(shuf -i25000-30000 -n1)

MODEL_NAME_OR_PATH=meta-llama/Llama-2-7b-chat-hf
DATA_DIR=data
DATA_CONFIG_DIR=configs/dataset_configs/task_adaptation_configs
INSTRUCTION_FILE=configs/instruction_configs/instruction.json
OUTPUT_DIR=output/GNER-391xALL-woDeG-LLaMA2-7B-chat

DEEPSPEED_CONFIG=configs/deepspeed_configs/deepspeed_zero1_llama.json
RUN_NAME=GNER-391xALL-woDeG-LLaMA2-7B-chat

deepspeed --include="localhost:0" --master_port $port src/run.py \
    --bf16 True --tf32 True \
    --do_train \
    --predict_with_generate \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --data_dir $DATA_DIR \
    --preprocessing_num_workers 4 \
    --load_best_model_at_end True \
    --metric_for_best_model "eval_average_f1" \
    --max_eval_samples 250 \
    --greater_is_better True \
    --data_config_dir $DATA_CONFIG_DIR \
    --instruction_file $INSTRUCTION_FILE \
    --output_dir $OUTPUT_DIR \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 32 \
    --learning_rate 3e-04 \
    --weight_decay 0.01 \
    --warmup_ratio 0.04 \
    --num_train_epochs 2 \
    --lr_scheduler_type "cosine" \
    --deepspeed $DEEPSPEED_CONFIG \
    --run_name $RUN_NAME \
    --max_source_length 2048 \
    --max_target_length 1024 \
    --overwrite_output_dir \
    --overwrite_cache \
    --logging_strategy "steps" \
    --logging_steps 10 \
    --evaluation_strategy "steps" \
    --eval_steps 250 \
    --save_strategy "steps" \
    --save_steps 250 \
    --seed 1234