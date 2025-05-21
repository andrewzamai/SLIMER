#!/bin/bash
#SBATCH --job-name=eval_Qwen

#SBATCH -A dmx@h100
#SBATCH -C h100
#SBATCH --qos=qos_gpu_h100-dev  # Quality of service up 2h otherwise qos_gpu_a100-t3
#SBATCH --time=01:00:00

#SBATCH --nodes=1               # number of nodes
#SBATCH --ntasks-per-node=1     # number of tasks per node
#SBATCH --gres=gpu:1            # number of GPUs per node

#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread    # disable hyperthreading

#SBATCH --output=eval_LLM/eval_LLM_%j.out 
#SBATCH --error=eval_LLM/eval_LLM_%j.err   

module purge
module load arch/h100
module load cuda/12.4.1
module load nccl/2.21.5-1-cuda
module load cudnn/9.2.0.82-cuda
#module load intel-compilers/2021.9.0   # for a100 only?

# load conda env with pytorch 2.5.1 cuda12.4.1, vllm==0.7.2, flash_attn, deepspeed, accelerate ...
source ~/.bashrc
conda activate venv_py26  # vllm==0.7.2

export PYTHONPATH=$WORK/SLIMER

export HF_HOME=$WORK/.cache/.huggingface
export XDG_CACHE_HOME=$WORK/.cache

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

set -x -e

# Set your base model and output directories
BASE_MODEL="Qwen/Qwen3-8B"

#TRAINED_MODEL="./trained_models/"
#MERGED_MODEL="./merged_models/"

# Merge LoRA weights
#python src/    /src/commons/merge_lora_weights.py "$BASE_MODEL" "$TRAINED_MODEL" "$MERGED_MODEL"

python src/SFT_finetuning/evaluating/evaluate_SLIMER_PARALLEL_GRPO.py \
--merged_model_name "$BASE_MODEL" \
--max_tagNames_per_prompt 20 \
--temperature 0.7 \
--top_p 0.8 \
--top_k 20 \
--min_p 0 \
--num_GPUs 1 \
--system_message "You are a helpful NER assistant. Follow the instruction below to annotate the asked entities from a provided text. Return the annotated entities in the requested JSON format, enclosed between ```json and ``` tags. Nothing else should be returned outside these tags."

#--with_guidelines \


# non thinking
# --temperature 0.7 \
# --top_p 0.8 \
#--system_message "You are a helpful NER assistant. Follow the instruction below to annotate the asked entities from a provided text. Engage in a thorough reasoning process: carefully scrutinize the definitions and guidelines to guide your annotation, discuss ambiguous cases, and resolve class assignments by comparing closely related candidates. Enclose your reasoning between <think> and </think> tags. Finally, return the annotated entities in the requested JSON format, enclosed between ```json and ``` tags. Nothing else should be returned outside these tags."


# thinking
# --temperature 0.6 \
# --top_p 0.9 \