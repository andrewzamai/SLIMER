#!/bin/bash

# Define the parameters
number_NEs=391
number_pos_samples_per_NE=5
number_neg_samples_per_NE=5
use_guidelines="--with_guidelines"

# Iterate over model_suffix values
for model_suffix in 1 2 3
do
  echo "Running for model_suffix=$model_suffix"
  
  # Compute the model name
  model_name="LLaMA3.1-8B_${number_pos_samples_per_NE}pos_${number_neg_samples_per_NE}neg_perNE_top${number_NEs}NEs_TrueDef_${model_suffix}"
  
  # Step 1: Finetuning
  echo "Starting finetuning for $model_name"
  CUDA_VISIBLE_DEVICES=2 python src/SFT_finetuning/training/finetune_sft.py \
    --number_NEs $number_NEs \
    --number_pos_samples_per_NE $number_pos_samples_per_NE \
    --number_neg_samples_per_NE $number_neg_samples_per_NE \
    $use_guidelines \
    --model_suffix $model_suffix
  
  # Step 2: Merging LoRA weights
  echo "Merging LoRA weights for $model_name"
  CUDA_VISIBLE_DEVICES=2 python src/SFT_finetuning/commons/merge_lora_weights.py \
    --number_NEs $number_NEs \
    --number_pos_samples_per_NE $number_pos_samples_per_NE \
    --number_neg_samples_per_NE $number_neg_samples_per_NE \
    $use_guidelines \
    --model_suffix $model_suffix
  
  # Step 3: Evaluation
  echo "Evaluating model $model_name"
  CUDA_VISIBLE_DEVICES=2 python src/SFT_finetuning/evaluating/evaluate_vLLM.py \
    ./merged_models/$model_name \
    LLaMA3-chat \
    $use_guidelines \
    >> ./txt_predictions/$model_name.txt
  
  echo "Completed processing for model_suffix=$model_suffix"
  echo "---------------------------------------------"
done

echo "All tasks completed!"

