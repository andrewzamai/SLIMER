MODEL_NAME_OR_PATH=merged_models/GNER-391xALL-wDeG-LLaMA3-8B-Instruct
JSON_TEST_FILE=data/zero-shot-test-wDeG-5NEperPrompt.jsonl

python src/run_predictions_vllm.py \
--merged_model_name $MODEL_NAME_OR_PATH \
--json_test_file $JSON_TEST_FILE \
--max_source_length 2048 \
--max_new_tokens 2048 \
--temperature 0 \
--stop_token "<|eot_id|>"
