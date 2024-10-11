MODEL_NAME_OR_PATH=meta-llama/Llama-2-7b-chat-hf
JSON_TEST_FILE=data/zero-shot-test-wDeG.json

python src/run_prediction_vllm.py \
--merged_model_name $MODEL_NAME_OR_PATH \
--json_test_file $JSON_TEST_FILE \
--max_source_length 2048 \
--max_new_tokens 2048 \
--temperature 0 \
--stop_token "<|eot_id|>"

