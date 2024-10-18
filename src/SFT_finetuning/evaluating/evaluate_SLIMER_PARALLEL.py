"""
Evaluate SLIMER-PARALLEL model for zero-shot NER on CrossNER/MIT/BUSTER datasets

- Using provided uniNER official evaluation script

- Using vLLM library for faster inference

UniNER's authors provide the crossNER/MIT test datasets already converted to QA format
./data/eval_data_UniNER/test_data/CrossNER_ai.json

Importantly these provided datasets exclude MISCELLANEOUS class
"""

# noinspection PyUnresolvedReferences
from vllm import LLM, SamplingParams

from datasets import Dataset, DatasetDict, load_dataset
from collections import defaultdict
import numpy as np
import argparse
import json
import sys
import os
import re

# copy of uniNER official eval script from their github
import uniNER_official_eval_script

# my libraries
from src.data_handlers import data_handler_pileNER, data_handler_BUSTER
from src.SFT_finetuning.commons.preprocessing import truncate_input
from src.SFT_finetuning.commons.prompter import Prompter
from src.SFT_finetuning.evaluating.eval_utils import chunk_document_with_sliding_window, aggregate_preds_from_chunks

import json


def parse_json_pred(sample, response):
    try:
        parsed_response = json.loads(response)
    except json.JSONDecodeError:
        parsed_response = {}
    try:
        parsed_gold_output = json.loads(sample['output'])
    except json.JSONDecodeError:
        parsed_gold_output = {}

    # check for hallucinated types (unexpected keys)
    expected_keys = set(parsed_gold_output.keys())
    keys_in_response = set(parsed_response.keys())

    # identify and remove unexpected (hallucinated) keys
    unexpected_keys = keys_in_response - expected_keys
    for key in unexpected_keys:
        parsed_response.pop(key)

    # check for missing keys or not parsable
    for key in expected_keys:
        # if missing set it to []
        value = parsed_response.get(key, [])
        if not isinstance(value, list):
            parsed_response[key] = []
        else:
            value = [x for x in value if isinstance(x, str)]
            parsed_response[key] = value

    return parsed_gold_output, parsed_response

def load_or_build_dataset_SLIMER_format(datasets_cluster_name, subdataset_name, data_handler, with_definition,
                                        max_tagNames_per_prompt):
    """
    universal-ner github provides the crossNER and MIT NER-datasets already in a conversation-QA format (eval_dataset_uniNER folder);
    here we convert the dataset to our usual features and replace "instruction" with the NE D&G if with_definition=True
    """
    print(f"\nConverting {subdataset_name} to SLIMER-PARALLEL format for inference...\n")
    sys.stdout.flush()

    if datasets_cluster_name == 'crossNER':
        path_to_eval_dataset_uniNER = f"./data/eval_data_UniNER/test_data/CrossNER_{subdataset_name}.json"
        path_to_guidelines_folder = f"./src/data_handlers/questions/{datasets_cluster_name}/gpt_guidelines"
        path_to_subdataset_guidelines = os.path.join(path_to_guidelines_folder, subdataset_name + '_NE_definitions.json')
        return data_handler.convert_MIT_CrossNER_test_sets_for_SLIMER_PARALLEL_inference(subdataset_name,
                                                                                         path_to_eval_dataset_uniNER,
                                                                                         with_definition,
                                                                                         path_to_subdataset_guidelines,
                                                                                         max_tagNames_per_prompt)

    elif datasets_cluster_name == 'MIT':
        path_to_eval_dataset_uniNER = f"./data/eval_data_UniNER/test_data/mit-{subdataset_name}.json"
        path_to_guidelines_folder = f"./src/data_handlers/questions/{datasets_cluster_name}/gpt_guidelines"
        path_to_subdataset_guidelines = os.path.join(path_to_guidelines_folder,
                                                     subdataset_name + '_NE_definitions.json')
        return data_handler.convert_MIT_CrossNER_test_sets_for_SLIMER_PARALLEL_inference(subdataset_name,
                                                                                         path_to_eval_dataset_uniNER,
                                                                                         with_definition,
                                                                                         path_to_subdataset_guidelines,
                                                                                         max_tagNames_per_prompt)

    elif datasets_cluster_name == 'BUSTER':
        pass
    else:
        raise ValueError(f"{datasets_cluster_name} not in [crossNER, MIT, BUSTER] options")


if __name__ == '__main__':
    # CUDA_VISIBLE_DEVICES=2 python src/SFT_finetuning/evaluating/evaluate_SLIMER_PARALLEL.py meta-llama/Llama-3.1-8B-Instruct 5 --with_guidelines
    parser = argparse.ArgumentParser(description='''Evaluate SLIMER-PARALLEL Zero-Shot NER performance''')
    parser.add_argument('merged_model_name', type=str, help='path_to_merged_model')
    parser.add_argument('max_tagNames_per_prompt', type=int, help='max_tagNames_per_prompt')
    parser.add_argument('--with_guidelines', action='store_true', help='Whether to use Def & Guidelines')
    args = parser.parse_args()

    print("\nCrossNER/MIT/BUSTER ZERO-SHOT NER EVALUATIONS with UniNER official eval script:\n")

    to_eval_on = [
        # converting from uniNER eval datasets using function inside data_handler_pileNER
        {'datasets_cluster_name': 'crossNER', 'data_handler': data_handler_pileNER, 'subdataset_names': ['ai', 'literature', 'music', 'politics', 'science']},
        {'datasets_cluster_name': 'MIT', 'data_handler': data_handler_pileNER, 'subdataset_names': ['movie', 'restaurant']},
        # {'datasets_cluster_name': 'BUSTER', 'data_handler': data_handler_BUSTER, 'subdataset_names': ['BUSTER']},
    ]

    print(f"\nLLM model: {args.merged_model_name}")

    print(f"\nWith Definition & Guidelines: {args.with_guidelines}")

    partial_evaluate = False
    print(f"\npartial_evaluate: {partial_evaluate}")

    cutoff_len = 2048
    print(f"\ninput_cutoff_len: {cutoff_len}")

    max_new_tokens = 2048
    print(f"\nmax_new_tokens: {max_new_tokens}\n")

    vllm_model = LLM(model=args.merged_model_name)
    tokenizer = vllm_model.get_tokenizer()

    sampling_params = SamplingParams(temperature=0, max_tokens=max_new_tokens, stop=tokenizer.eos_token)
    print(sampling_params)

    # prompter to prefix input to the instruction
    input_instruction_prompter = Prompter('LLaMA3-chat-NOheaders', template_path='./src/SFT_finetuning/templates')

    for data in to_eval_on:

        for subdataset_name in data['subdataset_names']:

            print(f"\n\nEvaluating model on '{subdataset_name}' test fold...\n")

            # 1) Load dataset and convert it to SLIMER-PARALLEL format
            dataset_SLIMER_PARALLEL_format = load_or_build_dataset_SLIMER_format(
                data['datasets_cluster_name'],
                subdataset_name,
                data['data_handler'],
                args.with_guidelines,
                args.max_tagNames_per_prompt
            )
            print(dataset_SLIMER_PARALLEL_format)
            print(dataset_SLIMER_PARALLEL_format[0])
            sys.stdout.flush()

            def format_chat_template(row):
                #system_message = "You are a helpful NER assistant designed to output JSON."
                system_message = "You are a helpful assistant."
                conversation = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": input_instruction_prompter.generate_prompt(input=row["input"], instruction=row["instruction"])},  # the input_text + instruction
                ]
                row["prompt"] = tokenizer.apply_chat_template(conversation, tokenize=False, truncation=True, max_length=cutoff_len, add_generation_prompt=True)
                return row

            # 5) run inference on SLIMER via vLLM
            dataset_SLIMER_PARALLEL_format = dataset_SLIMER_PARALLEL_format.map(format_chat_template)
            prompts = dataset_SLIMER_PARALLEL_format['prompt']
            print(prompts[0])

            responses = vllm_model.generate(prompts, sampling_params)

            # 7) retrieve pred answers, aggregate them from chunks back to document level
            all_pred_answers = [output.outputs[0].text.strip() for output in responses]

            print(all_pred_answers[0:10])

            all_pred_answers_per_type = defaultdict(list)
            all_gold_answers_per_type = defaultdict(list)
            for i, sample in enumerate(dataset_SLIMER_PARALLEL_format):
                pred = all_pred_answers[i]
                # parse gold and pred json, removing hallucinated types
                # and checking that each pred is a list of str
                parsed_gold_output, parsed_response = parse_json_pred(sample, pred)
                #print(parsed_gold_output)
                #print(parsed_response)
                #print("--------------------------------")

                for tagName, this_tag_preds in parsed_response.items():
                    all_pred_answers_per_type[tagName].append(this_tag_preds)

                for tagName, this_tag_golds in parsed_gold_output.items():
                    all_gold_answers_per_type[tagName].append(this_tag_golds)

            # Flatten the nested list of lists
            pred_answers_for_micro = [pred for preds in all_pred_answers_per_type.values() for pred in preds]
            gold_answers_for_micro = [gold for golds in all_gold_answers_per_type.values() for gold in golds]

            print(pred_answers_for_micro[0:5])
            print(gold_answers_for_micro[0:5])
            if partial_evaluate:
                eval_result = uniNER_official_eval_script.NEREvaluator().partial_evaluate(pred_answers_for_micro, gold_answers_for_micro)
            else:
                eval_result = uniNER_official_eval_script.NEREvaluator().evaluate(pred_answers_for_micro, gold_answers_for_micro)

            precision = round(eval_result["precision"]*100, 2)
            recall = round(eval_result["recall"]*100, 2)
            f1 = round(eval_result["f1"]*100, 2)
            print("\n{} ==> micro-Precision: {:.2f}, micro-Recall: {:.2f}, micro-F1: {:.2f}".format(subdataset_name, precision, recall, f1))

            print("\nMetrics per NE category (100%):\n")
            this_dataset_metrics = {}
            for tagName in all_gold_answers_per_type.keys():
                this_tagName_golds = all_gold_answers_per_type[tagName]
                this_tagName_preds = all_pred_answers_per_type[tagName]
                if partial_evaluate:
                    eval_result = uniNER_official_eval_script.NEREvaluator().partial_evaluate(this_tagName_preds, this_tagName_golds)
                else:
                    eval_result = uniNER_official_eval_script.NEREvaluator().evaluate(this_tagName_preds, this_tagName_golds)

                print("{} --> support: {}".format(tagName, eval_result['support']))
                print("{} --> TP: {}, FN: {}, FP: {}, TN: {}".format(tagName, eval_result['TP'], eval_result['FN'], eval_result['FP'], -1))
                precision = round(eval_result["precision"] * 100, 2)
                recall = round(eval_result["recall"] * 100, 2)
                f1 = round(eval_result["f1"] * 100, 2)
                print("{} --> Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}".format(tagName, precision, recall, f1))
                print("---------------------------------------------------------- ")
                this_dataset_metrics[tagName] = {
                    'support': eval_result['support'],
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                }

            # computing MACRO scores
            this_dataset_precisions = [this_dataset_metrics[tagName]['precision'] for tagName in this_dataset_metrics]
            this_dataset_recalls = [this_dataset_metrics[tagName]['recall'] for tagName in this_dataset_metrics]
            this_dataset_f1s = [this_dataset_metrics[tagName]['f1'] for tagName in this_dataset_metrics]
            this_dataset_supports = [this_dataset_metrics[tagName]['support'] for tagName in this_dataset_metrics]
            print(
                "\n{} ==> MACRO-Precision: {:.2f} +- {:.2f}, MACRO-Recall: {:.2f} +- {:.2f}, MACRO-F1: {:.2f} +- {:.2f}".format(
                    subdataset_name,
                    np.average(this_dataset_precisions),
                    np.std(this_dataset_precisions),
                    np.average(this_dataset_recalls),
                    np.std(this_dataset_recalls),
                    np.average(this_dataset_f1s),
                    np.std(this_dataset_f1s))
            )

            # computing WEIGHTED scores
            this_dataset_supports_sum = sum(this_dataset_supports)
            this_dataset_precisions_weighted = [this_dataset_metrics[tagName]['precision'] * (
                        this_dataset_metrics[tagName]['support'] / this_dataset_supports_sum) for tagName in
                                                this_dataset_metrics]
            this_dataset_recalls_weighted = [this_dataset_metrics[tagName]['recall'] * (
                        this_dataset_metrics[tagName]['support'] / this_dataset_supports_sum) for tagName in
                                             this_dataset_metrics]
            this_dataset_f1s_weighted = [this_dataset_metrics[tagName]['f1'] * (
                        this_dataset_metrics[tagName]['support'] / this_dataset_supports_sum) for tagName in
                                         this_dataset_metrics]
            print("\n{} ==> Weighted-Precision: {:.2f}, Weighted-Recall: {:.2f}, Weighted-F1: {:.2f}".format(
                subdataset_name,
                np.sum(this_dataset_precisions_weighted),
                np.sum(this_dataset_recalls_weighted),
                np.sum(this_dataset_f1s_weighted))
            )

            # Finally, save predictions
            preds_to_save = []
            for i, sample in enumerate(dataset_SLIMER_PARALLEL_format):
                preds_to_save.append({
                    'input': sample['input'],
                    'gold_answers': sample['output'],
                    'pred_answers': all_pred_answers[i]
                })

            path_to_save_predictions = os.path.join("./predictions", args.merged_model_name.split('/')[-1])
            if not os.path.exists(path_to_save_predictions):
                os.makedirs(path_to_save_predictions)
            with open(os.path.join(path_to_save_predictions, subdataset_name + '.json'), 'w', encoding='utf-8') as f:
                json.dump(preds_to_save, f, ensure_ascii=False, indent=2)
            print("\n")

    print("\nDONE :)")
    sys.stdout.flush()
