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


def load_or_build_dataset_SLIMER_format(datasets_cluster_name, subdataset_name, data_handler, with_definition, max_tagNames_per_prompt):
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
        return data_handler.convert_MIT_CrossNER_test_sets_for_SLIMER_PARALLEL_inference(subdataset_name, path_to_eval_dataset_uniNER, with_definition, path_to_subdataset_guidelines, max_tagNames_per_prompt)

    elif datasets_cluster_name == 'MIT':
        path_to_eval_dataset_uniNER = f"./data/eval_data_UniNER/test_data/mit-{subdataset_name}.json"
        path_to_guidelines_folder = f"./src/data_handlers/questions/{datasets_cluster_name}/gpt_guidelines"
        path_to_subdataset_guidelines = os.path.join(path_to_guidelines_folder, subdataset_name + '_NE_definitions.json')
        return data_handler.convert_MIT_CrossNER_test_sets_for_SLIMER_PARALLEL_inference(subdataset_name, path_to_eval_dataset_uniNER, with_definition, path_to_subdataset_guidelines, max_tagNames_per_prompt)

    elif datasets_cluster_name == 'BUSTER':
        pass
    else:
        raise ValueError(f"{datasets_cluster_name} not in [crossNER, MIT, BUSTER] options")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='''Evaluate SLIMER-PARALLEL Zero-Shot NER performance''')
    parser.add_argument('merged_model_name', type=str, help='path_to_merged_model')
    parser.add_argument('max_tagNames_per_prompt', type=int, help='max_tagNames_per_prompt')
    parser.add_argument('--with_guidelines', action='store_true', help='Whether to use Def & Guidelines')
    args = parser.parse_args()

    print("\nCrossNER/MIT/BUSTER ZERO-SHOT NER EVALUATIONS with UniNER official eval script:\n")

    to_eval_on = [
        # converting from uniNER eval datasets using function inside data_handler_pileNER
        {'datasets_cluster_name': 'crossNER', 'data_handler': data_handler_pileNER, 'subdataset_names': ['ai', 'literature', 'music', 'politics', 'science']},
        #{'datasets_cluster_name': 'MIT', 'data_handler': data_handler_pileNER, 'subdataset_names': ['movie', 'restaurant']},
        #{'datasets_cluster_name': 'BUSTER', 'data_handler': data_handler_BUSTER, 'subdataset_names': ['BUSTER']},
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

    input_instruction_prompter = Prompter('LLaMA3-chat-NOheaders', template_path='./src/SFT_finetuning/templates')

    for data in to_eval_on:

        for subdataset_name in data['subdataset_names']:

            print(f"\n\nEvaluating model on '{subdataset_name}' test fold...\n")

            # 1) Load dataset and convert it to SLIMER-PARALLEL format
            dataset_SLIMER_PARALLEL_format = load_or_build_dataset_SLIMER_format(data['datasets_cluster_name'], subdataset_name, data['data_handler'], args.with_guidelines, args.max_tagNames_per_prompt)
            print(dataset_SLIMER_PARALLEL_format)
            print(dataset_SLIMER_PARALLEL_format[0])
            sys.stdout.flush()

            """
            # 2) for each tagName save the indices of the associated samples
            preds_per_tagName = defaultdict(list)

            # 3) input, instructions and gold answers
            gold_answers_per_tagName = defaultdict(list)
            for sample in dataset_SLIMER_PARALLEL_format:
                output = json.loads(dataset_SLIMER_PARALLEL_format['output'])
                for tagName, ga in output.items():
                    gold_answers_per_tagName[output].append(ga)
            """

            def format_chat_template(row):
                system_message = "You are a helpful NER assistant designed to output JSON."
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
