"""
EVALUATE Llama-2-7b for zero-shot NER with uniNER official evaluation script

UniNER's authors provide the crossNER/MIT test datasets already converted to QA format

We use convert_official_uniNER_eval_dataset_for_inference for:
 - replacing question with definition if with_definition=True
 - format to input expected by SFT_finetuning preprocess and tokenizer function
"""

__package__ = "SFT_finetuning.evaluating"

import sys
import os

import uniNER_official_eval_script

# my libraries
from MSEQA_4_NER.data_handlers import data_handler_pileNER

from ..commons.initialization import init_model
from ..commons.generation import evaluate, batch_evaluate
from ..commons.prompter import Prompter


def load_or_build_dataset_GenQA_format(datasets_cluster_name, subdataset_name, data_handler, with_definition):
    path_to_eval_dataset_uniNER = f"./datasets/eval_data_UniNER/CrossNER_{subdataset_name}.json" if datasets_cluster_name == 'crossNER' else f"./datasets/eval_data_UniNER/mit-{subdataset_name}.json"
    path_to_guidelines_folder = f"./datasets/questions/{datasets_cluster_name}/gpt_guidelines"

    path_to_subdataset_guidelines = None
    if with_definition:
        path_to_subdataset_guidelines = os.path.join(path_to_guidelines_folder, subdataset_name + '_NE_definitions.json')

    print("Loading train/validation/test Datasets in MS-EQA format...")
    print(" ...converting uniNER Datasets in GEnQA format for inference")
    sys.stdout.flush()

    dataset_MSEQA_format = data_handler.convert_official_uniNER_eval_dataset_for_GenQA(subdataset_name, path_to_eval_dataset_uniNER, with_definition, path_to_subdataset_guidelines)

    return dataset_MSEQA_format


if __name__ == '__main__':

    print("CrossNER/MIT ZERO-SHOT EVALUATIONS with UniNER official eval script:\n")

    to_eval_on = [
        # converting from uniNER dataset using function inside data_handler_pileNER
        {'datasets_cluster_name': 'MIT', 'data_handler': data_handler_pileNER, 'subdataset_names': ['movie', 'restaurant']},
        {'datasets_cluster_name': 'crossNER', 'data_handler': data_handler_pileNER, 'subdataset_names': ['ai', 'literature', 'music', 'politics', 'science']},
    ]

    WITH_DEFINITION = True
    print(f"With definition: {WITH_DEFINITION}")

    base_model = 'meta-llama/Llama-2-7b-chat-hf'
    path_to_LORA_adapters = './trained_models/llama2_4_NER_noQuant'
    # TODO: load from configs parameters
    cutoff_len = 768

    with open('./.env', 'r') as file:
        api_keys = file.readlines()
    api_keys_dict = {}
    for api_key in api_keys:
        api_name, api_value = api_key.split('=')
        api_keys_dict[api_name] = api_value
    # print(api_keys_dict)

    tokenizer, model = init_model(
        base_model,
        load_8bit=False,
        load_4bit=False,
        cutoff_len=cutoff_len,
        device_map="auto",
        use_flash_attention=False,
        lora_weights=path_to_LORA_adapters,
        padding_side="left"  # requires to pad left for batch_generation !
    )

    model = model.to(device='cuda')

    print(tokenizer.padding_side)

    prompter = Prompter('reverse_INST', template_path='./SFT_finetuning/templates', eos_text='')

    for data in to_eval_on:
        for subdataset_name in data['subdataset_names']:
            print(f"\n\nEvaluating MS-EQA model named '{base_model.split('/')[-1]}' on '{subdataset_name}' test fold in ZERO-SHOT setting\n")

            dataset_MSEQA_format = load_or_build_dataset_GenQA_format(data['datasets_cluster_name'], subdataset_name, data['data_handler'], WITH_DEFINITION)

            EVAL_BATCH_SIZE = 4 if data['datasets_cluster_name'] == 'MIT' else 2
            print("BATCH_SIZE for evaluation: {}".format(EVAL_BATCH_SIZE))
            sys.stdout.flush()

            """
            all_gold_answers = []
            all_pred_answers = []
            for i, sample in enumerate(dataset_MSEQA_format):
                response, boh = evaluate(
                    tokenizer=tokenizer,
                    model=model,
                    prompter=prompter,
                    instruction=sample['instruction'],
                    input=sample['input'],
                    cutoff_len=768
                )

                if "</s>" in response:
                    response = response[:-len("</s>")]

                all_pred_answers.append(response)
                all_gold_answers.append(sample['output'])
            """

            all_gold_answers = []
            all_pred_answers = []
            for i in range(0, len(dataset_MSEQA_format), EVAL_BATCH_SIZE):
                instructions = dataset_MSEQA_format['instruction'][i:i+EVAL_BATCH_SIZE]
                inputs = dataset_MSEQA_format['input'][i:i+EVAL_BATCH_SIZE]

                all_gold_answers.extend(dataset_MSEQA_format['output'][i:i+EVAL_BATCH_SIZE])

                batch_pred_answers, boh = batch_evaluate(
                    tokenizer=tokenizer,
                    model=model,
                    prompter=prompter,
                    instructions=instructions,
                    inputs=inputs,
                    cutoff_len=768,
                    verbose=False
                )

                #print(type(batch_pred_answers))
                #for pred in batch_pred_answers:
                #print(pred)
                #sys.stdout.flush()
                all_pred_answers.extend(batch_pred_answers)

            print("gold_answers")
            print(all_gold_answers[0:10])
            print("pred_answers")
            print(all_pred_answers[0:10])
            eval_result = uniNER_official_eval_script.NEREvaluator().evaluate(all_pred_answers, all_gold_answers)
            print(f"\n{subdataset_name}")
            print(f'Precision: {eval_result["precision"]}, Recall: {eval_result["recall"]}, F1: {eval_result["f1"]}')
            print("\n ------------------------------------ ")


    print("\nDONE :)")
    sys.stdout.flush()
