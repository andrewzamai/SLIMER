import argparse
import logging
import os
import sys
import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import datasets
import numpy as np
from datasets import load_dataset

import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, is_offline_mode, send_example_telemetry
from transformers.utils.versions import require_version

from gner_trainer import GNERTrainer
from gner_collator import DataCollatorForGNER
from slim_gner_evaluator import compute_metrics

from src.SFT_finetuning.commons.initialization import init_model, wrap_model_for_peft, get_HF_access_token

# noinspection PyUnresolvedReferences
from vllm import LLM, SamplingParams
from datasets import Dataset

# off wandb
os.environ['WANDB_DISABLED'] = "True"
logger = logging.getLogger(__name__)
CURRENT_DIR = os.path.dirname(__file__)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_dir: str = field(
        default=None, metadata={"help": "The directory for saving the train/dev/test splits and labels."}
    )
    no_load_gner_customized_datasets: bool = field(
        default=False, metadata={"help": "Whether to load GNER datasets. If False, you should provide json files"}
    )
    train_json_dir: str = field(
        default=None, metadata={"help": "The directory for saving the train data."}
    )
    valid_json_dir: str = field(
        default=None, metadata={"help": "The directory for saving the valid data."}
    )
    test_json_dir: str = field(
        default=None, metadata={"help": "The directory for saving the test data."}
    )
    data_config_dir: str = field(
        default=None, metadata={"help": "The json file for config training and testing tasks"}
    )
    instruction_file: str = field(
        default=None, metadata={"help": "The instruction file for different tasks."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=648,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_target_length: Optional[int] = field(
        default=648,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``."
            )
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    add_dataset_name: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to preappend dataset name before the task input."}
    )


def main():
    # Argument Parsing
    parser = argparse.ArgumentParser(description='Evaluate SLIM-GNER Zero-Shot NER performance on MIT-CrossNER')
    parser.add_argument('--merged_model_name', type=str, required=True, help='path_to_merged_model')
    parser.add_argument('--json_test_file', type=str, required=True, help='path_to_json_test_file')
    parser.add_argument('--max_source_length', type=int, required=True, help='max_source_length')
    parser.add_argument('--max_new_tokens', type=int, required=True, help='max_generation_new_tokens')
    parser.add_argument('--temperature', type=float, required=True, help='generation_temperature')
    parser.add_argument('--stop_token', type=str, required=True, help='stop_token')  # Changed to str

    args = parser.parse_args()

    # Example Output
    print(f"Evaluating model {args.merged_model_name} with the following settings:")
    print(f"Test file: {args.json_test_file}")
    print(f"Max source length: {args.max_source_length}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"Stop token: {args.stop_token}")

    vllm_model = LLM(model=args.merged_model_name)
    sampling_params = SamplingParams(temperature=args.temperature, max_tokens=args.max_new_tokens, stop=[args.stop_token])

    test_set = load_dataset("json", data_files=args.json_test_file, split='train')
    test_set = test_set.to_list()

    tokenizer = vllm_model.get_tokenizer()
    inputs = []
    for sample_GNER in test_set:
        system_message = "You are an expert in Named Entity Recognition."
        conversation = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": sample_GNER['instance']['instruction_inputs']}  # the input_text + instruction
        ]

        model_inputs = tokenizer.apply_chat_template(
            conversation=conversation,
            tokenize=False,
            truncation=True,
            padding=False,
            max_length=args.max_source_length,
            add_generation_prompt=True,  # start the assistant response for continuation
            return_tensors=None,
            return_dict=False
        )

        # For LLaMA Model, instruction part are wrapped with [INST] tag
        # input_texts = f"[INST] {sample_GNER['instance']['instruction_inputs']} [/INST]"
        inputs.append(model_inputs)

    responses = vllm_model.generate(inputs, sampling_params)
    for i, response in enumerate(responses):
        response = response.outputs[0].text
        # response = response[response.find("[/INST]") + len("[/INST]"):].strip()
        # print(response)
        test_set[i]['prediction'] = response
        if i < 10:
            print(test_set[i])
            sys.stdout.flush()

    test_set = Dataset.from_list(test_set)

    path_to_save_to = f"./model_predictions/{args.merged_model_name.split('/')[-1]}/{args.json_test_file.split('/')[-1]}"
    test_set.to_json(path_to_save_to)

    from slim_gner_evaluator import SLIMGNEREvaluator
    # load tokenizer and prediction data
    # tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    all_examples = defaultdict(list)
    with open(path_to_save_to, 'r') as fh:
        for line in fh.readlines():
            line_data = json.loads(line)
            all_examples[line_data['dataset']].append(line_data)

    # evaluate
    tot_f1, tot_dataset = 0, 0
    for dataset in all_examples:
        eval_result = SLIMGNEREvaluator().evaluate(all_examples[dataset])
        print(
            f'\nDataset: {dataset}, F1: {eval_result["f1"]}, Precision: {eval_result["precision"]}, Recall: {eval_result["recall"]}')
        tot_f1 += eval_result["f1"]
        tot_dataset += 1
    print(f'avg_f1: {tot_f1 / tot_dataset}')


if __name__ == "__main__":
    main()