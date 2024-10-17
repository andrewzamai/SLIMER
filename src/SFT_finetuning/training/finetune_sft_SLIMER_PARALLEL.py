""" Supervised Fine-tuning Trainer (SFT) for generative LLMs """

import os
import sys
import shutil

import yaml
import argparse
from typing import List

import torch
import transformers
from datasets import load_dataset
from peft import set_peft_model_state_dict
from transformers import EarlyStoppingCallback

from src.SFT_finetuning.commons.initialization import init_model, wrap_model_for_peft, get_HF_access_token
from src.SFT_finetuning.commons.preprocessing import generate_and_tokenize_prompt
from src.SFT_finetuning.commons.prompter import Prompter


def train(
        # model/data params
        base_model: str = "",  # the only required argument
        data_path: str = "",
        val_data_path: str = "",
        output_dir: str = "./lora-alpaca",
        task_instruction: str = '',
        text_col: str = '',
        output_col: str = '',
        dataset_name: str = '',
        # training hyperparams
        batch_size: int = 128,
        micro_batch_size: int = 4,
        num_epochs: int = 3,
        learning_rate: float = 3e-4,
        cutoff_len: int = 4096,
        select_train_portion: int = -1,
        val_set_size: int = 2000,
        # lora hyperparams
        use_lora: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: List[str] = ("q_proj", "v_proj"),
        # llm hyperparams
        train_on_inputs: bool = False,  # if False, masks out inputs in loss
        group_by_length: bool = False,  # faster, but produces an odd training loss curve
        # wandb params
        wandb_project: str = "",
        wandb_run_name: str = "",
        wandb_watch: str = "",  # options: false | gradients | all
        wandb_log_model: str = "",  # options: false | true
        resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
        prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
        eos_text: str = None,
        load_8bit: bool = False,
        load_4bit: bool = False,
        save_total_limit: int = 5,
        warmup_steps: int = 100,
        eval_steps: int = 100,
        logging_steps: int = 10,
        max_grad_norm: float = 0.3,
        use_flash_attention: bool = False,
        shuffle: bool = True,
        gradient_checkpointing: bool = False,
        early_stopping_patience=5,
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"LoRA SFT training with params:\n"
            f"base_model: {base_model}\n"
            f"task instruction: {task_instruction}\n"
            f"text col: {text_col}\n"
            f"output col: {output_col}\n"
            f"load_8bit: {load_8bit}\n"
            f"load_4bit: {load_4bit}\n"
            f"dataset_name: {dataset_name}\n"
            f"data_path: {data_path}\n"
            f"val_data_path: {val_data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"early_stopping_patience: {early_stopping_patience}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"select_train_portion: {select_train_portion}\n"
            f"use_lora: {use_lora}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
            f"eos_text: {eos_text}\n"
            f"use flash attn: {use_flash_attention}\n"
            f"shuffle: {shuffle}\n"
            f"gradient checkpointing: {gradient_checkpointing}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='mosaicml/mpt-7b-instruct'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    prompter = Prompter(prompt_template_name, template_path='./src/SFT_finetuning/templates', eos_text=eos_text)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    print(f"World size: {world_size}")
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
            "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    tokenizer, model = init_model(
        base_model,
        load_8bit=load_8bit,
        load_4bit=load_4bit,
        cutoff_len=cutoff_len,
        device_map=device_map,
        use_flash_attention=use_flash_attention
    )

    if use_lora:
        model = wrap_model_for_peft(
            model,
            load_8bit=load_8bit,
            load_4bit=load_4bit,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_target_modules=lora_target_modules,
            lora_dropout=lora_dropout
        )
        model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    """ Loading train/validation datasets """

    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path, dataset_name) if dataset_name else load_dataset(data_path)

    if task_instruction:
        def add_instruction(example):
            example["instruction"] = task_instruction
            return example
        data = data.map(lambda x: add_instruction(x), desc="Adding instruction to examples")

    if text_col:
        data = data.rename_column(text_col, "input")

    if output_col:
        data = data.rename_column(output_col, "output")

    if resume_from_checkpoint:
        raise ValueError("resume_from_checkpoint not implemented")

    train_data = data["train"]
    print(train_data)
    print(train_data['instruction'][0])
    print(train_data['output'][0])

    if shuffle:
        # shuffle train_data
        train_data = train_data.shuffle(seed=42)

    if val_set_size > 0 and not val_data_path:
        train_val = train_data.train_test_split(test_size=val_set_size, shuffle=False)
        train_data = (
            train_val["train"].map(lambda x: generate_and_tokenize_prompt(x, tokenizer, prompter, cutoff_len, train_on_inputs), num_proc=30)
        )

        train_data = train_data.filter(lambda x: len(x["input_ids"]) > 5)

        val_data = (
            train_val["test"].map(lambda x: generate_and_tokenize_prompt(x, tokenizer, prompter, cutoff_len, train_on_inputs), num_proc=30)
        )

        val_data = val_data.filter(lambda x: len(x["input_ids"]) > 5)

    elif val_data_path:
        train_data = data["train"].map(lambda x: generate_and_tokenize_prompt(x, tokenizer, prompter, cutoff_len, train_on_inputs), num_proc=30)
        val_data = load_dataset("json", data_files=val_data_path)
        # if -1 use all validation data
        if val_set_size == -1:
            val_set_size = len(val_data["train"])
        val_data = val_data["train"].select(list(range(min(val_set_size, len(val_data["train"])))))  # no more than val_set_size 5k examples

        print("Validation statistics: ")
        print(val_data)

        val_data = val_data.map(lambda x: generate_and_tokenize_prompt(x, tokenizer, prompter, cutoff_len, train_on_inputs), num_proc=30)
        val_set_size = len(val_data)
    else:
        train_data = data["train"].map(lambda x: generate_and_tokenize_prompt(x, tokenizer, prompter, cutoff_len, train_on_inputs), num_proc=30)
        if "validation" in data:
            print("Validation is in the dataset")
            val_data = data["validation"]
            val_set_size = len(val_data)
        else:
            val_data = None

    if select_train_portion > 0:
        train_data = train_data.select(list(range(min(select_train_portion, len(train_data)))))

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    if not train_on_inputs:
        data_collator = transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        )
    else:
        data_collator = transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)

    training_arguments = transformers.TrainingArguments(
        per_device_train_batch_size=micro_batch_size,
        per_device_eval_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        lr_scheduler_type='cosine',
        bf16=True,
        logging_steps=logging_steps,
        optim="adamw_torch",
        evaluation_strategy="steps" if val_set_size > 0 or val_data_path else "no",
        save_strategy="steps",
        eval_steps=eval_steps if val_set_size > 0 or val_data_path else None,
        save_steps=eval_steps,
        max_grad_norm=max_grad_norm,
        output_dir=output_dir,
        save_total_limit=save_total_limit,
        load_best_model_at_end=True if val_set_size > 0 or val_data_path else False,
        ddp_find_unused_parameters=False if ddp else None,
        group_by_length=group_by_length,
        report_to="wandb" if use_wandb else None,
        run_name=wandb_run_name if use_wandb else None,
        gradient_checkpointing=gradient_checkpointing,
        dataloader_num_workers=world_size
    )

    print(train_data)
    print("====================")
    print(val_data)
    print("\n\n")
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=training_arguments,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)] if early_stopping_patience > -1 else []
    )
    model.config.use_cache = False

    # if torch.__version__ >= "2" and sys.platform != "win32":
    #     model = torch.compile(model)

    if resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    else:
        trainer.train()

    # tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)
    print(f'Saving model at: {output_dir}')
    print("\n If there's a warning about missing keys above, please disregard :)")


if __name__ == "__main__":

    from src.data_handlers import data_handler_pileNER
    from src.SFT_finetuning.commons.basic_utils import load_json

    top_391_NEs_list = load_json("./src/data_handlers/questions/pileNER/top391NEs_definitions.json")
    top_391_NEs_list = list(top_391_NEs_list.keys())

    max_tagNames_per_prompt = 5
    datasetDict_SLIMER_PARALLEL_format = data_handler_pileNER.build_dataset_SLIMER_PARALLEL_format(
        top_391_NEs_list,
        max_tagNames_per_prompt=max_tagNames_per_prompt,
        path_to_DeG="./src/data_handlers/questions/pileNER/top391NEs_definitions.json"
    )

    dataset_name = f"SLIMER_PARALLEL_{max_tagNames_per_prompt}tagNamesPerPrompt_wDeG"

    datasetDict_SLIMER_PARALLEL_format['train'].to_json(f'./data/pileNER/{dataset_name}/train.jsonl')
    datasetDict_SLIMER_PARALLEL_format['validation'].to_json(f'./data/pileNER/{dataset_name}/validation.jsonl')

    # now loading training config from yml and overriding some variables like dataset name and output_dir
    path_to_training_config = './src/SFT_finetuning/training_config/llama2_4_NER_XDef_NsamplesPerNE.yml'
    with open(path_to_training_config, 'rb') as f:
        configs = yaml.safe_load(f.read())
    configs['data_path'] = f'./data/pileNER/{dataset_name}/train.jsonl'
    configs['val_data_path'] = f'./data/pileNER/{dataset_name}/validation.jsonl'
    configs['output_dir'] = f"./trained_models/SLIMER-PARALLEL-LLaMA3-{max_tagNames_per_prompt}tagNamesPerPrompt_wDeG"

    train(**configs)

    # copy training config file
    shutil.copy(src=path_to_training_config, dst=os.path.join(configs['output_dir'], 'training_configs.yml'))

    torch.cuda.empty_cache()

    print("Training DONE :)\n\n")
