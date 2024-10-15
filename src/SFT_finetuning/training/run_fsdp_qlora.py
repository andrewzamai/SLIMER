import logging
from dataclasses import dataclass, field
import os
import random
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments
from trl.commands.cli_utils import TrlParser
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,

)
from trl import setup_chat_format
from peft import LoraConfig

from trl import SFTTrainer


# 1) accelerate config and follow https://huggingface.co/blog/ram-efficient-pytorch-fsdp
# 2) accelerate launch ./src/SFT_finetuning/training/run_fsdp_qlora.py --config ./src/SFT_finetuning/training_config/LLaMA3.yaml

@dataclass
class ScriptArguments:
    dataset_path: str = field(
        default=None,
        metadata={
            "help": "Path to the dataset"
        },
    )
    model_id: str = field(
        default=None, metadata={"help": "Model ID to use for SFT training"}
    )
    max_seq_length: int = field(
        default=512, metadata={"help": "The maximum sequence length for SFT Trainer"}
    )


def training_function(script_args, training_args):

    from src.data_handlers import data_handler_pileNER

    with training_args.main_process_first(
            desc="Log a few random samples from the processed training set"
    ):
        if not os.path.exists(script_args.dataset_path):
            dataset_MSEQA_format_with_n_samples_per_NE_FalseDef = data_handler_pileNER.build_dataset_MSEQA_format_with_n_samples_per_NE_pos_neg(
                n_pos_samples_per_NE=10,
                n_neg_samples_per_NE=10,
                removeTestDatasetsNEs=True,
                keep_only_top_tagNames=391
            )

            data_handler_pileNER.convert_MSEQA_dataset_to_GenQA_format_SI(
                dataset_MSEQA_format=dataset_MSEQA_format_with_n_samples_per_NE_FalseDef,
                with_definition=True,
                path_to_NE_guidelines_json="./src/data_handlers/questions/pileNER/top391NEs_definitions.json",
                path_to_save_to=f'./data/pileNER/{script_args.dataset_path}'
            )

    train_dataset = load_dataset(
        "json",
        data_files=os.path.join(script_args.dataset_path, "train.jsonl"),
        split="train",
    )
    test_dataset = load_dataset(
        "json",
        data_files=os.path.join(script_args.dataset_path, "validation.jsonl"),
        split="train",
    )

    # Tokenizer        
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_id, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    #tokenizer.chat_template = LLAMA_3_CHAT_TEMPLATE

    def format_chat_template(row):
        system_message = "You are an expert in Named Entity Recognition designed to output JSON only."
        user_provides_text_instruction = "You are given a text chunk (delimited by triple quotes) and an instruction. Read the text and answer to the instruction in the end.\n\"\"\"\n{input}\n\"\"\"\nInstruction: {instruction}"
        row_json = [{"role": "system", "content": system_message},
                    {"role": "user", "content": user_provides_text_instruction.format(input=row['input'], instruction=row['instruction'])},
                    {"role": "assistant", "content": row['output']}]
        row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
        return row

    train_dataset = train_dataset.map(format_chat_template, remove_columns=["input", "instruction", "output"])
    test_dataset = test_dataset.map(format_chat_template, remove_columns=["input", "instruction", "output"])
    
    # print random sample
    with training_args.main_process_first(
        desc="Log a few random samples from the processed training set"
    ):
        for index in random.sample(range(len(train_dataset)), 2):
            print(train_dataset[index]["text"])

    # Model    
    torch_dtype = torch.bfloat16
    quant_storage_dtype = torch.bfloat16

    quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_storage=quant_storage_dtype,
        )

    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_id,
        quantization_config=quantization_config,
        attn_implementation="sdpa",  # use sdpa, alternatively use "flash_attention_2"
        torch_dtype=quant_storage_dtype,
        use_cache=True  # this is needed for gradient checkpointing
    )
    
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # LoRA config based on QLoRA paper & Sebastian Raschka experiment
    peft_config = LoraConfig(
        lora_alpha=8,
        lora_dropout=0.05,
        r=16,
        bias="none",
        target_modules=["q_proj", "v_proj", "k_proj"],  #  "all-linear",
        task_type="CAUSAL_LM",
        # modules_to_save=["lm_head", "embed_tokens"]  # add if you want to use the Llama 3 instruct template
    )

    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        dataset_text_field="text",
        eval_dataset=test_dataset,
        peft_config=peft_config,
        max_seq_length=script_args.max_seq_length,
        tokenizer=tokenizer,
        packing=True,  # True
        dataset_kwargs={
            "add_special_tokens": False,  # We template with special tokens
            "append_concat_token": False,  # No need to add additional separator token
        },
    )
    if trainer.accelerator.is_main_process:
        trainer.model.print_trainable_parameters()

    ##########################
    # Train model
    ##########################
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    trainer.train(resume_from_checkpoint=checkpoint)

    ##########################
    # SAVE MODEL FOR SAGEMAKER
    ##########################
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model()


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, TrainingArguments))
    script_args, training_args = parser.parse_args_and_config()    
    
    # set use reentrant to False
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}
    # set seed
    set_seed(training_args.seed)
  
    # launch training
    training_function(script_args, training_args)
