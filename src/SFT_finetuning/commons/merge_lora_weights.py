import argparse
import torch
import sys

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Llama merger parser")
    parser.add_argument("base_model", type=str, help="Path to the base model")
    parser.add_argument("path_to_LORA", type=str, help="Path to the LoRA checkpoint")
    parser.add_argument("path_to_save_to", type=str, help="Path to save the merged model")
    args = parser.parse_args()

    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype='auto',
        device_map={"": "cuda"}
    )

    print("Loading LoRA weights...")
    model = PeftModel.from_pretrained(
        base_model,
        args.path_to_LORA,
        torch_dtype='auto',
        device_map={"": "cuda"}
    )

    print("Merging LoRA weights and unloading...")
    try:
        model = model.merge_and_unload()
    except Exception as e:
        print(f"Error during merge_and_unload: {e}")
        sys.exit(1)

    print("Saving merged model...")
    try:
        # model._hf_peft_config_loaded = False
        model.save_pretrained(args.path_to_save_to)
    except Exception as e:
        print(f"Error during model.save_pretrained: {e}")
        sys.exit(1)

    print("Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.path_to_LORA)
    except Exception as e:
        print(f"Tokenizer not found in LoRA directory, loading base model tokenizer. Error: {e}")
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    print("Saving tokenizer...")
    tokenizer.save_pretrained(args.path_to_save_to)

    print(f"Merged model and tokenizer saved to {args.path_to_save_to}")