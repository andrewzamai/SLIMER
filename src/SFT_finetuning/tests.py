from transformers import AutoTokenizer
from src.data_handlers import data_handler_pileNER

if __name__ == "__main__":

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-70B-Instruct")

    dataset = data_handler_pileNER.convert_MIT_CrossNER_test_sets_for_SLIMER_inference(
        'ai',
        '../../data/eval_data_UniNER/test_data/CrossNER_AI.json',
        path_to_NE_guidelines_json='../data_handlers/questions/crossNER/gpt_guidelines/ai_NE_definitions.json',
        with_definition=True
    )
    sample = dataset[9]
    print(sample)


    def format_chat_template(row):
        system_message = "You are an expert in Named Entity Recognition designed to output JSON only."
        user_provides_text_instruction = "You are given a text chunk (delimited by triple quotes) and an instruction. Read the text and answer to the instruction in the end.\n\"\"\"\n{input}\n\"\"\"\nInstruction: {instruction}"
        row_json = [{"role": "system", "content": system_message},
                    {"role": "user", "content": user_provides_text_instruction.format(input=row['input'], instruction=row['instruction'])},
                    {"role": "assistant", "content": row['output']}]
        row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
        return row


    dataset = dataset.map(
        format_chat_template,
        num_proc=4,
    )

    print(dataset[0]['text'])