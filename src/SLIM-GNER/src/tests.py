from datasets import load_dataset
import os

if __name__ == '__main__':

    raw_datasets = load_dataset(
                os.path.join(os.path.dirname(__file__), "gner_dataset.py"),
                data_dir='../data/',
                instruction_file='../configs/instruction_configs/instruction.json',
                data_config_dir='../configs/dataset_configs/task_adaptation_configs',
                add_dataset_name=False,
                trust_remote_code=True,
                path_to_DeG="../../data_handlers/questions/pileNER/top391NEs_definitions.json"
    )
    print(raw_datasets)
    print(raw_datasets['validation'][0]['instance']['instruction_inputs'])
    #raw_datasets['test'].to_json('../../../../datasets/KIND/test_gner_format.json')
    #raw_datasets['train'].to_json('../data/pileNER-SLIMER-391x100_wD&G_train.json')
    raw_datasets['validation'].to_json('../data/pileNER-SLIMER-391x100_wD&G_validation.json')

    #data = load_dataset("json", data_files=f'../data/MultinerdIT/test_GNER_format.json')['train']
    #print(data)
    #print(data[0])

    from transformers import AutoTokenizer

    #tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

    chat_template = tokenizer.get_chat_template()
    print(chat_template)

    system_message = "You are an expert in Named Entity Recognition."
    row_json = [{"role": "system", "content": system_message},
                {"role": "user", "content": raw_datasets['validation'][0]['instance']['instruction_inputs']},
                {"role": "assistant", "content": raw_datasets['validation'][0]['instance']['prompt_labels']}]
    formatted_input = tokenizer.apply_chat_template(row_json, tokenize=False) #, add_generation_prompt=True)
    print(formatted_input)

    model_inputs = tokenizer.apply_chat_template(
        conversation=row_json[:-1],  # exclude last assistant message
        tokenize=False,
        truncation=True,
        padding=False,
        max_length=1024,
        add_generation_prompt=True,  # start the assistant response for continuation
        return_tensors=None,
        return_dict=False
    )
    print(model_inputs)
