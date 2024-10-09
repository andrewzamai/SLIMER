from datasets import load_dataset
import os

if __name__ == '__main__':

    raw_datasets = load_dataset(
                os.path.join(os.path.dirname(__file__), "gner_dataset.py"),
                data_dir='../data/',
                instruction_file='../configs/instruction_configs/instruction.json',
                data_config_dir='../configs/dataset_configs/task_adaptation_configs',
                add_dataset_name=False,
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
