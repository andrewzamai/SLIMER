import copy
import json
import os

from datasets import load_dataset, Dataset


def load_DeG_per_NEs(path_to_DeG):
    """ load json and eval to dictionary the D&G for each NE """
    if not path_to_DeG:
        raise Exception("Path to Def & Guidelines not provided")
    if not os.path.exists(path_to_DeG):
        raise ValueError(f"Can't find or read D&G at {path_to_DeG}")
    with open(path_to_DeG) as fp:
        DeG_per_NEs_raw = json.load(fp)
    # converting list to dict for fast access
    if DeG_per_NEs_raw and isinstance(DeG_per_NEs_raw, list):
        DeG_per_NEs_raw = {x['named_entity']: x for x in DeG_per_NEs_raw}
    for ne_tag, values in DeG_per_NEs_raw.items():
        gpt_definition = values['gpt_answer']
        if not gpt_definition.endswith("}"):
            if not gpt_definition.endswith("\""):
                gpt_definition += "\""
            gpt_definition += "}"

        this_ne_guidelines = eval(gpt_definition)
        # replacing ne types occurrences between single quotes to their UPPERCASE
        # ne_type_in_natural_language = values['real_name']
        # pattern = re.compile(rf'\'{re.escape(ne_type_in_natural_language)}\'', re.IGNORECASE)
        # this_ne_guidelines = {k: pattern.sub(f'{ne_type_in_natural_language.upper()}', v) for k, v in this_ne_guidelines.items()}
        values['gpt_answer'] = this_ne_guidelines

    return DeG_per_NEs_raw

def map_dataset_to_DeG_path_file(dataset_name):
    if "mit" in dataset_name:
        dataset, subdataset = dataset_name.split("-")
        return f"MIT/gpt_guidelines/{subdataset}_NE_definitions.json"
    elif "crossner" in dataset_name:
        dataset, subdataset = dataset_name.split("_")
        return f"crossNER/gpt_guidelines/{subdataset}_NE_definitions.json"
    else:
        raise ValueError(f"Unknown {dataset_name}")

def chunk_labels(lst, N):
    """Yield successive N-sized labels from lst."""
    for i in range(0, len(lst), N):
        yield lst[i:i + N]

def filter_labels(labels, label_sublist):
    """Set to O the labels which are not in label_sublist"""
    for i, label in enumerate(labels):
        if label == "O":
            # If label is "O", no changes needed
            continue
        else:
            # Split the label by "-"
            prefix, tagName = label.split("-")
            if tagName not in label_sublist:
                labels[i] = "O"
    return labels

def _generate_labeled_string(words, labels):
    label_text_list = []
    for word, label in zip(words, labels):
        label_text_list.append(f"{word}({label})")

    return " ".join(label_text_list)

def process_sample(all_datasets_DeG, general_instruction, gner_sample, labels_per_prompt=5):
    label_list = gner_sample['label_list']
    new_chunked_samples = []
    # create samples with N labels per prompt extraction
    for label_sublist in chunk_labels(label_list, labels_per_prompt):
        new_gner_sample = copy.deepcopy(gner_sample)  # Use deep copy
        new_gner_sample['label_list'] = label_sublist
        # copy gold labels but set to O the ones we are not extracting
        gold_labels = new_gner_sample['instance']['labels']
        gold_labels = filter_labels(gold_labels, label_sublist)
        new_gner_sample['instance']['labels'] = gold_labels

        instruction = general_instruction
        instruction += f"\nUse the specific entity tags: {', '.join(label_sublist)} and O.\n"
        # it the path to DeG is provided, append the Def and Guidelines for each NE
        instruction += "To help you, here are dedicated DEFINITION and GUIDELINES for each entity tag.\n"

        sampled_labels_DeG = {}

        for ne_tag in label_sublist:
            orig_tag = ne_tag
            ne_tag = ne_tag.lower()

            special_maps = {
                "average ratings": "ratings average",
                "metric": "metrics"
            }


            if ne_tag in special_maps:
                ne_tag = special_maps[ne_tag]

            sampled_labels_DeG[orig_tag] = all_datasets_DeG[gner_sample['dataset']][ne_tag]['gpt_answer']

        instruction += json.dumps(sampled_labels_DeG, indent=2)
        instruction += '\n'

        instruction += "Sentence: " + " ".join(gner_sample['instance']['words'])

        new_gner_sample['instance']['instruction_inputs'] = instruction

        new_gner_sample['instance']['prompt_labels'] = _generate_labeled_string(new_gner_sample['instance']['words'], new_gner_sample['instance']['labels'])

        new_chunked_samples.append(new_gner_sample)

    return new_chunked_samples


if __name__ == '__main__':
    test_set = load_dataset("json", data_files=f'../data/zero-shot-test.jsonl', split='train')
    print(test_set[0]['instance'])

    path_to_questions_folder = "../../data_handlers/questions"
    datasets = list(set(test_set['dataset']))
    print(datasets)
    # load all DeG per dataset
    all_datasets_DeG = {ds: {} for ds in datasets}
    for dataset_name in datasets:
        # get path to DeG for this dataset name
        subpath_to_DeG_file = map_dataset_to_DeG_path_file(dataset_name)
        this_dataset_DeG = load_DeG_per_NEs(path_to_DeG=os.path.join(path_to_questions_folder, subpath_to_DeG_file))
        # save by real name NEs
        DeG_per_real_name_key = {values['real_name']: values for ne, values in this_dataset_DeG.items()}
        all_datasets_DeG[dataset_name] = DeG_per_real_name_key
    print(all_datasets_DeG)

    print(all_datasets_DeG['crossner_science']['astronomical object']['gpt_answer'])

    from src.SFT_finetuning.commons.basic_utils import load_json
    general_instruction = load_json("../configs/instruction_configs/instruction.json")[0]
    #print(general_instruction)

    zero_shot_datasets_N_labels_per_prompt = []
    for sample in test_set:
        new_chunked_samples = process_sample(all_datasets_DeG, general_instruction, sample, labels_per_prompt=5)
        zero_shot_datasets_N_labels_per_prompt.extend(new_chunked_samples)

    print(len(zero_shot_datasets_N_labels_per_prompt))
    zero_shot_datasets_N_labels_per_prompt = Dataset.from_list(zero_shot_datasets_N_labels_per_prompt)
    zero_shot_datasets_N_labels_per_prompt.to_json("../data/zero-shot-test-wDeG-5NEperPrompt.jsonl")

