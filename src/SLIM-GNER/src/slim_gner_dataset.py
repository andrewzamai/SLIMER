import pandas as pd
import datasets
import random
import json
import csv
import os
import re

logger = datasets.logging.get_logger(__name__)

# modified from https://github.com/BeyonderXX/InstructUIE/blob/master/src/uie_dataset.py
class SLIMGNERConfig(datasets.BuilderConfig):
    """
    SLIMGNERDataset config
    """

    def __init__(
        self,
        *args,
        data_dir=None,
        instruction_file=None,
        data_config_dir=None,
        add_dataset_name=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.data_dir = data_dir
        self.instructions = self._parse_instruction(instruction_file)
        self.data_configs = self._parse_data_config(data_config_dir)
        self.add_dataset_name = add_dataset_name

    def _parse_instruction(self, instruction_file):
        """
        Instruction file example:
        [
            "Please analyze the sentence provided, list the inherent entities\n",
            "List all the entities in the sentence"
        ]
        """
        if not instruction_file:
            return None
        with open(instruction_file, 'r+') as f:
            instructions = json.load(f)
        return instructions

    def _parse_data_config(self, data_config_dir):
        """
        Task config file example:
        [
            {
                "dataset name": "mit-movie",
                "sampling strategy": "random",
                "max_num_instances_per_task": 200,
                "over_sampling": false
            },
            {
                "dataset name": "mit-restaurant",
                "sampling strategy": "random",
                "max_num_instances_per_task": 200,
                "over_sampling": false
            }
        ]
        """
        if not data_config_dir:
            return None

        data_configs = {}
        for split in ["train", "dev", "test"]:
            file_name = f"{split}_configs.json"
            data_config_file = os.path.join(data_config_dir, file_name)

            if not os.path.exists(data_config_file):
                raise ValueError('Please check {} split, {} not exists!'.format(split, data_config_file))

            with open(data_config_file, 'r+') as f:
                data_configs[split] = json.loads(f.read())

        return data_configs


class SLIMGNERDataset(datasets.GeneratorBasedBuilder):
    """SLIMGNER Datasets"""

    BUILDER_CONFIG_CLASS = SLIMGNERConfig

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "dataset": datasets.Value("string"),
                    "split": datasets.Value("string"),
                    "label_list": datasets.Sequence(datasets.Value("string")),
                    "instance": {
                        "id": datasets.Value("string"),
                        "words": datasets.Sequence(datasets.Value("string")),
                        "labels": datasets.Sequence(datasets.Value("string")),
                        "instruction_inputs": datasets.Value("string"),
                        "prompt_labels": datasets.Value("string"),
                    }
                }
            ),
            supervised_keys=None
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        if self.config.data_dir is None or self.config.data_configs is None:
            logger.error("Please provide right input: data_dir or data_config_dir!")

        data_configs = self.config.data_configs

        split_generators = []
        if len(data_configs['train']) > 0:
            split_generators.append(
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "data_config": data_configs['train'],
                        "add_dataset_name": self.config.add_dataset_name,
                        "split": "train"
                    }
                )
            )
        if len(data_configs['dev']) > 0:
            split_generators.append(
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={
                        "data_config": data_configs['dev'],
                        "add_dataset_name": self.config.add_dataset_name,
                        "split": "dev"
                    }
                )
            )
        if len(data_configs['test']) > 0:
            split_generators.append(
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                        "data_config": data_configs['test'],
                        "add_dataset_name": self.config.add_dataset_name,
                        "split": "test"
                    }
                ),
            )

        return split_generators

    # read conll-style dataset
    def _load_dataset(self, dataset_path, labels_path):
        data_df = pd.read_csv(dataset_path, delimiter='\t', quoting=csv.QUOTE_NONE, skip_blank_lines=False, header=None, keep_default_na=False, na_values=[''], low_memory=False)

        with open(labels_path, "r") as f:
            labels_list = f.read().splitlines()
            assert "O" not in labels_list

        instances = []
        idx, words, labels = 0, [], []
        for row in data_df.values:
            if not pd.isna(row[1]):
                words.append(row[0])
                labels.append(row[-1])
            elif words != []:
                instances.append({"id": idx, "words": words, "labels": labels})
                idx += 1
                words, labels = [], []

        if words != []:
            instances.append({"id": idx, "words": words, "labels": labels})

        return instances, labels_list

    def _get_instruction(self):
        return self.config.instructions[0]

    def _generate_labeled_string(self, words, labels):
        """
        Generates a labeled string from words and BIO labels in span-based notation.

        Args:
        - words (list of str): The list of words in the sentence.
        - labels (list of str): The list of BIO labels corresponding to the words.

        Returns:
        - str: The sentence with entities labeled in the desired format.
        """
        # Initialize variables
        output = ""
        current_entity = ""
        current_entity_type = ""
        is_entity_open = False

        # Parse through labels and words
        for label, word in zip(labels, words):
            if label.startswith('B-'):
                # If an entity is already open, close it first
                if is_entity_open:
                    output += f"[{current_entity_type}] {current_entity} [/{current_entity_type}] "
                # Start a new entity
                current_entity_type = label[2:]  # Extract entity type (e.g., 'ORGANIZATION')
                current_entity = word
                is_entity_open = True
            elif label.startswith('I-'):
                # Continue the entity
                current_entity += " " + word
            else:  # label == 'O'
                # Close the current entity if it was open
                if is_entity_open:
                    output += f"[{current_entity_type}] {current_entity} [/{current_entity_type}] "
                    current_entity = ""
                    current_entity_type = ""
                    is_entity_open = False
                # Add the regular word to the output
                output += word + " "

        # Close any remaining open entity at the end
        if is_entity_open:
            output += f"[{current_entity_type}] {current_entity} [/{current_entity_type}] "

        # Return the final formatted string, stripping any trailing whitespace
        return output.strip()

    # sample instances
    def _sampling_dataset(self, instances, sampling_strategy, max_num_instances, over_sampling):
        if sampling_strategy == 'random' and max_num_instances is not None and max_num_instances >= 0:
            instances = instances[:max_num_instances]
        if max_num_instances is not None and over_sampling and len(instances) < max_num_instances:
            origin_instances = instances.copy()
            while len(instances) < max_num_instances:
                instances.extend(origin_instances)
        return instances

    def load_DeG_per_NEs(self, path_to_DeG):
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
            ne_type_in_natural_language = ne_tag
            pattern = re.compile(rf'\'{re.escape(ne_type_in_natural_language)}\'', re.IGNORECASE)
            this_ne_guidelines = {k: pattern.sub(f'{ne_type_in_natural_language.upper()}', v) for k, v in this_ne_guidelines.items()}
            values['gpt_answer'] = this_ne_guidelines

        return DeG_per_NEs_raw

    def _generate_examples(self, data_config, add_dataset_name, split):
        """Yields examples."""
        data_dir = self.config.data_dir
        if len(data_config) == 0:
            return
        # Load data from files
        for dataset in data_config:
            # Read info from data_config
            dataset_name = dataset["dataset name"]
            sampling_strategy = dataset.get("sampling strategy", "random")
            max_num_instances = dataset.get("max_num_instances", "full")
            over_sampling = dataset.get("over_sampling", False)
            dataset_path = os.path.join(data_dir, dataset_name, split + ".txt")
            labels_path = os.path.join(data_dir, dataset_name, "label.txt")
            assert os.path.exists(dataset_path)
            assert os.path.exists(labels_path)

            # load data from files
            instances, label_list = self._load_dataset(dataset_path, labels_path)
            instances = self._sampling_dataset(instances, sampling_strategy, max_num_instances, over_sampling)
            for idx, instance in enumerate(instances):
                words, labels = instance["words"], instance["labels"]
                instruction = self._get_instruction()

                # define the number of labels to extract per prompt
                labels_per_prompt = dataset.get("labels_per_prompt", 5)
                # extract unique labels from the instance, excluding "O"
                labels_in_instance = set(label.split('-')[-1] for label in labels if label != "O")
                # sample N-1 labels from labels_in_instance, allowing for repetition
                sampled_labels = []
                if labels_in_instance:
                    sampled_labels = random.choices(list(labels_in_instance), k=labels_per_prompt - 1)
                # Sample 1 label that is not in labels_in_instance from label_list
                available_labels = [label for label in label_list if label not in labels_in_instance]
                # Sample one label from available_labels, or default to "O" if none are available
                if available_labels:
                    new_label = random.choice(available_labels)
                    sampled_labels.append(new_label)

                # Set any label in instance['labels'] that is not in sampled_labels to "O"
                for i in range(len(labels)):
                    if labels[i].split('-')[-1] not in sampled_labels:
                        labels[i] = "O"  # Replace with "O" if not in sampled_labels
                    # AND CAPITALIZE LABELS
                    if labels[i] != "O":
                        prefix, tag = labels[i].split("-")
                        labels[i] = prefix + '-' + tag.upper()

                random.shuffle(sampled_labels)

                sampled_labels = [l.upper() for l in sampled_labels]

                instruction += f"\nIdentify the named entities with these specific entity tags: {', '.join(sampled_labels)}.\n"

                # it the path to DeG is provided, append the Def and Guidelines for each NE
                path_to_DeG = dataset.get("path_to_DeG", "")
                if path_to_DeG:
                    DeG_per_NEs = self.load_DeG_per_NEs(path_to_DeG)
                    instruction += "To help you, here are dedicated DEFINITION and GUIDELINES for each entity tag.\n"

                    sampled_labels_DeG = {}
                    for ne_tag in sampled_labels:
                        ne_tag_lowercased = ne_tag.lower()
                        sampled_labels_DeG[ne_tag] = DeG_per_NEs[ne_tag_lowercased]['gpt_answer']
                    instruction += json.dumps(sampled_labels_DeG, indent=2)
                    instruction += '\n'

                if add_dataset_name:
                    instruction += f"Dataset: {dataset_name}.\n"

                instruction += "Sentence: " + " ".join(words)

                label_text = self._generate_labeled_string(words, labels)

                yield f"{dataset_name}##{idx}", {
                    "dataset": dataset_name,
                    "split": split,
                    "label_list": label_list,
                    "instance": {
                        "id": str(idx),
                        "words": instance["words"],
                        "labels": instance["labels"],
                        "instruction_inputs": instruction,
                        "prompt_labels": label_text,
                    }
                }
