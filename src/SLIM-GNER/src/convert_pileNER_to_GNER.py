import json
import re
from datasets import load_dataset
from datasets import Dataset

"""
Generate words, labels lists from each SLIMER sample (one tagName over one sample)
# Sample input data
data = {
    "doc_question_pairID": "29363:3",
    "tagName": "culture",
    "input": "World of the Five Gods\n\nThe World of the Five Gods is the setting for an RPG campaign. It starts in the Weald, a Celtic-inspired fantasy land which is trying to become strong enough to prevent invasion while maintaining the core of its traditional culture. It's magics are rooted in nature and spirits and sacrifice. Beyond the Weald is a world of high magic, steampunk technology, and mutant psionics. Greatly different than our own world is the system of practical theology that pervades magic, religion, and culture of every nation. This world setting borrows heavily by the setting of Lois McMaster Bujold's Five Gods series (with her permission), but adapts it to use with a homebrewed D&D4E game system. It also adapts elements from the the Iskryne trilogy by Elizabeth Moon, Dune by Frank Herbert, and various other wonderful worlds.",
    "instruction": "Extract the Named Entities of type CULTURE from the text chunk you have read.\nReturn a JSON list of instances of this Named Entity type. Return an empty list if no instances are present.",
    "output": "[\"Celtic\", \"Iskryne\", \"Dune\"]"
}

# Function to split text into words and punctuation
def split_text(text):
    words = re.findall(r'\w+|[^\w\s]', text, re.UNICODE)
    return words

# Function to generate BIO labels
def generate_bio_labels(words, entities, tag):
    labels = ["O"] * len(words)
    for entity in entities:
        entity_words = entity.split()  # Split the entity into constituent words
        in_entity = False
        for i, word in enumerate(words):
            if word == entity_words[0]:
                # Check if the sequence of words matches the entity
                if words[i:i+len(entity_words)] == entity_words:
                    in_entity = True
                    labels[i] = f"B-{tag}"
                    for j in range(1, len(entity_words)):
                        labels[i + j] = f"I-{tag}"
                    break  # Move to the next entity
            if in_entity:
                break
    return labels
"""

# Sample input data with multiple tags and entities
data = {
    "docID": "29363",
    "tags_entities": {
        "culture": ["Celtic", "Iskryne", "Dune"],
        "technology": ["steampunk", "psionics"]
    },
    "input": "World of the Five Gods\n\nThe World of the Five Gods is the setting for an RPG campaign. It starts in the Weald, a Celtic-inspired fantasy land which is trying to become strong enough to prevent invasion while maintaining the core of its traditional culture. It's magics are rooted in nature and spirits and sacrifice. Beyond the Weald is a world of high magic, steampunk technology, and mutant psionics. Greatly different than our own world is the system of practical theology that pervades magic, religion, and culture of every nation. This world setting borrows heavily by the setting of Lois McMaster Bujold's Five Gods series (with her permission), but adapts it to use with a homebrewed D&D4E game system. It also adapts elements from the the Iskryne trilogy by Elizabeth Moon, Dune by Frank Herbert, and various other wonderful worlds.",
}


# Function to split text into words and punctuation
def split_text(text):
    words = re.findall(r'\w+|[^\w\s]', text, re.UNICODE)
    return words


# Function to generate BIO labels for multiple tags
def generate_bio_labels(words, tags_entities):
    labels = ["O"] * len(words)

    for tag, entities in tags_entities.items():
        for entity in entities:
            entity_words = entity.split()  # Split the entity into constituent words
            i = 0
            while i < len(words):
                if words[i:i + len(entity_words)] == entity_words:
                    labels[i] = f"B-{tag}"
                    for j in range(1, len(entity_words)):
                        labels[i + j] = f"I-{tag}"
                    i += len(entity_words)  # Skip past the entity just labeled
                else:
                    i += 1  # Move to the next word
    return labels


# Save to CoNLL-style format
def save_conll_format(dataset_path, words_list, labels_list):
    with open(dataset_path, 'w') as f:
        for words, labels in zip(words_list, labels_list):
            for word, label in zip(words, labels):
                f.write(f"{word}\t{label}\n")
            f.write("\n")  # Sentence boundary


if __name__ == '__main__':

    # load pileNER subset used for training SLIMER-391x5(391 NEs, 5pos+5neg per NE)
    #data_SLIMER_format = load_dataset("json", data_files=f'../../../datasets/pileNER/391x5pos_5neg_GenQA_FalseDef-SI/train.jsonl')['train']
    #data_SLIMER_format = load_dataset("json", data_files=f'../../../datasets/pileNER/391x50pos_50neg_GenQA_FalseDef/validation.jsonl')['train']

    #from MSEQA_4_NER.data_handlers import data_handler_pileNER
    #dataset_MSEQA_format_FalseDef = data_handler_pileNER.build_dataset_MSEQA_format_with_n_samples_per_NE_pos_neg(n_pos_samples_per_NE=10, n_neg_samples_per_NE=10, removeTestDatasetsNEs=True, keep_only_top_tagNames=391)
    #data_handler_pileNER.convert_MSEQA_dataset_to_GenQA_format(dataset_MSEQA_format_FalseDef, with_definition=False, path_to_save_to="../../../datasets/pileNER/391x10pos_10neg_GenQA_FalseDef")

    data_SLIMER_format = load_dataset("json", data_files=f'../../../datasets/pileNER/391x50pos_50neg_GenQA_FalseDef/train.jsonl')['train']
    print(data_SLIMER_format)

    # HERE we group the SLIMER samples per docID and collect all the tagName-occurrences for a docID

    # count occurrences of each unique docID
    id_list = [sample["doc_question_pairID"].split(':')[0] for sample in data_SLIMER_format]
    from collections import Counter
    id_counts = Counter(id_list)
    # Sort the dictionary by count in descending order
    sorted_id_counts = dict(sorted(id_counts.items(), key=lambda item: item[1], reverse=True))

    # collect the tagName-entities for each docID
    entities_per_docID = {docID: {} for docID in sorted_id_counts.keys()}
    unique_tagNames = set()
    for sample in data_SLIMER_format:
        docID = sample["doc_question_pairID"].split(':')[0]
        tagName = sample['tagName']
        unique_tagNames.add(tagName)
        entities = json.loads(sample['output'])
        entities_per_docID[docID][tagName] = entities
    #print(entities_per_docID)

    inputs_per_docID = {sample["doc_question_pairID"].split(':')[0]: sample['input'] for sample in data_SLIMER_format}
    # GENERATE the words, labels lists for each docID
    data_BIO_format = []
    # max_dataset_size = 3910
    for docID in sorted_id_counts:
        input = inputs_per_docID[docID]
        # Process the input data
        words = split_text(input)
        # Extract entities from the output
        entities_per_tagName = entities_per_docID[docID]
        bio_labels = generate_bio_labels(words, entities_per_tagName)

        # TODO: ONLY FOR GOLLIE merge spaces with dash _
        # bio_labels = ['_'.join(label.split()) for label in bio_labels]

        if len(words) == len(bio_labels):
            # do not add if negative sample (all O)
            if not all(element == "O" for element in bio_labels):
                data_BIO_format.append((words, bio_labels))
                # max_dataset_size -= 1

        #if max_dataset_size <= 0:
        #    break

    print(len(data_BIO_format))
    print(data_BIO_format[0])

    count_music_group = 0
    for docID in inputs_per_docID:
        entities_per_tagName = entities_per_docID[docID]
        for ne in entities_per_tagName.keys():
            if ne == 'person':
                count_music_group += 1
    print(count_music_group)

    #save_conll_format('../../../datasets/pileNER/PileNER-3910samples-GoLLIE/train.txt', [x[0] for x in data_BIO_format], [x[1] for x in data_BIO_format])
    print(len(unique_tagNames))
    print(unique_tagNames)

    def convert_labels_from_json_to_txt(json_filename, txt_filename):
        # Load the JSON file
        with open(json_filename, 'r') as json_file:
            words = json.load(json_file)

        # Ensure the loaded data is a list of words
        if not isinstance(words, list):
            raise ValueError("JSON content is not a list.")

        # Save the words to a text file, one word per line
        with open(txt_filename, 'w') as txt_file:
            for word in words:
                txt_file.write(word + '\n')


    # Example usage
    json_filename = '../../../datasets/pileNER/top_391_NamedEntities.json'
    txt_filename = '../../../datasets/pileNER/391x5pos_5neg_BIO/label.txt'
    #convert_labels_from_json_to_txt(json_filename, txt_filename)

    """
    data_BIO_format = []
    for sample in data_SLIMER_format:
        # Process the input data
        words = split_text(sample["input"])
        # Extract entities from the output
        entities = json.loads(sample['output'])
        bio_labels = generate_bio_labels(words, entities, sample["tagName"])

        if len(words) == len(bio_labels) and entities != []:
            data_BIO_format.append((words, bio_labels))

    print(len(data_BIO_format))
    #save_conll_format('../../../datasets/pileNER/391x5pos_5neg_BIO/train.txt', [x[0] for x in data_BIO_format], [x[1] for x in data_BIO_format])

    def convert_labels_from_json_to_txt(json_filename, txt_filename):
        # Load the JSON file
        with open(json_filename, 'r') as json_file:
            words = json.load(json_file)

        # Ensure the loaded data is a list of words
        if not isinstance(words, list):
            raise ValueError("JSON content is not a list.")

        # Save the words to a text file, one word per line
        with open(txt_filename, 'w') as txt_file:
            for word in words:
                txt_file.write(word + '\n')


    # Example usage
    json_filename = '../../../datasets/pileNER/top_391_NamedEntities.json'
    txt_filename = '../../../datasets/pileNER/391x5pos_5neg_BIO/label.txt'
    convert_labels_from_json_to_txt(json_filename, txt_filename)
    """

    """
    data_BIO_format = []
    for sample in data_SLIMER_format:
        # Process the input data
        words = split_text(sample["input"])
        # Extract entities from the output
        entities = json.loads(sample['output'])
        bio_labels = generate_bio_labels(words, entities, sample["tagName"])

        if len(words) == len(bio_labels):
            data_BIO_format.append(
                {
                    "id": sample['doc_question_pairID'],
                    "tagName": sample['tagName'],
                    "negative_sample": json.loads(sample['output']) == [],
                    "words": words,
                    "labels": bio_labels
                }
            )

    data_BIO_format = Dataset.from_list(data_BIO_format)
    print(data_BIO_format)
    data_BIO_format.to_json('../../../datasets/pileNER/391x5pos_5neg_BIO/train.jsonl')
    """

