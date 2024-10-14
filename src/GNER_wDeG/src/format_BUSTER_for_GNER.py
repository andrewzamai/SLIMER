""" Data handler for converting BUSTER dataset to input format expected by GNER models https://github.com/yyDing1/GNER """
import json

from datasets import Dataset
from src.data_handlers import data_handler_BUSTER


def chunk_document_w_sliding_window(document_tokens, document_labels, window_size=150, overlap=30):
    """ splits a long BUSTER document in chunks of length=window_size, with an overlap b/t two consecutive windows of 'overlap' words """
    chunks = []
    start = 0
    end = window_size
    while start < len(document_tokens):
        chunk_tokens = document_tokens[start:end]
        chunk_labels = document_labels[start:end]
        chunks.append({"chunk_tokens": chunk_tokens, "chunk_labels": chunk_labels})
        start += window_size - overlap
        end += window_size - overlap

    return chunks


def convert_test_dataset_for_GNER_inference_sliding_window_chunking(BUSTER_handler, with_DeG=True, window_size=150, overlap=15):
    """ splits a long BUSTER document in chunks of length=window_size, with an overlap b/t two consecutive windows of 'overlap' words """
    BUSTER_BIO = BUSTER_handler.datasetdict_BIO['test']
    label_list = list(BUSTER_handler.get_map_to_extended_NE_name().values())
    print(label_list)
    label_list_to_str = ', '.join(label_list)

    if with_DeG:
        BUSTER_Def_and_Guidelines = BUSTER_handler.load_DeG_per_NEs()

    GNER_samples = []
    n_chunks_per_document = {x: 0 for x in BUSTER_BIO['id']}
    for BIO_sample in BUSTER_BIO:
        document_id = BIO_sample['id']
        document_tokens = BIO_sample['tokens']
        document_labels = BIO_sample['labels']

        chunks = chunk_document_w_sliding_window(document_tokens, document_labels, window_size, overlap)
        for chunk_id, chunk in enumerate(chunks):
            chunk_tokens = chunk['chunk_tokens']
            chunk_labels = chunk['chunk_labels']

            n_chunks_per_document[document_id] += 1

            chunk_input = ' '.join(chunk_tokens)

            instruction = "Please analyze the sentence provided, identifying the type of entity for each word on a token-by-token basis.\nOutput format is: word_1(label_1), word_2(label_2), ...\nWe'll use the BIO-format to label the entities, where:\n1. B- (Begin) indicates the start of a named entity.\n2. I- (Inside) is used for words within a named entity but are not the first word.\n3. O (Outside) denotes words that are not part of a named entity.\n\n"
            instruction += f"Use the specific entity tags: {label_list_to_str} and O.\n"

            if with_DeG:
                print()
                BUSTER_Def_and_Guidelines = {x['real_name']: x for ne, x in BUSTER_Def_and_Guidelines.items()}
                instruction += "To help you, here are dedicated DEFINITION and GUIDELINES for each entity tag.\n"
                sampled_labels_DeG = {}
                for ne_tag in label_list:
                    sampled_labels_DeG[ne_tag] = BUSTER_Def_and_Guidelines[ne_tag]['gpt_answer']
                instruction += json.dumps(sampled_labels_DeG, indent=2)
                instruction += '\n'
            instruction += f"\nSentence: {chunk_input}."

            gold_labels_per_token = []
            for label in chunk_labels:
                if label != 'O':
                    label_prefix, label_tagFamTagName = label.split('-')  # splitting B-tagFamily.TagName
                    # converting tagName in Natural language format as label_list
                    label_NL = BUSTER_handler.get_map_to_extended_NE_name()[label_tagFamTagName]
                    label = label_prefix + '-' + label_NL
                    gold_labels_per_token.append(label)
                else:
                    gold_labels_per_token.append(label)

            GNER_sample = {
                "task": "NER",
                "dataset": "BUSTER",
                "split": "test",
                "label_list": label_list,
                "negative_boundary": None,
                "instance": {
                    "id": document_id,
                    "subpart": chunk_id,
                    "words": chunk_tokens,
                    "labels": gold_labels_per_token,
                    "instruction_inputs": instruction,
                },
                "prediction": ""
            }

            GNER_samples.append(GNER_sample)

    return Dataset.from_list(GNER_samples)


if __name__ == '__main__':

    BUSTER_handler = data_handler_BUSTER.BUSTER(
        "expertai/BUSTER",
        #path_to_templates='../SFT_finetuning/templates',
        #SLIMER_prompter_name='SLIMER_instruction_template',
        path_to_DeG='../../data_handlers/questions/BUSTER/gpt_guidelines/BUSTER_NE_definitions.json'
    )

    BUSTER_BIO = BUSTER_handler.datasetdict_BIO
    print(BUSTER_BIO)

    #ne_types_list = ['generic consulting company', 'legal consulting company', 'annual revenues', 'acquired company', 'buying company', 'selling company']
    #print(ne_types_list)

    BUSTER_GNER_test_sliding_window = convert_test_dataset_for_GNER_inference_sliding_window_chunking(BUSTER_handler, window_size=100, overlap=15)

    BUSTER_GNER_test_sliding_window.to_json("../data/BUSTER_test_GNER_wDeG.jsonl")




