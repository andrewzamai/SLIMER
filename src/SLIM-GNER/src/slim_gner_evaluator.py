import json
import re
import string
from collections import defaultdict
from transformers import AutoTokenizer
from tqdm import tqdm
import argparse


# Extract entities from the tagged predictions
def extract_entities(preds_text):
    # Regex to find all tags and their corresponding content
    # This pattern captures both closed tags like [TYPE] entity [/TYPE] and unclosed tags like [TYPE] entity
    pattern = r'\[(.*?)\]\s*([^[]*?)(?:\s*\[\/\1\]|\s*[\.\!\?]|$)'
    entities = []

    try:
        # Iterate over all matches
        for match in re.finditer(pattern, preds_text):
            # If there is a match, extract the type and entity text
            entity_type = match.group(1).strip().upper()  # Convert to uppercase
            entity_text = match.group(2).strip()
            if entity_text:  # Only consider non-empty entities
                entities.append((entity_text, entity_type))
    except:
        entities = []

    return entities


# Convert the unstructured texts into structured entities
def extract_predictions(example):
    prediction = example['prediction'].strip()
    #prediction = example['instance']['prompt_labels'].strip()
    predictions = extract_entities(prediction)
    return predictions


# Compute F1 score
class SLIMGNEREvaluator:
    def evaluate(self, examples: list):
        n_correct, n_pos_gold, n_pos_pred = 0, 0, 0
        for example in tqdm(examples):
            words = example['instance']['words']
            labels = example['instance']['labels']
            predictions = extract_predictions(example)
            predictions = list(set([(normalize_answer(x[0]), x[1]) for x in predictions]))
            gold_entities = parser(words, labels)

            for pred in predictions:
                if pred in gold_entities:
                    n_correct += 1
                """
                else:
                    print(example['instance']['prompt_labels'])
                    print("prediction:", predictions)
                    print(example['instance']['labels'])
                    print(example['instance']['words'])
                    print("gold entities:", gold_entities)
                    print("---------------------------------------------")
                """
                n_pos_pred += 1
            n_pos_gold += len(gold_entities)

        prec = n_correct / (n_pos_pred + 1e-10)
        recall = n_correct / (n_pos_gold + 1e-10)
        f1 = 2 * prec * recall / (prec + recall + 1e-10)
        return {
            'precision': prec,
            'recall': recall,
            'f1': f1,
        }


def compute_metrics(examples):
    all_examples = defaultdict(list)
    for example in examples:
        all_examples[example['dataset']].append(example)

    # evaluate
    results = {}
    tot_f1, tot_dataset = 0, 0
    for dataset in all_examples:
        eval_result = SLIMGNEREvaluator().evaluate(all_examples[dataset])
        results[f"{dataset}_precision"] = eval_result["precision"]
        results[f"{dataset}_recall"] = eval_result["recall"]
        results[f"{dataset}_f1"] = eval_result["f1"]
        tot_f1 += eval_result["f1"]
        tot_dataset += 1
    results["average_f1"] = tot_f1 / tot_dataset
    return results


# normalize answer, 
# cp from https://github.com/universal-ner/universal-ner/blob/main/src/eval/evaluate.py
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

# parser BIO format into entity format, 
# modified from https://github.com/universal-ner/universal-ner/blob/main/src/eval/evaluate.py
def parser(words, labels):
    assert len(words) == len(labels)
    spans_list = []
    span_words, span_label = [], None
    for word, label in zip(words, labels):
        if len(span_words) > 0 and (label[0] == 'B' or label[0] == 'O'):
            spans_list.append((' '.join(span_words), span_label))
            span_words, span_label = [], None
        if label != 'O':
            span_words.append(word)
            span_label = label[2:]
    if span_label is not None:
        spans_list.append((' '.join(span_words), span_label))
    formatted_items = []
    for item in spans_list:
        """
        if isinstance(item, list) or isinstance(item, tuple):
            item = tuple([normalize_answer(element) for element in item])
        else:
            item = normalize_answer(item)
        """
        item = (normalize_answer(item[0]), item[1])
        if item not in formatted_items:
            formatted_items.append(item)
    return formatted_items


def main():
    prediction_path = '../data/SLIMGNER_pileNER_wD&G_validation.json'
    all_examples = defaultdict(list)
    with open(prediction_path, 'r') as fh:
        for line in fh.readlines():
            line_data = json.loads(line)
            all_examples[line_data['dataset']].append(line_data)

    # evaluate
    tot_f1, tot_dataset = 0, 0
    for dataset in all_examples:
        eval_result = SLIMGNEREvaluator().evaluate(all_examples[dataset])
        print(f'Dataset: {dataset}, F1: {eval_result["f1"]}, Precision: {eval_result["precision"]}, Recall: {eval_result["recall"]}')
        tot_f1 += eval_result["f1"]
        tot_dataset += 1
    print(f'avg_f1: {tot_f1 / tot_dataset}')


if __name__ == "__main__":
    main()

'''
Example of predictions:
{
    "task": "NER", 
    "dataset": "WikiNeural", 
    "split": "test", 
    "label_list": ["location", "person", "organization"], 
    "negative_boundary": null, 
    "instance": {
        "id": "11596", 
        "subpart": "1", 
        "words": ["This", "system", "was", "widely", "copied", "in", "various", "NATO", "forces", "."], 
        "labels": ["O", "O", "O", "O", "O", "O", "O", "B-organization", "O", "O"], 
        "instruction_inputs": "Please analyze the sentence provided, identifying the type of entity for each word on a token-by-token basis.\nOutput format is: word_1(label_1), word_2(label_2), ...\nWe'll use the BIO-format to label the entities, where:\n1. B- (Begin) indicates the start of a named entity.\n2. I- (Inside) is used for words within a named entity but are not the first word.\n3. O (Outside) denotes words that are not part of a named entity.\n\nUse the specific entity tags: location, person, organization, else and O.\nDataset: WikiNeural.\nSentence: This system was widely copied in various NATO forces .", 
        "prompt_labels": "This(O) system(O) was(O) widely(O) copied(O) in(O) various(O) NATO(B-organization) forces(O) .(O)"
    }, 
    "prediction": "This(O) system(O) was(O) widely(O) copied(O) in(O) various(O) NATO(B-organization) forces(O).(O)"
}
'''
