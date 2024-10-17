"""
SLIMER data_handler for Pile-NER-type dataset

Does not implement DataInterface

- https://huggingface.co/datasets/Universal-NER/Pile-NER-type
- https://universal-ner.github.io/

Chunks of text from PILE corpus, synthetically annotated by ChatGPT (see UniNER paper)
Original PileNER-type dataset comprises over 13K distinct NE classes
To reduce overlap with test sets MIT/CrossNER in Zero-Shot NER evaluation on never-seen-before NEs
    1) we only consider 455-top frequent NEs
    2) after some further deletions and NE mergings (lower casing + dict_of_merges) --> 423 NEs
    3) deleting those in MIT/CrossNER test --> 391 NEs
"""

import os
import re
import ast
import sys
import json
import math
import random
import string
import numpy as np
from collections import OrderedDict, Counter
from datasets import Dataset, DatasetDict, load_dataset

# SLIMER prompter to format a ne_tag, Def and Guidelines into a prompt for NER
from src.SFT_finetuning.commons.prompter import SLIMER_instruction_prompter, SLIMER_PARALLEL_instruction_prompter


def extract_context_quests_answers(conversation):
    """
    given a single UniversalNER conversation sample, extract context (text passage) + (questions-gold_answers) list
    """
    # first element in the conversation list is the passage of text (context) provided by the human
    context = conversation.pop(0)
    if context["from"] == "human" and context["value"][:5] == "Text:":
        context = context["value"][len("Text: "):]
    else:
        raise ValueError("Invalid context or source in the conversation")

    # gpt confirms { "from": "gpt", "value": "I've read this text." }
    conversation.pop(0)

    # extracting list of questions for the context and to each q its associated list of answers
    quests_answers = []
    # reading 2 by 2
    for i in range(0, len(conversation), 2):
        if conversation[i]["from"] == "human" and conversation[i + 1]["from"] == "gpt":
            # extracting question from human
            question = conversation[i]["value"]
            # NE type being extracted
            start_char_ne_type = len("What describes ")
            end_char_ne_type = question.find("in the text?") - 1
            # extracting NE type and lowering so PERSON/Person/person are same
            ne_type = question[start_char_ne_type:end_char_ne_type].lower()
            # lower casing NE e.g. Person, PERSON to be both person
            question = "What describes " + ne_type + " in the text?"

            # extracting answers from gpt
            ga_answers = {'answer_start': [], 'text': []}
            answers = ast.literal_eval(conversation[i + 1]["value"])
            for ans in answers:
                # finding target_word occurrences in the context
                # by returning a list of all starting positions in character
                def find_start_positions(text, target_word):
                    start_positions = []
                    # Find occurrences of the target_word
                    pattern = re.compile(r'\b' + re.escape(target_word) + r'\b')  # word bound delimiter
                    matches = pattern.finditer(text)
                    for match in matches:
                        start_positions.append({"answer_start": match.start(), "text": target_word})
                    return start_positions

                start_positions = find_start_positions(context, ans)

                for sp in start_positions:
                    ga_answers['text'].append(sp["text"])
                    ga_answers['answer_start'].append(sp["answer_start"])

                if len(ga_answers['text']) != len(ga_answers['answer_start']):
                    raise ValueError("number of answer text not matching number of answer_start")

            quests_answers.append({"question": question, "ne_type": ne_type, "answers": ga_answers})

        else:
            raise ValueError("human-gpt non matched conversation")

    return {"context": context, "questions_answers": quests_answers}


def get_dataset_statistics():
    raw_dataset = load_dataset("Universal-NER/Pile-NER-type")

    context_lengths = []
    n_total_samples = 0
    n_negative_samples = 0
    for raw_sample in raw_dataset['train']['conversations']:
        # extract context and list of question-goldAnswers associated to each context
        context, questions_answers_list = extract_context_quests_answers(raw_sample).values()

        context_length = len(context.split())
        context_lengths.append(context_length)

        for q_ne_answers in questions_answers_list:
            n_total_samples += 1
            if q_ne_answers['answers']['text'] == []:
                n_negative_samples += 1

    fullPileNER_tagName_list = {}
    for raw_sample in raw_dataset['train']['conversations']:
        context, questions_answers_list = extract_context_quests_answers(raw_sample).values()
        for question in questions_answers_list:
            fullPileNER_tagName_list[question['ne_type']] = 1

    return {
        'contexts_average_number_words': np.average(context_lengths),
        'contexts_min_number_words': np.min(context_lengths),
        'contexts_max_number_words': np.max(context_lengths),
        'fullPileNER_tagName_list': [len(list(fullPileNER_tagName_list.keys())), list(fullPileNER_tagName_list.keys())],
        'number_total_QA_samples': n_total_samples,
        'number_negative_QA_samples': [n_negative_samples, f"{n_negative_samples/n_total_samples*100}%"]
    }


def get_statistics_for_QA_dataset(dataset_QA, input_column_name, instruction_column_name, output_column_name):
    """ get statistics for MSEQA/GenQA Dataset fold (e.g. train) """
    context_lengths = []
    n_total_samples = 0
    n_negative_samples = 0
    for sample in dataset_QA:
        # counting number words approximately
        context = sample[input_column_name]
        context_length = len(context.split())
        context_lengths.append(context_length)

        # counting number negative samples
        output = sample[output_column_name]
        answers_text = None
        # MSEQA case {'answer_start': [], 'text': []}
        if isinstance(output, dict):
            if 'text' in output:
                answers_text = output['text']
            else:
                raise Exception("Unexpected keys, expected 'text'")
        # GenQA case is a dumped JSON list
        elif isinstance(output, str):
            try:
                answers_text = json.loads(output)
            except:
                answers_text = []

        n_total_samples += 1
        if not answers_text:
            n_negative_samples += 1

    # list of unique NEs
    tagName_list = {}
    for sample in dataset_QA:
        tagName_list[sample['tagName']] = 1

    return {
        'contexts_average_number_words': math.ceil(np.average(context_lengths)),
        'contexts_min_number_words': np.min(context_lengths),
        'contexts_max_number_words': np.max(context_lengths),
        'fullPileNER_tagName_list': [len(list(tagName_list.keys())), list(tagName_list.keys())],
        'number_total_QA_samples': n_total_samples,
        'number_negative_QA_samples': [n_negative_samples, f"{round(n_negative_samples/n_total_samples*100, 2)}%"]
    }


def build_dataset_MSEQA_format():
    print("Downloading PileNER dataset from HF...")
    sys.stdout.flush()
    # downloading raw dataset from huggingface repo (has only "train" partition)
    raw_dataset = load_dataset("Universal-NER/Pile-NER-type")

    # populate list of {context-question-goldAnswers} elements
    context_question_list = []
    context_progressiveID = 0
    for raw_sample in raw_dataset['train']['conversations']:
        # extract context and list of question-goldAnswers associated to each context
        context, questions_answers_list = extract_context_quests_answers(raw_sample).values()
        question_progressiveID = 0
        # copy the context for each question associated to it
        for q_a in questions_answers_list:
            context_question_list.append(
                {"doc_question_pairID": str(context_progressiveID) + ":" + str(question_progressiveID),
                 "document_context": context,
                 "tagName": q_a["ne_type"],
                 "question": q_a["question"],
                 "answers": q_a["answers"]
                 })
            question_progressiveID += 1
        context_progressiveID += 1

    # 358181 context-question pairs
    # using 0.9 for training, 0.05 for validation, 0.05 for test
    train_ratio = 0.9

    num_samples = len(context_question_list)
    num_train = int(train_ratio * num_samples)
    train_fold = context_question_list[:num_train]
    val_test_fold = context_question_list[num_train:]

    val_fold = val_test_fold[:math.floor(len(val_test_fold) / 2.0)]
    test_fold = val_test_fold[math.floor(len(val_test_fold) / 2.0):]

    # shuffling here after partitioning in fold so that same context is not both in train and val/test
    random.seed(42)
    random.shuffle(train_fold)
    random.shuffle(val_fold)
    random.shuffle(test_fold)

    train_dataset = Dataset.from_list(train_fold)
    validation_dataset = Dataset.from_list(val_fold)
    test_dataset = Dataset.from_list(test_fold)

    return DatasetDict({
        "train": train_dataset,
        "validation": validation_dataset,
        "test": test_dataset
    })


def remove_bad_ne_types(dataset_MSEQA_format):
    """
    get same NE types list for which we have GPT guidelines.
    if the pileNER dataset is built by retaining only those NE which number of occurrences is > 100, the total number of NEs now should be 455
    by plotting the dendrogram of the word embeddings using plot_word_emb.ipynb we produce this mapping to a new list of NEs
    by removing some bad NE categories or merging some now the new list of NEs should be of length 423
    """
    print("Keeping only a subset of NE types...")
    sys.stdout.flush()
    ne_types_list = get_ne_types_list(dataset_MSEQA_format, min_num_samples_per_ne_type=100)

    new_ne_type_list_mapping = {
        "misc": None,
        "miscellaneous": None,
        "other": None,
        "unknown": None,
        "general": None,
        "entity type not specified": None,
        "entity type": None,
        "entity": None,
        "text": None,
        "import": None,

        "bacteria": "bacterium",
        "biological": "biological entity",
        "cell": "cell type",
        "cellular component": "cell component",
        "governmental body": "government body",
        "movie": "film",
        "work": "work of art",
        "musical group": "music group",
        "org": "organization",

        "anatomical_structure": "anatomical structure",
        "anatomicalstructure": "anatomical structure",
        "biological_process": "biological process",
        "body_part": "body part",
        "gpe": "geopolitical entity",
        "gene/protein": "gene",
        "work_of_art": "work of art",
        "job_title": "job title",
        "organisation": "organization",
        "chemical_substance": "chemical substance",
        "medical_condition": "medical condition",
        "medicalcondition": "medical condition",

        "fieldterminology": None,
        "cryptocurrency": "cryptocurrency",
        "demonym": "demonym",
        "norp": "norp"
    }

    # new dataset with re-mapped Named Entities
    new_dataset_MSEQA_format_list = {split: [] for split in dataset_MSEQA_format.keys()}
    for split in dataset_MSEQA_format.keys():
        for sample in dataset_MSEQA_format[split]:
            ne_type = sample['tagName']
            old_ne_type = ne_type
            if ne_type in new_ne_type_list_mapping:
                ne_type = new_ne_type_list_mapping[ne_type]  # new NE name or None if to be removed
            # if has not been remove and the new mapping is in the list of NEs for which we have the gpt definition
            if ne_type is not None and ne_type in ne_types_list:
                # assign new NE type
                sample['tagName'] = ne_type
                # replacing the old ne type occurrence to their new UPPERCASE
                pattern = re.compile(re.escape(old_ne_type))
                sample['question'] = pattern.sub(ne_type.upper(), sample['question'])

                new_dataset_MSEQA_format_list[split].append(sample)

    return DatasetDict({split: Dataset.from_list(values) for split, values in new_dataset_MSEQA_format_list.items()})


def get_ne_types_list(dataset_MSEQA_format, min_num_samples_per_ne_type=100):
    """ list of NEs which number of answer spans (i.e. occurrences) across ALL splits is >= min_num_samples_per_ne_type """
    ne_types = {}
    for split in dataset_MSEQA_format.keys():
        for sample in dataset_MSEQA_format[split]:
            if sample["tagName"] in ne_types:
                ne_types[sample["tagName"]] += len(sample['answers']['text'])  # number of occurrences
            else:
                ne_types[sample["tagName"]] = len(sample['answers']['text'])

    ne_types = [a[0] for a in sorted(ne_types.items(), key=lambda item: item[1], reverse=True) if
                a[1] >= min_num_samples_per_ne_type]

    # with open("./questions/ne_types_list.json", 'w') as f:
    # json.dump(ne_types, f, indent=2)

    return ne_types


""" --- functions to extract n sentences per NE type as examples to build definitions through GPT prompting --- """

def split_into_sentences(passage):
    # split the passage into sentences based on punctuation .?! while not splitting "Dr." or "Fig.1"
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s(?! \d)(?!\d)', passage)
    return [sentence for sentence in sentences if sentence.strip()]


def count_target_words(sentence, target_words):
    # count occurrences of the target words in the sentence
    # occurrences = [match.group() for match in re.finditer(r'\b(?:' + '|'.join(map(re.escape, target_words)) + r')\b', sentence, flags=re.IGNORECASE)]
    # return len(occurrences)
    matches = re.finditer(r'\b(?:' + '|'.join(map(re.escape, target_words)) + r')\b', sentence)

    # get the list of target words found in this sentence
    target_words_found = [match.group() for match in matches]

    # count the number of target words found in this sentence
    occurrences_count = len(target_words_found)

    return occurrences_count, target_words_found


def has_too_many_whitespaces(sentence, threshold=4):
    # count consecutive whitespaces
    consecutive_whitespaces = re.findall(r'\s+', sentence)

    # check if the count exceeds the threshold
    return any(len(whitespace) > threshold for whitespace in consecutive_whitespaces)


def has_too_many_newline(sentence, threshold=2):
    # count consecutive newline
    consecutive_newline = re.findall(r'\n+', sentence)

    # check if the count exceeds the threshold
    return any(len(whitespace) >= threshold for whitespace in consecutive_newline)


def has_more_than_n_foreign_chars(sentence, threshold=2):
    foreign_char_count = sum(1 for char in sentence if ord(char) > 127)
    return foreign_char_count > threshold

def has_too_many_punctuations_and_digits(sentence, threshold=5):
    # discard sentences like B [**\\#1**]{} (19\\#2) \\#3]{} \\#1\\#2
    # define the set of allowed punctuations
    allowed_punctuations = set(string.punctuation)
    # count the number of punctuations and digits in the sentence
    punctuation_count = sum(1 for char in sentence if char in allowed_punctuations or char.isdigit())
    return punctuation_count > threshold


def get_one_sentence_from_sample(sample):
    document_context = sample['document_context']
    answers = sample['answers']
    ne_type = sample['tagName']
    # split in sentences according to punctuation .?!
    sentences = split_into_sentences(document_context)
    target_words = answers['text']
    # print(target_words)
    # count the occurrences of target words in each sentence
    # to return the one with at least 1/highest number of occ.
    target_word_counts = []
    for sentence in sentences:
        occurrences_count, target_words_found = count_target_words(sentence, target_words)
        target_word_counts.append({"sentence": sentence,
                                   "target_words_in_it": target_words_found,
                                   "occurrences_count": occurrences_count
                                   })

    # sort sentences by decreasing occurrences_count
    target_word_counts = sorted(target_word_counts, key=lambda x: x['occurrences_count'], reverse=True)
    # returning the sentence with highest number of occurrences, but with some contraints
    sentence_to_ret = None
    i = 0
    while i < len(target_word_counts):
        if target_word_counts[i]['occurrences_count'] != 0:
            if 50 < len(target_word_counts[i]['sentence']) < 100:
                if not has_too_many_whitespaces(target_word_counts[i]['sentence'], 4):
                    if not has_too_many_newline(target_word_counts[i]['sentence'], 1):
                        if not has_more_than_n_foreign_chars(target_word_counts[i]['sentence'], 2):
                            if not has_too_many_punctuations_and_digits(target_word_counts[i]['sentence'], 10):
                                sentence_to_ret = target_word_counts[i]
                                break
            elif ne_type in ['namespace', 'import', 'keyword', 'surname', 'file name', 'header file', 'related art', 'boolean', 'struct', 'html attribute', 'protein domain', 'fieldterminology', 'constant', 'legal citation'] and len(target_word_counts[i]['sentence']) < 200:
                sentence_to_ret = target_word_counts[i]
                break

        i += 1

    return sentence_to_ret


def get_n_sentences_per_ne_type(dataset_MSEQA_format, ne_types_list, n_sentences_per_ne=3):
    # getting from training set n_sentences_per_ne as positive examples from which to let gpt infer NE definition
    sentences_per_ne_type = {ne: [] for ne in ne_types_list}
    trainDataset = dataset_MSEQA_format['train'].to_list()
    random.seed(4)
    random.shuffle(trainDataset)
    for ne_type in ne_types_list:
        i = 0
        while len(sentences_per_ne_type[ne_type]) < n_sentences_per_ne and i < len(trainDataset):
            sample = trainDataset[i]
            if sample['tagName'] == ne_type and len(sample['answers']['text']) != 0:
                sentence_target_words = get_one_sentence_from_sample(sample)
                if sentence_target_words is not None:
                    # removing duplicates in list of target words
                    sentence_target_words['target_words_in_it'] = list(set(sentence_target_words['target_words_in_it']))
                    sentences_per_ne_type[ne_type].append(sentence_target_words)
            i += 1

    not_enough_sentences = []
    for ne_type, sentences in sentences_per_ne_type.items():
        if len(sentences) < n_sentences_per_ne:
            # raise ValueError(f"not enough sentences for {ne_type}")
            not_enough_sentences.append((ne_type, len(sentences)))
    print(f"NE types with less than n_sentences_per_ne: {len(not_enough_sentences)}")
    print(not_enough_sentences)

    return sentences_per_ne_type


def convert_MSEQA_dataset_to_GenQA_format_SI(dataset_MSEQA_format, with_definition, path_to_NE_guidelines_json=None, path_to_save_to='./unk_dataset_GenQA', SLIMER_prompter_name='SLIMER_instruction_template'):

    slimer_prompter = SLIMER_instruction_prompter(SLIMER_prompter_name, '/Users/andrew/ExpertAI/SLIMER/src/SFT_finetuning/templates')

    print("Converting to SLIMER format, adding Definition and Guidelines if specified ...")
    sys.stdout.flush()

    if with_definition:
        # definition and guidelines for each NE in new_NE_type_list
        # obtained by prompting gpt, check prompt_tests.ipynb
        with open(path_to_NE_guidelines_json, 'r') as file:
            all_NEs_guidelines = json.load(file)

    for split_name in dataset_MSEQA_format.keys():
        dataset_GenQA = []
        for MSEQA_sample in dataset_MSEQA_format[split_name]:
            genQA_sample = {
                "doc_tag_pairID": MSEQA_sample['doc_question_pairID'],
                "tagName": MSEQA_sample['tagName'],
                # new column names as finetune_sft.py requires
                "input": MSEQA_sample['document_context'],
                "instruction": "",
                "output": ""
            }
            if with_definition:
                ne_type = MSEQA_sample['tagName']
                # from string to dict
                gpt_definition = all_NEs_guidelines[ne_type]['gpt_answer'].strip()
                # gpt answer may have been truncated, ensure it ends by "} before evaluating to dict
                if not gpt_definition.endswith("}"):
                    if not gpt_definition.endswith("\""):
                        gpt_definition += "\""
                    gpt_definition += "}"
                this_ne_guidelines = eval(gpt_definition)
                # replacing ne types occurrences between single quotes to their UPPERCASE
                pattern = re.compile(rf'\'{re.escape(ne_type)}\'')
                this_ne_guidelines = {k: pattern.sub(f'{ne_type.upper()}', v) for k, v in this_ne_guidelines.items()}

                instruction = slimer_prompter.generate_prompt(ne_tag=ne_type.upper(),definition=this_ne_guidelines['Definition'], guidelines=this_ne_guidelines['Guidelines'])
                genQA_sample['instruction'] = instruction
            else:
                # same instruction but without Definition and Guidelines
                instruction_wo_guidelines = slimer_prompter.generate_prompt(ne_tag=MSEQA_sample['tagName'].upper(), definition="", guidelines="")
                genQA_sample['instruction'] = instruction_wo_guidelines

            # sorting the text answers by ascending starting positions to give the LLM a pattern: extract the occurences in the order they appear in the passage of text
            # this is because although the evaluation metrics are order independent the NTP loss penalizes order
            # we also delete duplicate occurrences thus obtaining a SET of gold_answers
            gold_answers_with_char_starts = MSEQA_sample['answers']
            # sort text answers by ascending start positions
            sorted_start_answers = sorted(zip(gold_answers_with_char_starts['answer_start'], gold_answers_with_char_starts['text']), key=lambda x: x[0])
            # retrieve only text answers
            sorted_answers_text_only = [item[1] for item in sorted_start_answers]
            # deleting any duplicate while preserving order (order within document context)
            sorted_textonly_gold_answers_wo_duplicates = list(OrderedDict.fromkeys(sorted_answers_text_only).keys())
            genQA_sample["output"] = json.dumps(sorted_textonly_gold_answers_wo_duplicates)  # stringifying list

            dataset_GenQA.append(genQA_sample)

        dataset_GenQA = Dataset.from_list(dataset_GenQA)

        dataset_GenQA.to_json(os.path.join(path_to_save_to, split_name + '.jsonl'))


def build_dataset_MSEQA_format_with_n_samples_per_NE_pos_neg(n_pos_samples_per_NE, n_neg_samples_per_NE, removeTestDatasetsNEs=True, keep_only_top_tagNames=-1):
    """
    build MSEQA dataset (default is FalseDef) with N positive samples per NE and M negative samples per NE
    train fold with N + M samples per NE
    validation fold with ceil(N/4) + ceil(M/4) samples per NE
    test fold is copied unchanged
    """
    dataset_MSEQA_format = build_dataset_MSEQA_format()
    dataset_MSEQA_format = remove_bad_ne_types(dataset_MSEQA_format)

    if removeTestDatasetsNEs:
        dataset_MSEQA_format = remove_MIT_CrossNER_NEs_from_train(dataset_MSEQA_format)

    # if keep_only_top_tagNames==391 or -1 we consider it already filtered with 391 NEs
    if keep_only_top_tagNames > -1 and keep_only_top_tagNames != 391:
        dataset_MSEQA_format = keep_only_top_N_tagNames(dataset_MSEQA_format, keep_only_top_tagNames)

    if n_pos_samples_per_NE == -1:
        return dataset_MSEQA_format

    print("Retaining N samples per NE ...")
    sys.stdout.flush()

    n_samples_per_NE_MSEQA_dataset = {split: [] for split in dataset_MSEQA_format.keys()}
    n_samples_per_NE_MSEQA_dataset['test'] = dataset_MSEQA_format['test']  # copy test fold unchanged
    for split in dataset_MSEQA_format.keys():
        # draw few samples only for train and validation
        if split != 'test':
            # count how many pos/neg samples we have per NE
            ne_list = {}
            for sample in dataset_MSEQA_format[split]:
                ne_type = sample['tagName']
                if ne_type not in ne_list:
                    ne_list[ne_type] = {'yes_answer': 0, 'no_answer': 0}
                if not sample['answers']['text']:
                    ne_list[ne_type]['no_answer'] += 1
                else:
                    ne_list[ne_type]['yes_answer'] += 1

            # if validation use 1/4 samples per NE
            if split == 'validation':
                n_pos_samples_per_NE = math.ceil(n_pos_samples_per_NE/4.0)
                n_neg_samples_per_NE = math.ceil(n_neg_samples_per_NE/4.0)
            ne_list = {ne: {'yes_answer': n_pos_samples_per_NE if values['yes_answer'] > n_pos_samples_per_NE else values['yes_answer'], 'no_answer': n_neg_samples_per_NE if values['no_answer'] > n_neg_samples_per_NE else values['no_answer']} for ne, values in ne_list.items()}

            for sample in dataset_MSEQA_format[split]:
                has_answer = 'yes_answer'
                if not sample['answers']['text']:
                    has_answer = 'no_answer'
                if ne_list[sample['tagName']][has_answer] > 0:
                    n_samples_per_NE_MSEQA_dataset[split].append(sample)
                    ne_list[sample['tagName']][has_answer] -= 1

            random.shuffle(n_samples_per_NE_MSEQA_dataset[split])

    return DatasetDict({split: Dataset.from_list(values) for split, values in n_samples_per_NE_MSEQA_dataset.items()})


def remove_MIT_CrossNER_NEs_from_train(datasetDict_QA):
    """
    removing from train fold all NEs that are in MIT and CrossNER to have True Zero-shot setting in test
    Most common tags person, location, country, organization not removed
    """
    print("Removing test sets named entities ...")
    sys.stdout.flush()

    tagName_to_remove_list = ["actor", "character", "genre", "song", "year",  # MOVIE, title left because polysemous
                              "dish", "restaurant",  # RESTAURANT
                              "algorithm", "field", "metric", "product", "programming language", "task", "university",  # AI
                              "award", "book", "event", "genre", "magazine",  # LITERATURE
                              "album", "award", "band", "artist", "instrument", "musical instrument", "music genre", "genre", "song",  # MUSIC
                              "event", "political party",   # POLITICS
                              "journal", "object", "chemical compound", "chemical", "element", "enzyme", "event",  # SCIENCE
                              "company", "legal"]  # BUSTER
    tagName_to_remove_list = list(set(tagName_to_remove_list))

    datasetDict_QA['train'] = datasetDict_QA['train'].filter(lambda sample: sample['tagName'] not in tagName_to_remove_list)
    datasetDict_QA['validation'] = datasetDict_QA['validation'].filter(lambda sample: sample['tagName'] not in tagName_to_remove_list)
    datasetDict_QA['test'] = datasetDict_QA['test'].filter(lambda sample: sample['tagName'] not in tagName_to_remove_list)

    return datasetDict_QA


def keep_only_top_N_tagNames(datasetDict_QA, top_N_tagNames):
    """
        Filters the dataset to keep only the samples which tagName is in the top_N_tagNames (most commons)

        Args:
            datasetDict_QA (HF.DatasetDict): DatasetDict with train, validation, test fields
            top_N_tagNames (int): retain N most common tagNames

        Returns:
            dict: filtered DatasetDict, test set left unchanged
    """
    number_samples_per_ne_type = Counter(sample['tagName'] for sample in datasetDict_QA['train'])
    sorted_tagNames_list = [tag for tag, _ in number_samples_per_ne_type.most_common()]
    valid_tagNames_list = sorted_tagNames_list[:top_N_tagNames]

    datasetDict_QA['train'] = datasetDict_QA['train'].filter(lambda sample: sample['tagName'] in valid_tagNames_list)
    datasetDict_QA['validation'] = datasetDict_QA['validation'].filter(lambda sample: sample['tagName'] in valid_tagNames_list)
    # test set is left unchanged

    return datasetDict_QA


def convert_MIT_CrossNER_test_sets_for_SLIMER_inference(dataset_name, path_to_dataset, with_definition, path_to_NE_guidelines_json, SLIMER_prompter_name='SLIMER_instruction_template'):
    """
    Converts MIT/CrossNER test sets for SLIMER inference format.

    This function converts datasets from the UniNER repository, which are in their specific format, into SLIMER format.
    It modifies the column names and adds Named Entity (NE) guidelines as instructions to the dataset.
    Specifically, it changes the 'document_context' column to 'input' and the 'answers' column to 'output'.
    Optionally, the function also includes Definitions and Guidelines if specified.

    Args:
        dataset_name (str): The name of the dataset to be converted. Examples include 'movie', 'restaurant',
            'ai', 'music', 'literature', 'science', 'politics'.
        path_to_dataset (str): The file path to the JSON dataset file to be converted.
        with_definition (bool): Specifies whether to include Definitions and Guidelines in the instruction.
            If True, Definitions and Guidelines will be added to the dataset.
        path_to_NE_guidelines_json (str): The file path to the JSON file containing the NE Guidelines.
            This is required even if `with_definition` is set to False.

    Returns:
        dict: A Dataset representing the converted dataset in SLIMER format, with modified 'input', 'output',
            and 'instruction' fields.

    Raises:
        FileNotFoundError: If the dataset or NE guidelines file cannot be found at the provided paths.
        ValueError: If the dataset name is invalid or not supported.
    """

    slimer_prompter = SLIMER_instruction_prompter(SLIMER_prompter_name, './src/SFT_finetuning/templates')

    try:
        with open(path_to_dataset, 'r') as fh:
            uniNER_eval_samples = json.load(fh)
    except FileNotFoundError:
        raise FileNotFoundError(f"{dataset_name} json file not found at {path_to_dataset}")

    # we load guidelines also if with_def False to make NE mapping to canonical names (uniNER eval NEs are different)
    try:
        with open(path_to_NE_guidelines_json, 'r') as file:
            all_NEs_guidelines = json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Definition and Guidelines file not found at {path_to_NE_guidelines_json}, required also if D&G set to False")

    # converting list to dict for fast key/NE access
    if all_NEs_guidelines and isinstance(all_NEs_guidelines, list):
        all_NEs_guidelines = {x['named_entity']: x for x in all_NEs_guidelines}

    dataset_GenQA = []  # dataset being constructed
    for uniNER_sample in uniNER_eval_samples:

        context, questions_answers_list = extract_context_quests_answers(uniNER_sample['conversations']).values()

        if len(questions_answers_list) > 1:
            raise ValueError("Expected only 1 question")

        question, ne_type, answers = questions_answers_list[0].values()

        # some uniNER NEs are different from the original NEs
        try:
            gpt_definition = all_NEs_guidelines[ne_type]['gpt_answer'].strip()
            # NE name in natural languange form, e.g. ORG --> organization
            real_name_ne = all_NEs_guidelines[ne_type]['real_name']
        except KeyError:
            if dataset_name in ['ai', 'literature', 'science', 'politics', 'music']:
                ne_mapping = {
                    'organization': 'organisation',
                    'program language': 'programlang',
                    'literary genre': 'literarygenre',
                    'astronomical object': 'astronomicalobject',
                    'chemical element': 'chemicalelement',
                    'chemical compound': 'chemicalcompound',
                    'academic journal': 'academicjournal',
                    'political party': 'politicalparty',
                    'musical artist': 'musicalartist',
                    'musical instrument': 'musicalinstrument',
                    'music genre': 'musicgenre',
                }
            elif dataset_name == 'movie':
                ne_mapping = {
                    'character': 'CHARACTER',
                    'plot': 'PLOT',
                    'year': 'YEAR',
                    'director': 'DIRECTOR',
                    'rating': 'RATING',
                    'average ratings': 'RATINGS_AVERAGE',
                    'actor': 'ACTOR',
                    'genre': 'GENRE',
                    'song': 'SONG',
                    'trailer': 'TRAILER',
                    'review': 'REVIEW',
                    'title': 'TITLE'
                }
            elif dataset_name == 'restaurant':
                ne_mapping = {
                    'amenity': 'Amenity',
                    'location': 'Location',
                    'cuisine': 'Cuisine',
                    'restaurant name': 'Restaurant_Name',
                    'rating': 'Rating',
                    'hours': 'Hours',
                    'price': 'Price',
                    'dish': 'Dish'
                }
            ne_type = ne_mapping[ne_type]
            gpt_definition = all_NEs_guidelines[ne_type]['gpt_answer'].strip()
            real_name_ne = all_NEs_guidelines[ne_type]['real_name']

        # gpt answer may have been truncated, ensure it ends by "} before evaluating to dict
        if not gpt_definition.endswith("}"):
            if not gpt_definition.endswith("\""):
                gpt_definition += "\""
            gpt_definition += "}"

        this_ne_guidelines = eval(gpt_definition)
        # replacing ne types occurrences between single quotes to their UPPERCASE
        ne_type_in_natural_language = all_NEs_guidelines[ne_type]['real_name']
        pattern = re.compile(rf'\'{re.escape(ne_type_in_natural_language)}\'')
        this_ne_guidelines = {k: pattern.sub(f'{ne_type_in_natural_language.upper()}', v) for k, v in this_ne_guidelines.items()}

        if with_definition:
            question = slimer_prompter.generate_prompt(ne_tag=ne_type_in_natural_language.upper(), definition=this_ne_guidelines['Definition'],guidelines=this_ne_guidelines['Guidelines'])
        else:
            question = slimer_prompter.generate_prompt(ne_tag=ne_type_in_natural_language.upper(), definition="", guidelines="")

        genQA_sample = {
            "doc_tag_pairID": uniNER_sample['id'],
            "input": context,
            "tagName": ne_type,
            "instruction": question,
            "output": uniNER_sample['conversations'][-1]['value']
        }
        dataset_GenQA.append(genQA_sample)

    return Dataset.from_list(dataset_GenQA)


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
        ne_type_in_natural_language = ne_tag
        pattern = re.compile(rf'\'{re.escape(ne_type_in_natural_language)}\'', re.IGNORECASE)
        this_ne_guidelines = {k: pattern.sub(f'{ne_type_in_natural_language.upper()}', v) for k, v in this_ne_guidelines.items()}
        values['gpt_answer'] = this_ne_guidelines

    return DeG_per_NEs_raw

def build_dataset_SLIMER_PARALLEL_format(top_391_NEs_list, max_tagNames_per_prompt=5, path_to_DeG=""):
    print("Downloading PileNER dataset from HF...")
    sys.stdout.flush()
    # downloading raw dataset from huggingface repo (has only "train" partition)
    raw_dataset = load_dataset("Universal-NER/Pile-NER-type")

    tagName_to_remove_list = ["actor", "character", "genre", "song", "year",  # MOVIE, title left because polysemous
                              "dish", "restaurant",  # RESTAURANT
                              "algorithm", "field", "metric", "product", "programming language", "task", "university",  # AI
                              "award", "book", "event", "genre", "magazine",  # LITERATURE
                              "album", "award", "band", "artist", "instrument", "musical instrument", "music genre",
                              "genre", "song",  # MUSIC
                              "event", "political party",  # POLITICS
                              "journal", "object", "chemical compound", "chemical", "element", "enzyme", "event",  # SCIENCE
                              "company", "legal"]  # BUSTER
    tagName_to_remove_list = list(set(tagName_to_remove_list))

    if path_to_DeG:
        DeG_per_NEs = load_DeG_per_NEs(path_to_DeG)
    slimer_parallel_prompter = SLIMER_PARALLEL_instruction_prompter("SLIMER_PARALLEL_instruction_template", '../SFT_finetuning/templates')

    samples = []
    samples_progressiveID = 0
    for raw_sample in raw_dataset['train']['conversations']:
        # extract context and list of question-goldAnswers associated to each context
        context, questions_answers_list = extract_context_quests_answers(raw_sample).values()

        answers_per_tagName_dict = {}
        at_least_one_positive = False
        for q_a in questions_answers_list:
            tagName = q_a["ne_type"]
            if tagName in top_391_NEs_list and tagName not in tagName_to_remove_list:
                if len(answers_per_tagName_dict.keys()) < max_tagNames_per_prompt:
                    answers_per_tagName_dict[tagName.upper()] = list(set(q_a["answers"]['text']))
                    if q_a["answers"]['text']:
                        at_least_one_positive = True

        tagNames_DeG = ""
        if path_to_DeG:
            tagNames_DeG = {}
            for ne_tag in answers_per_tagName_dict.keys():
                tagNames_DeG[ne_tag] = DeG_per_NEs[ne_tag.lower()]['gpt_answer']
            tagNames_DeG = json.dumps(tagNames_DeG, indent=2)

        instruction = slimer_parallel_prompter.generate_prompt(ne_tags=", ".join(answers_per_tagName_dict.keys()), def_and_guidelines=tagNames_DeG)

        # add the sample only if there are some tagNames and are not all []
        if at_least_one_positive:
            samples.append({
                "id": samples_progressiveID,
                "input": context,
                "instruction": instruction,
                "output": json.dumps(answers_per_tagName_dict, indent=2)
            })
            samples_progressiveID += 1

    train_ratio = 0.9
    num_samples = len(samples)
    num_train = int(train_ratio * num_samples)
    train_fold = samples[:num_train]
    val_test_fold = samples[num_train:]

    val_fold = val_test_fold[:math.floor(len(val_test_fold) / 2.0)]
    test_fold = val_test_fold[math.floor(len(val_test_fold) / 2.0):]

    random.seed(42)
    random.shuffle(train_fold)
    random.shuffle(val_fold)
    random.shuffle(test_fold)

    train_dataset = Dataset.from_list(train_fold)
    validation_dataset = Dataset.from_list(val_fold)
    test_dataset = Dataset.from_list(test_fold)

    return DatasetDict({
        "train": train_dataset,
        "validation": validation_dataset,
        "test": test_dataset
    })


if __name__ == "__main__":

    from src.SFT_finetuning.commons.basic_utils import load_json
    top_391_NEs_list = list(load_json("./questions/pileNER/top391NEs_definitions.json").keys())
    print(top_391_NEs_list)

    datasetDict_SLIMER_PARALLEL_format = build_dataset_SLIMER_PARALLEL_format(top_391_NEs_list, max_tagNames_per_prompt=5, path_to_DeG="./questions/pileNER/top391NEs_definitions.json")
    print(datasetDict_SLIMER_PARALLEL_format)
    print(datasetDict_SLIMER_PARALLEL_format['train'])
    print(datasetDict_SLIMER_PARALLEL_format['train'][0])
    print(datasetDict_SLIMER_PARALLEL_format['train'][1])
    print(datasetDict_SLIMER_PARALLEL_format['train'][11]['instruction'])
    print(datasetDict_SLIMER_PARALLEL_format['train'][11]['input'])
    print(datasetDict_SLIMER_PARALLEL_format['train'][11]['output'])

    datasetDict_SLIMER_PARALLEL_format['train'].to_json("../../data/pileNER/pileNER_SLIMER_PARALLEL_format_train.jsonl")

    """

    dataset_MSEQA_format_with_n_samples_per_NE_FalseDef = build_dataset_MSEQA_format_with_n_samples_per_NE_pos_neg(
        n_pos_samples_per_NE=5,
        n_neg_samples_per_NE=5,
        removeTestDatasetsNEs=True,
        keep_only_top_tagNames=391
    )
    print("\nTrain tagName list:")
    ne_list = {}
    for sample in dataset_MSEQA_format_with_n_samples_per_NE_FalseDef['train']:
        ne_type = sample['tagName']
        if ne_type in ne_list:
            ne_list[ne_type] += 1
        else:
            ne_list[ne_type] = 1
    # print(len(ne_list.items()))
    print(sorted(ne_list.items(), key=lambda x: x[1], reverse=True))

    convert_MSEQA_dataset_to_GenQA_format_SI(
        dataset_MSEQA_format=dataset_MSEQA_format_with_n_samples_per_NE_FalseDef,
        with_definition=True,
        path_to_NE_guidelines_json="./questions/pileNER/top391NEs_definitions.json",
        path_to_save_to='../../data/pileNER/5pos_5neg_perNE_top391NEs_TrueDef'
    )

    path_to_dataset = "../../data/pileNER/5pos_5neg_perNE_top391NEs_TrueDef/train.jsonl"
    pileNER_subset_train = load_dataset("json", data_files=path_to_dataset)
    print(pileNER_subset_train)
    print(pileNER_subset_train['train'][0])
    
    """

    """
    crossNER_test_Dataset = convert_MIT_CrossNER_test_sets_for_SLIMER_inference(
        dataset_name="ai",
        path_to_dataset="../../data/eval_data_UniNER/test_data/CrossNER_AI.json",
        with_definition=True,
        path_to_NE_guidelines_json="./questions/crossNER/gpt_guidelines/ai_NE_definitions.json"
    )

    print(crossNER_test_Dataset)
    print(crossNER_test_Dataset[0])
    """