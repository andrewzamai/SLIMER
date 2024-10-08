"""
Data handler for NuNER dataset.
"""

from datasets import load_dataset
import re

from src.SFT_finetuning.commons.basic_utils import save_json, load_json

def parse_NuNER_sample(sample):
    """
    Parse the output field of a sample to extract the list of text_span, tagName, [tag_description]
    """
    try:
        if 'output' in sample:
            output = eval(sample['output'])  # eval str to dict
        else:
            output = eval(sample)
    except:
        return []

    data_list = []
    for out in output:
        # split on <>
        out_splitted = out.split('<>')
        try:
            data = {
                'text_span': out_splitted[0].strip(),
                'tagName': out_splitted[1].strip().lower()  # lower case the labels
            }
            # if 3 fields the last is a brief tag description
            if len(out_splitted) == 3:
                tag_description = out_splitted[2].strip()
                # add . at the end
                if tag_description[-1] != '.':
                    tag_description += '.'
                data['tag_description'] = lowercase_first_word(tag_description)
            else:
                data['tag_description'] = ''
            data_list.append(data)
        except:
            pass

    return data_list

def lowercase_first_word(s):
    words = s.split(' ', 1)  # Split only on the first space
    if words:
        words[0] = words[0].lower()  # Set the first word to lowercase
    return ' '.join(words)


# apply parse_NuNER_sample using the map function
def extract_tags(batch):
    # initialize lists to store the flattened data
    flattened_input = []
    flattened_text_span = []
    flattened_tagName = []
    flattened_tag_description = []
    flattened_input_output_len = []

    # Iterate through each example in the batch
    for i in range(len(batch['input'])):
        parsed_data = parse_NuNER_sample(batch['output'][i])

        if parsed_data:
            for data in parsed_data:
                if batch['input'][i] and data['tagName'] and data['tag_description']:
                    if not has_too_many_punctuations(batch['input'][i], threshold=0.3):
                        input_text = batch['input'][i]  # original input for this example
                        text_span = data['text_span']  # extracted text span
                        tag_name = data['tagName']  # tag name
                        tag_description = data['tag_description']  # tag description
                        input_output_len = len(input_text.split()) + len(tag_description.split())

                        flattened_input.append(input_text)
                        flattened_text_span.append(text_span)
                        flattened_tagName.append(tag_name)
                        flattened_tag_description.append(tag_description)
                        flattened_input_output_len.append(input_output_len)

    return {
        'input': flattened_input,
        'text_span': flattened_text_span,
        'tagName': flattened_tagName,
        'tag_description': flattened_tag_description,
        'input_output_len': flattened_input_output_len
    }


def has_too_many_punctuations(input_string, threshold=0.3):
    # Regular expression to match all punctuation characters
    punctuations_regex = r'[!"#$%&\'()*+,\-./:;<=>?@[\\\]^_`{|}~]'

    # Count the total number of characters
    total_chars = len(input_string)

    # Count the number of punctuation characters
    punctuation_count = len(re.findall(punctuations_regex, input_string))

    # Calculate the ratio of punctuation characters to total characters
    if total_chars > 0 and (punctuation_count / total_chars) > threshold:
        return True  # Too many punctuations
    return False  # It's fine

def extract_tagNames_list(dataset, min_num_occurrences=500):
    tagNames = {}
    for sample in dataset:
        gold_spans = parse_NuNER_sample(sample)
        for gs in gold_spans:
            if gs['tagName'] in tagNames:
                tagNames[gs['tagName']] += 1
            else:
                tagNames[gs['tagName']] = 1
    print(len(tagNames))
    ne_types = [a[0] for a in sorted(tagNames.items(), key=lambda item: item[1], reverse=True) if a[1] >= min_num_occurrences]
    # return dict(sorted(tagNames.items(), key=lambda item: item[1], reverse=True)[0:500])
    return ne_types


def format_chat_template(row):
    system_message = "You are an expert in Named Entity Recognition."
    user_provides_text_instruction = "You are given a text chunk (delimited by triple quotes) and an instruction. Read the text and answer to the instruction in the end.\n\"\"\"\n{input}\n\"\"\"\nInstruction: {instruction}"
    instruction = "The span \"{text_span}\" belongs to the class \"{tag_name}\". What is the motivation for this classification?\nThe span \"{text_span}\" is classified as \"{tag_name}\" because it represents {initial_completion_words}."
    initial_completion_words = " ".join(row['tag_description'].split()[:2]) + '..'
    completion_words = "..." + " ".join(row['tag_description'].split()[2:])
    instruction = instruction.format(text_span=row['text_span'], tag_name=row['tagName'], initial_completion_words=initial_completion_words)
    row_json = [{"role": "system", "content": system_message},
                {"role": "user", "content": user_provides_text_instruction.format(input=row['input'], instruction=instruction)},
                {"role": "assistant", "content": completion_words}]
    row["conv_text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
    return row


if __name__ == '__main__':

    """
    datasetDict = load_dataset("numind/NuNER")
    print(datasetDict)

    # select FULL partition with brief tag_description
    dataset = datasetDict['full']

    #top_tagNames_list = extract_tagNames_list(dataset, min_num_occurrences=500)
    #print(len(top_tagNames_list))
    #save_json(top_tagNames_list, "../../data/NuNER/top_tagNames_list.json")

    dataset_with_tags = dataset.map(extract_tags, batched=True, remove_columns=['output'])
    print(dataset_with_tags)
    print(dataset_with_tags[0])
    #dataset_with_tags.to_json("../../data/NuNER/NuNER_completion.json")

    # remove empty rows from non-parsable samples
    #dataset_with_tags = dataset_with_tags.filter(lambda example: example['tagName'] is not None)

    dataset_with_tags = dataset_with_tags.sort('input_output_len', reverse=True)
    print(dataset_with_tags)
    print(dataset_with_tags[0:10])

    # retain only top 200K samples
    dataset_with_tags = dataset_with_tags.select(range(200000))

    top_tagNames_list = load_json("../../data/NuNER/top_tagNames_list.json")
    dataset_with_tags = dataset_with_tags.filter(lambda example: example['tagName'] in top_tagNames_list)

    # retain only top 100K samples
    dataset_with_tags = dataset_with_tags.select(range(min(100000, len(dataset_with_tags))))

    print(dataset_with_tags)
    print(dataset_with_tags[0:10])

    dataset_with_tags.to_json("../../data/NuNER/NuNER_completion_100K.json")
    
    """

    dataset = load_dataset('json', data_files="../../data/NuNER/NuNER_completion_100K.json", split='train')
    print(dataset)
    print(dataset[0])

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

    dataset_conv_format = dataset.map(format_chat_template)
    print(dataset_conv_format['conv_text'][0])
    print(dataset_conv_format['conv_text'][10])

