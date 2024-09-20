"""
data_handler for CrossNER dataset: ai, literature, music, politics, science

"""

from datasets import Dataset, DatasetDict, load_dataset
import json
import os
import re

# ABSTRACT class which inherits from
from src.data_handlers.Data_Interface import Data_Interface

class CrossNER(Data_Interface):

    def load_datasetdict_BIO(self, path_to_BIO, test_only=True):

        subdataset = path_to_BIO.split('/')[-1]
        if test_only:
            splits = ['test']
        else:
            splits = ['train', 'dev', 'test']

        return DatasetDict({split: Dataset.from_list(self.read_bio_file(os.path.join(path_to_BIO, split+'.txt'), subdataset, split)) for split in splits})

    def get_map_to_extended_NE_name(self):

        remap_to = {
            'programlang': 'programming language',

            'musicalartist': 'musical artist',
            'musicalinstrument': 'musical instrument',
            'musicgenre': 'music genre',

            'literarygenre': 'literary genre',

            'politicalparty': 'political party',

            'academicjournal': 'academic journal',
            'astronomicalobject': 'astronomical object',

            'chemicalcompound': 'chemical compound',
            'chemicalelement': 'chemical element'
        }

        return {ne: remap_to.get(ne, ne) for ne in self.get_ne_categories()}


if __name__ == '__main__':

    CrossNER_handler = CrossNER(
        "../../data/CrossNER/ner_data/ai",
        test_only=True,
        path_to_templates='../SFT_finetuning/templates',
        SLIMER_prompter_name='SLIMER_instruction_template',
        path_to_DeG='./questions/crossNER/gpt_guidelines/ai_NE_definitions.json'
    )

    print(CrossNER_handler.get_ne_categories())
    print(CrossNER_handler.get_dataset_statistics())

    # print(CrossNER_handler.datasetdict_BIO)

    CrossNER_handler.dataset_dict_SLIMER['test'].to_json('../../data/CrossNER/SLIMER/ai_test.json')
    #print(CrossNER_handler.convert_dataset_for_SLIMER()['test'][1]['instruction'])
    #print(CrossNER_handler.convert_dataset_for_SLIMER()['test'][1]['output'])




