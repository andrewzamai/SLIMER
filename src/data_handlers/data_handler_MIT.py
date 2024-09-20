"""
data_handler for MIT dataset: movie and restaurant subdatasets
"""

from datasets import Dataset, DatasetDict, load_dataset
import json
import os
import re

# ABSTRACT class which inherits from
from src.data_handlers.Data_Interface import Data_Interface

class MIT(Data_Interface):

    def MIT_read_bio_file(self, path_to_bio_txt, ds_name, split_name):
        sentences = self.read_bio_file(path_to_bio_txt, ds_name, split_name)
        for sentence in sentences:
            sentence['tokens'], sentence['labels'] = sentence['labels'], sentence['tokens']
        return sentences

    def load_datasetdict_BIO(self, path_to_BIO, test_only=True):

        subdataset = path_to_BIO.split('/')[-1]
        if test_only:
            splits = ['test']
        else:
            splits = ['train', 'test']

        return DatasetDict({split: Dataset.from_list(self.MIT_read_bio_file(os.path.join(path_to_BIO, split+'.txt'), subdataset, split)[0:100]) for split in splits})

    def get_map_to_extended_NE_name(self):

        remap_to = {
            'RATINGS_AVERAGE': 'RATINGS AVERAGE',
            'Restaurant_Name': 'Restaurant Name'
        }

        return {ne: remap_to.get(ne, ne) for ne in self.get_ne_categories()}


if __name__ == '__main__':

    subdataset_name = "movie"
    MIT_handler = MIT(
        f"../../data/MIT/{subdataset_name}",
        test_only=True,
        path_to_templates='../SFT_finetuning/templates',
        SLIMER_prompter_name='SLIMER_instruction_template',
        path_to_DeG=f'./questions/MIT/gpt_guidelines/{subdataset_name}_NE_definitions.json'
    )

    print(MIT_handler.get_ne_categories())
    #print(MIT_handler.get_dataset_statistics())

    # print(CrossNER_handler.datasetdict_BIO)

    MIT_handler.dataset_dict_SLIMER['test'].to_json(f'../../data/MIT/SLIMER/{subdataset_name}_test.json')
    #print(CrossNER_handler.convert_dataset_for_SLIMER()['test'][1]['instruction'])
    #print(CrossNER_handler.convert_dataset_for_SLIMER()['test'][1]['output'])




