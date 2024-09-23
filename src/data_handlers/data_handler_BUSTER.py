"""
data_handler for BUSTER dataset: a document-level NER dataset with focus on the main actors involved in financial transactions

"""

from datasets import Dataset, DatasetDict, load_dataset
import json
import os
import re

# ABSTRACT class which inherits from
from src.data_handlers.Data_Interface import Data_Interface

class BUSTER(Data_Interface):

    def load_datasetdict_BIO(self, path_to_BIO, test_only=True):
        # in SLIMER paper we use the fold5 as test dataset
        fold5_test = load_dataset("expertai/BUSTER")['FOLD_5']
        fold5_test = fold5_test.rename_column("document_id", "id")

        return DatasetDict({"test": fold5_test})

    def get_map_to_extended_NE_name(self):

        return {
            "Advisors.GENERIC_CONSULTING_COMPANY": "generic consulting company",
            "Advisors.LEGAL_CONSULTING_COMPANY": "legal consulting company",
            "Generic_Info.ANNUAL_REVENUES": "annual revenues",
            "Parties.ACQUIRED_COMPANY": "acquired company",
            "Parties.BUYING_COMPANY": "buying company",
            "Parties.SELLING_COMPANY": "selling company"
        }


if __name__ == '__main__':

    BUSTER_handler = BUSTER(
        "expertai/BUSTER",
        path_to_templates='../SFT_finetuning/templates',
        SLIMER_prompter_name='SLIMER_instruction_template',
        path_to_DeG='./questions/BUSTER/gpt_guidelines/BUSTER_NE_definitions.json'
    )

    print(BUSTER_handler.get_dataset_statistics())

    print(BUSTER_handler.dataset_dict_SLIMER['test'])
    print(BUSTER_handler.dataset_dict_SLIMER['test'][1]['instruction'])
    print(BUSTER_handler.dataset_dict_SLIMER['test'][1]['output'])




