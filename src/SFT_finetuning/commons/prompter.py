"""
A dedicated helper to manage templates and prompt building.
"""

import json
import os.path as osp
from typing import Union


class Prompter(object):
    __slots__ = ("template", "template_path", "eos_text", "_verbose")

    def __init__(self, template_name: str = "", template_path: str = "templates", eos_text: Union[None, str] = None, verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            # Enforce the default here, so the constructor can be called with '' and will not break.
            template_name = "alpaca"
        file_name = osp.join(template_path, f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)

        self.eos_text = eos_text

        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            # res = self.template["prompt_input"].format(
            #     instruction=instruction, input=input
            # )
            res = self.template["prompt_input"].replace(
                "{instruction}", instruction).replace("{input}", input)
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
            if self.eos_text:
                res = f"{res} {self.eos_text}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()


if __name__ == '__main__':

    from datasets import load_dataset
    path_to_dataset = "../../../data/pileNER/5pos_5neg_perNE_top391NEs_TrueDef/train.jsonl"
    data = load_dataset("json", data_files=path_to_dataset)
    sample = data['train'][1]

    """
    from src.data_handlers import data_handler_pileNER
    data = data_handler_pileNER.convert_MIT_CrossNER_test_sets_for_SLIMER_inference(
        'ai',
        '../../../data/eval_data_UniNER/test_data/CrossNER_AI.json',
        path_to_NE_guidelines_json='../../data_handlers/questions/crossNER/gpt_guidelines/ai_NE_definitions.json',
        with_definition=True
    )
    sample = data[1]
    """

    prompt = Prompter("LLaMA2-chat", template_path="../templates").generate_prompt(
        instruction=sample['instruction'],
        input=sample['input'],
        label=sample['output']
    )

    #print(json.dumps(prompt))
    print(prompt)





