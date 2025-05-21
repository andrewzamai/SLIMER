from src.SFT_finetuning.evaluating.evaluate_SLIMER_PARALLEL_GRPO import parse_json_pred

if __name__ == "__main__":
   
   response = "<think>\n" + "list empty.\n\n</think>\n\n{\n  \"CONFERENCE\": [],\n  \"METRICS\": [],\n  \"COUNTRY\": [],\n  \"UNIVERSITY\": [],\n  \"PERSON\": [],\n  \"TASK\": [],\n  \"RESEARCHER\": [],\n  \"PRODUCT\": [\"AIBO\"],\n  \"ALGORITHM\": [],\n  \"FIELD\": [\"artificial intelligence\", \"Computer vision\"],\n  \"ORGANIZATION\": [],\n  \"LOCATION\": [],\n  \"PROGRAMMING LANGUAGE\": []\n}"

   print(response)

   parsed_gold_output, parsed_response, all_good_parsing = parse_json_pred(None, response)

   print(parsed_response)