from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList
from transformers import LogitsProcessor, LogitsProcessorList
import time
import argparse
import torch
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--pass_at", type=int, default=1)
FLAGS = parser.parse_args()

stop_words = ["\n\n", ("\n","\n")]
assert_stop_words = ["assert"] + stop_words
eof_id = 50256
eof_token = "<|endoftext|>"
imports = "\nimport math\nfrom typing import List\n"
checking_end = "    pass\n\n# Assume the above function is completed. Write 3 lines of testing code for the function.\n\nassert"
# 0
prompt_0 = "from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n"
# 31
prompt_31 = "\n\ndef is_prime(n):\n    \"\"\"Return true if a given number is prime, and false otherwise.\n    >>> is_prime(6)\n    False\n    >>> is_prime(101)\n    True\n    >>> is_prime(11)\n    True\n    >>> is_prime(13441)\n    True\n    >>> is_prime(61)\n    True\n    >>> is_prime(4)\n    False\n    >>> is_prime(1)\n    False\n    \"\"\"\n"
# 35
prompt_35 = "\n\ndef max_element(l: list):\n    \"\"\"Return maximum element in the list.\n    >>> max_element([1, 2, 3])\n    3\n    >>> max_element([5, 3, -5, 2, -3, 3, 9, 0, 123, 1, -10])\n    123\n    \"\"\"\n"
# 161
prompt_161 = "\ndef solve(s):\n    \"\"\"You are given a string s.\n    if s[i] is a letter, reverse its case from lower to upper or vise versa, \n    otherwise keep it as it is.\n    If the string contains no letters, reverse the string.\n    The function should return the resulted string.\n    Examples\n    solve(\"1234\") = \"4321\"\n    solve(\"ab\") = \"AB\"\n    solve(\"#a@C\") = \"#A@c\"\n    \"\"\"\n"

def trim_substring_from_end(answer, b):
    while answer.endswith(b):
        answer = answer[:-len(b)]
    return answer

def trim_with_stopwords(outputs, stopwords, original_prompt) -> str:
    trimmed = False
    result = []
    len_prompt = len(original_prompt)
    for output in outputs:
        answer = output[len_prompt:]
        answer = answer.lstrip('\n')
        min_i = len(answer)
        for w in sorted(stopwords, reverse=True):
            for i in range(len(answer)):
                if answer[i:].startswith(w) and min_i > i:
                    min_i = i
                    trimmed = True
        answer = answer[:min_i]
        result.append(answer)
    if not trimmed:
        print("This question is not trimmed!")
    return result


def main(args):
    pass_at = args.pass_at
    
    # Initialize model
    # "Salesforce/codegen-[350M, 2B, 6B, 16B]-mono"
    checkpoint = "Salesforce/codegen-6B-mono"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, padding_side='left', device_map="auto")
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Loading original model...")
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, 
        load_in_8bit=False,
        # load_in_4bit=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    
    # Stopping criteria for generation using the LogitsProcessor class
    class StopSequences(LogitsProcessor):
        def __init__(self, stop_sequences, batch_size, encounters=1, eos_token_id=50256):
            StoppingCriteria.__init__(self)
            self.stop_sequences = tokenizer.batch_encode_plus(stop_sequences)['input_ids']
            self.batch_size = batch_size
            self.encounters = [encounters] * batch_size
            self.NUM_ENCOUNTERS = encounters
            self.eos_token_id = eos_token_id
            self.just_started = True

        def __call__(self, input_ids, scores):
            if self.just_started:
                self.just_started = False
                return scores
            forced_eos = torch.full((scores.size(1),), -float("inf"))
            forced_eos[self.eos_token_id] = 0
            for stop in self.stop_sequences:
                # Check if the input_ids end with the stop sequence
                for i in range(self.batch_size):
                    if self.encounters[i] <= 0:
                        continue
                    if input_ids[i][-len(stop):].tolist() == stop:
                        self.encounters[i] -= 1
                        if self.encounters[i] <= 0:
                            scores[i] = forced_eos
            return scores
    
    all_start = time.time()
    for prompt in [prompt_0, prompt_31, prompt_35, prompt_161]:
        final_array = [prompt]*pass_at
        prompt_ids = tokenizer.batch_encode_plus(final_array, return_tensors="pt", padding=True).to(torch.cuda.current_device())
        logits_processor = LogitsProcessorList([StopSequences(assert_stop_words, batch_size=args.pass_at, encounters=1)])
        start = time.time()
        with torch.no_grad():
            if pass_at <= 1:
                answer_ids = model.generate(
                    **prompt_ids,
                    use_cache = True,
                    pad_token_id = tokenizer.pad_token_id,
                    eos_token_id = tokenizer.eos_token_id,
                    max_new_tokens = 1024,
                    do_sample = False,
                    num_beams = 1,
                    logits_processor = logits_processor
                )
            else:
                answer_ids = model.generate(
                    **prompt_ids,
                    use_cache = True,
                    # padding=True,
                    # truncation=True,
                    # padding_side='left',
                    pad_token_id = tokenizer.eos_token_id,
                    max_new_tokens = 1024,
                    do_sample = True,
                    top_k = 0,
                    top_p = 0.95,
                    temperature = 0.8,
                    num_beams = 1,
                    num_return_sequences = pass_at,
                    logits_processor = logits_processor
                )
        end = time.time()
        answer_ids = answer_ids[:, len(prompt_ids['input_ids'][0]):]
        answer_text = tokenizer.batch_decode(answer_ids, skip_special_tokens=False)
        answer_trimmed = [trim_substring_from_end(answer, eof_token) for answer in answer_text]
        all_num_ids = [len(answer_id) for answer_id in answer_ids]
        print(f"answer_ids has length {all_num_ids}")
        for j in range(len(answer_trimmed)):
            print(f"answer {j} is:\n{answer_trimmed[j]}\n")
        print(f"Time took {end - start} seconds")
    print(f"In total, took {time.time() - all_start} seconds")
    

if __name__== "__main__":
    main(FLAGS)