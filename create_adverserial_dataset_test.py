import re

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from utils.defaults import *

model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        trust_remote_code=True
    )
model.to(DEVICE)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def extract_answer(text: str) -> str:
    """Extract numerical answer from model response."""
    if '####' in text:
        parts = text.split('####')
        answer_part = parts[-1].strip()
        numbers = re.findall(r'-?\d+\.?\d*', answer_part)
        if numbers:
            return numbers[0]

    numbers = re.findall(r'-?\d+\.?\d*', text)
    if numbers:
        return numbers[-1]
    return None

def get_math_inference(prompt, model_to_use=model):
    # Generate response
    inputs = tokenizer(prompt, return_tensors="pt").to(model_to_use.device)
    print("using settings max_new_tokens=256, temperature=0.7,top_p=0.9,do_sample=True")
    outputs = model_to_use.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )

    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)

    answer = extract_answer(response)

    return response, answer


def ask_a_math_question(question, model_to_use=None):
    if model_to_use is None:
        model_to_use = model
        print("No model to use provided, using default model")

    prompt = f"""Solve this math problem step by step:

            {question}

            Provide your final answer in the format:
            [reasoning steps]
            ####
            [final answer (just the number)]"""
    # prompt = question

    print("Core module: prompt", prompt)

    response, answer = get_math_inference(prompt, model_to_use)

    print("Core module: response, answer", response, answer)

    return response, answer


if __name__ == "__main__":
    # question = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
    # question = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. Natalia also sold 49 clips on June. How many clips did Natalia sell altogether in April and May?"
    # question = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
    question = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. Natalia also sold 49 clips on June. Meanwhile, 12 of her friends bought 3 clips each, 7 friends bought 5 clips each, and 9 friends bought 2 clips each. How many clips did Natalia sell altogether in April and May?"
#     # question = "In April, Natalia distributed a certain number of clips among her friends. The following month, her total clip sales dropped to exactly one-half of what she had sold in April. If the combined number of clips sold over these two months is to be determined, given that she sold clips to 48 friends in April (one clip per friend), what is the total number of clips she sold across both months?"
#
#     # question = "Natalia was tracking her monthly distribution of clips, noting that each recipient received exactly one clip. During one particular spring month, the number of recipients she reached was 48. In the following month, her outreach reduced such that she only distributed clips to a group equal to half the size of the previous month’s recipients. Without directly computing month-wise totals separately, determine the aggregate number of clips she must have distributed across both months."
#
#     question = """natalia had 48 “interactons” in april (1 clip each but she didnt write clips count). next month was like… take april ppl, split in 2 same groups, she only did 1 of those. no repeats, no extra clip stuff (ignore random nums she wrote like 96, 24 etc).
#
# total clips across both ??"""

    prompt = f"""Solve this math problem step by step:

        {question}

        Provide your final answer in the format:
        [reasoning steps]
        ####
        [final answer (just the number)]"""

    print("asking", prompt)
    response, answer = get_math_inference(prompt)

    print("response", response)
    print("answer:", answer)