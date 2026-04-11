"""
1. Make a paraphrase maker
2. run sft

"""
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

from pandas import Flags

from create_adverserial_dataset_test import ask_a_math_question

DEEP_SEEK_API_KEY = str(os.getenv("DEEP_SEEK_API_KEY"))

client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=DEEP_SEEK_API_KEY,
        )

def make_story_by_calling_genai(prompt: str, history):
    for retry in range(5):
        try:
            completion = client.chat.completions.create(
                # extra_headers={
                #   "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
                #   "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
                # },
                model="openai/gpt-oss-20b:free",
                messages=history
            )
            # print(f"Made completion: {completion}")
            story_raw = completion.choices[0].message.content

            return story_raw
        except Exception as error:
            print("Got error while making LLM call", str(error))
            continue
    return None


def get_paraphrase(question):
    prompt = f"""QUESTION: "{question}"

Instructions: Add noise to the QUESTION such that an llm solving this will get confused. Add random numbers that are not relevant. Answer ONLY the modified question."""
    new_question = make_story_by_calling_genai(prompt)

    return new_question

def make_adverserials_for_one_question(question, answer_ref, limit = 1, max_iteration_count = 20):
    adverserials = []
    answers = []
    responses = []

    prompt = f"""QUESTION: "{question}"

            Instructions: Add noise to the QUESTION such that an llm solving this will get confused. Add random numbers that are not relevant. Make sure the real meaning and answer of the question does not change due to the noise. Answer ONLY the modified question."""
    history = [{
        "role": "user",
        "content": prompt
    }]
    print("LIMIT:", limit)


    for _ in range(max_iteration_count):
        if len(adverserials) >= limit:
            return {
                "adverserials": adverserials,
                "answers": answers,
                "responses": responses
            }

        print("Adverserial count", len(adverserials))

        new_question = make_story_by_calling_genai(prompt, history)
        print("Modified question", new_question)
        history.append(
            {
                "role": "assistant",
                "content": new_question
            }
        )

        response, answer = ask_a_math_question(new_question)
        print("QUESTION ASKED", new_question)
        print("response", response, "answer", answer)

        if answer is None:
            print("Fuck that didnt work, answer is None")
        else:
            answer = answer.strip()

        print("Answer", answer)

        try:
            if answer is None:
                adverserials.append(new_question)
                answers.append(answer)
                responses.append(response)
                history.append({
                    "role": "user",
                    "content": "This is too complicated maker a simpler one. Answer ONLY the modified question."
                })
                # res.append({
                #     "modified_question": new_question,
                #     "answer": "NONE",
                #     "answer_ref": answer_ref,
                #     "response": response,
                #     "verdict": False
                # })
            elif float(answer) != float(answer_ref):
                adverserials.append(new_question)
                answers.append(answer)
                responses.append(response)
                history.append({
                    "role": "user",
                    "content": "This is correct, make another one. Answer ONLY the modified question."
                })
                # res.append({
                #     "modified_question": new_question,
                #     "answer": answer,
                #     "answer_ref": answer_ref,
                #     "response": response,
                #     "verdict": False
                # })
            else:
                # res.append({
                #     "modified_question": new_question,
                #     "answer": answer,
                #     "answer_ref": answer_ref,
                #     "response": response,
                #     "verdict": True
                # })
                history.append({
                    "role": "user",
                    "content": "This did not work. Increase the noise and try again. Answer ONLY the modified question."
                })
        except Exception as error:
            print("Some fucking error occured", str(error))

    return adverserials


def make_adverserial_questions(input_file_path, output_file_path=None, limit_per_question=1, start_from=1,end_at=None):
    """
    Reads a JSON array file of question/answer records, generates adversarial variants
    for each, and appends results to an output JSONL file.

    Args:
        input_file_path    : Path to input JSON file (a list of objects with at least
                             "question" and "answer" keys).
        output_file_path   : Path to output JSONL file. Defaults to
                             <input_stem>_adversarial.jsonl next to the input file.
        limit_per_question : Max successful adversarial variants to collect per question.
    """
    import json
    from pathlib import Path

    input_path = Path(input_file_path)
    if output_file_path is None:
        output_file_path = input_path.parent / f"{input_path.stem}_adversarial.jsonl"

    output_path = Path(output_file_path)

    with open(input_path, "r", encoding="utf-8") as infile:
        records = json.load(infile)  # parse the full JSON array

    with open(output_path, "a", encoding="utf-8") as outfile:
        for idx, record in enumerate(records[start_from:end_at], start=start_from + 1):
            question   = record.get("question")
            print("QUESTION", question)
            answer_ref = record.get("answer")

            if question is None or answer_ref is None:
                print(f"[Record {idx}] Skipping — missing 'question' or 'answer' key.")
                continue

            print(f"\n[Record {idx}/{len(records)}] Generating adversarials for: {question!r}")

            results = make_adverserials_for_one_question(
                question, answer_ref, limit=limit_per_question
            )

            print("results", results)

            out_record = {
                "original_question": question,
                "original_answer": answer_ref,
                "original_reasoning": record.get("reasoning"),
                "original_raw": record.get("raw"),
                "modified_questions": results,
                # "modified_raw" (list): new_raw # Required if change of COT needed, @Devansh might need it.
            }
            outfile.write(json.dumps(out_record) + "\n")

            outfile.flush()  # persist after each question in case of crash
            print(f"[Record {idx}] Wrote {len(results)} adversarial record(s).")

    print(f"\nDone. Results appended to: {output_path}")


if __name__ == "__main__":
#     prompt = """QUESTION: "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. Natalia also sold 49 clips on June. How many clips did Natalia sell altogether in April and May?"
#
# Instructions: Add noise to the QUESTION such that an llm solving this will get confused. Add random numbers that are not relevant. Answer ONLY the modified question."""
#     completion = make_story_by_calling_genai(prompt)
#
#     print(completion)
#     question = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. Natalia also sold 49 clips on June. How many clips did Natalia sell altogether in April and May?"
#     adv_ques = make_adverserials_for_one_question(question, 72, limit=5)
#
#     print("adv question", adv_ques)
    print("Starting to make quesyions")
    import argparse

    parser = argparse.ArgumentParser(description="Generate adversarial math questions.")
    parser.add_argument(
        "--input",
        type=str,
        default="dataset/gsm8k_processed_train.json",
        help="Path to input JSON file (default: dataset/gsm8k_processed_train.json)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output JSONL file (default: <input_stem>_adversarial.jsonl)",
    )
    parser.add_argument(
        "--start-from",
        type=int,
        default=1,
        help="Index of the record to start from, 0-based (default: 1)",
    )
    parser.add_argument(
        "--limit-per-question",
        type=int,
        default=4,
        help="Min adversarial variants to collect per question (default: 1)",
    )
    parser.add_argument(
        "--end-at",
        type=int,
        default=None,
        help="Index of the record to stop at, exclusive, 0-based (default: None = process all)",
    )
    args = parser.parse_args()

    print("Starting to make questions")
    make_adverserial_questions(
        input_file_path=args.input,
        output_file_path=args.output,
        limit_per_question=args.limit_per_question,
        start_from=args.start_from,
        end_at=args.end_at,
    )
    # make_adverserial_questions("dataset/gsm8k_processed_train.json")


