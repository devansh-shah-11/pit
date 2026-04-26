import json
from datasets import load_dataset
import unicodedata
import re

def normalize_text(text: str) -> str:
    """
    Normalize text to remove weird unicode characters and optionally convert to ASCII.
    """
    # Normalize unicode (NFKD) and remove diacritics
    text = unicodedata.normalize("NFKD", text)
    # Encode as ASCII, ignore non-ASCII chars, then decode back
    text = text.encode("ascii", "ignore").decode("ascii")
    # Optional: clean multiple spaces
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def process_example(example):
    question = example["question"]
    raw = example["answer"]

    if "####" in raw:
        reasoning, answer = raw.split("####")
        reasoning = normalize_text(reasoning.strip())
        answer = answer.strip()
    else:
        reasoning = raw.strip()
        answer = ""

    answer = normalize_text(answer)

    reasoning = reasoning.replace("\n", " ").strip()

    combined = f"<reasoning>{reasoning}</reasoning><answer>{answer}</answer>"

    return {
        "question": question,
        "raw": raw,
        "reasoning": reasoning,
        "answer": answer,
        "combined": combined
    }

if __name__ == '__main__':
    dataset = load_dataset("gsm8k", "main")

    print("got the dataset")
    train_processed = [process_example(x) for x in dataset["train"]]
    test_processed = [process_example(x) for x in dataset["test"]]

    print("processed dataset found")

    # Save to file
    with open("dataset/gsm8k_processed_train.json", "w") as f:
        json.dump(train_processed, f, indent=2)

    with open("dataset/gsm8k_processed_test.json", "w") as f:
        json.dump(test_processed, f, indent=2)

    print("Saved processed dataset.")