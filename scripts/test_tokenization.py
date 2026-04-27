# from repo root
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from train.baseline_sft_noise_cot import NoiseCOTDataset
from evaluation.evaluate_sft import build_prompt, build_chat_prompt, extract_answer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B-Instruct", trust_remote_code=True)
tok.pad_token = tok.eos_token

ds = NoiseCOTDataset("dataset/noise_cot_train.jsonl", tok, max_length=2048, shuffle=False)
ex = ds[0]

ids = ex["input_ids"]
labels = ex["labels"]
first_label_idx = next(i for i, l in enumerate(labels) if l != -100)

print("=== prompt (masked) ===")
print(tok.decode(ids[:first_label_idx]))
print("=== completion (unmasked, what the model is trained to predict) ===")
print(tok.decode(ids[first_label_idx:]))
print("=== expected completion ===")
print(ds.samples[0]["raw"] + tok.eos_token)


tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B-Instruct", trust_remote_code=True)
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Math-1.5B-Instruct",
                                         torch_dtype=torch.bfloat16, device_map="auto")
q = "Janet has 3 apples. She buys 5 more, then gives 2 to her friend. How many does she have?"
for label, p in [("RAW", build_prompt(q)), ("CHAT", build_chat_prompt(q, tok))]:
    enc = tok(p, return_tensors="pt").to(m.device)
    out = m.generate(**enc, max_new_tokens=256, do_sample=False, pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"--- {label} ---\n{text}\nextracted: {extract_answer(text)}\n")
