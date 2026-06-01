#!/usr/bin/env python3
"""
Full held-out eval, locally: run all eval.jsonl prompts through base vs tuned
(Qwen2.5-7B + comedian LoRA) on MPS and write a side-by-side markdown report.
No Fireworks, no API key needed.

Usage:  ./.venv-infer/bin/python eval_local.py
Output: eval_results.md
"""
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER = "./adapter"
EVAL = "eval.jsonl"
OUT = "eval_results.md"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

rows = [json.loads(l) for l in open(EVAL) if l.strip()]
print(f"Loading model on {DEVICE}...")
tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).to(DEVICE)
model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()

def generate(system, prompt):
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    inputs = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                     return_tensors="pt", return_dict=True).to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=260, do_sample=True,
                             temperature=0.9, top_p=0.95, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

with open(OUT, "w") as f:
    f.write("# Held-out eval — base vs. tuned (local, Qwen2.5-7B + comedian LoRA)\n\n")
    f.write(f"{len(rows)} held-out prompts the model never trained on. "
            "Base = Qwen2.5-7B-Instruct with the same system prompt; "
            "Tuned = same model + our LoRA adapter.\n\n---\n\n")
    for i, row in enumerate(rows, 1):
        system = row["messages"][0]["content"]
        prompt = row["messages"][1]["content"]
        print(f"[{i}/{len(rows)}] {prompt[:60]}...")
        base = generate(system, prompt)
        tuned = generate(system, prompt)
        f.write(f"## {i}. {prompt}\n\n")
        f.write(f"**BASE:** {base}\n\n")
        f.write(f"**TUNED:** {tuned}\n\n---\n\n")
        f.flush()
print(f"Done -> {OUT}")
