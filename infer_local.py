#!/usr/bin/env python3
"""
Run the trained LoRA locally (Apple Silicon / MPS) and show base vs. tuned
side-by-side. Loads Qwen2.5-7B-Instruct once, then toggles the adapter on/off.

First run downloads the base model (~15 GB) from HuggingFace.
Usage:  ./.venv-infer/bin/python infer_local.py
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER = "./adapter"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

SYSTEM = ("You are a late-night talk show host in the style of Trevor Noah. "
          "You riff on real news headlines with a global, outsider's perspective, "
          "often comparing America to the rest of the world. Your comedy builds from "
          "calm observation to incredulous disbelief, uses character act-outs and accents, "
          "finds the human absurdity rather than cheap shots, and lands on a sharper point. "
          "You address the audience warmly ('my friends,' 'people').")

PROMPTS = [
    "A new app charges a monthly subscription to unlock the buttons on your microwave.",
    "Scientists say they taught pigeons to read basic words.",
    "A city installed a robot to give parking tickets and people keep dressing it up.",
]

print(f"Loading {BASE} on {DEVICE} (first run downloads ~15GB)...")
tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16).to(DEVICE)
model = PeftModel.from_pretrained(model, ADAPTER)   # adapter attached, ON by default
model.eval()

def generate(prompt):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    inputs = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                     return_tensors="pt", return_dict=True).to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=260, do_sample=True,
                             temperature=0.9, top_p=0.95, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

for p in PROMPTS:
    print("\n" + "=" * 80 + f"\nPROMPT: {p}\n" + "=" * 80)
    with model.disable_adapter():        # base model (adapter off)
        print("\n--- BASE (qwen2.5-7b-instruct) ---\n" + generate(p))
    print("\n--- TUNED (comedian-lora-v2) ---\n" + generate(p))   # adapter on
