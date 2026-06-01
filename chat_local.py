#!/usr/bin/env python3
"""
Interactive comedian — type a headline, get a bit in the tuned voice.
Runs the local Qwen2.5-7B + comedian LoRA on MPS.

Usage:  ./.venv-infer/bin/python chat_local.py
        (type a headline and press enter; 'quit' to exit;
         '/base' toggles between tuned and base model)
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

print(f"Loading model on {DEVICE} (first load takes a moment)...")
tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).to(DEVICE)
model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()

def generate(prompt, use_adapter=True):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    inputs = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                     return_tensors="pt", return_dict=True).to(DEVICE)
    ctx = torch.no_grad()
    with ctx:
        if use_adapter:
            out = model.generate(**inputs, max_new_tokens=450, do_sample=True,
                                 temperature=0.9, top_p=0.95, pad_token_id=tok.eos_token_id)
        else:
            with model.disable_adapter():
                out = model.generate(**inputs, max_new_tokens=450, do_sample=True,
                                     temperature=0.9, top_p=0.95, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

use_adapter = True
print("Ready. Type a headline. ('/base' toggles tuned/base, 'quit' to exit)\n")
while True:
    try:
        prompt = input(f"[{'TUNED' if use_adapter else 'BASE'}] headline> ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if not prompt:
        continue
    if prompt.lower() in ("quit", "exit"):
        break
    if prompt == "/base":
        use_adapter = not use_adapter
        print(f"-> now using {'TUNED' if use_adapter else 'BASE'} model\n")
        continue
    print("\n" + generate(prompt, use_adapter) + "\n")
