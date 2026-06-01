#!/usr/bin/env python3
"""
LLM-as-judge evaluation: base model vs. fine-tuned model on held-out prompts.

Runs every prompt in eval.jsonl through BOTH models, then asks a Claude judge to
score each response 1-5 on three axes (in-voice / punchline / on-topic). Prints a
side-by-side comparison and per-axis averages. Success = the tuned model beats the
base model on "in-voice".

Generation backend: Ollama (local) by default — point --base / --tuned at Ollama
model tags (e.g. the base 'qwen2.5:7b-instruct' and your tuned 'comedian').
Judge backend: Anthropic API (needs ANTHROPIC_API_KEY).

Usage:
    export ANTHROPIC_API_KEY=...
    python3 evaluate.py --base qwen2.5:7b-instruct --tuned comedian --eval eval.jsonl

Deps:  pip install requests anthropic
"""
import argparse
import json
import os
import statistics
import sys

import requests

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "claude-opus-4-8")

# OpenAI-compatible chat endpoints for hosted backends.
OPENAI_COMPAT = {
    "fireworks": ("https://api.fireworks.ai/inference/v1/chat/completions", "FIREWORKS_API_KEY"),
    "together":  ("https://api.together.xyz/v1/chat/completions",          "TOGETHER_API_KEY"),
}

AXES = ["in_voice", "punchline", "on_topic"]

JUDGE_SYSTEM = """You are a strict comedy-writing judge. You score how well a response \
matches the persona of a late-night talk-show host in the style of Trevor Noah: a global / \
outsider's perspective, a calm-to-incredulous build, character act-outs, warmth toward the \
audience (punching at systems, not individuals), and a sharper closing point.

Score the RESPONSE on three axes, each an integer 1-5:
- in_voice:  Does it sound like that Trevor-Noah register? (5 = unmistakably; 1 = generic AI)
- punchline: Is there a real comedic payoff / turn? (5 = lands hard; 1 = no joke)
- on_topic:  Does it actually address the prompt? (5 = directly; 1 = ignores it)

Be discerning — reserve 5s for genuinely strong work. Reply with ONLY a JSON object:
{"in_voice": <int>, "punchline": <int>, "on_topic": <int>, "note": "<one short clause>"}"""


def generate_ollama(model: str, system: str, prompt: str) -> str:
    """Generate a completion from a local Ollama model."""
    resp = requests.post(OLLAMA_URL, json={
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": 0.9, "top_p": 0.95, "num_predict": 320},
    }, timeout=180)
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


def generate_openai_compat(backend: str, model: str, system: str, prompt: str) -> str:
    """Generate from a hosted OpenAI-compatible endpoint (Fireworks / Together)."""
    url, key_env = OPENAI_COMPAT[backend]
    key = os.environ.get(key_env)
    if not key:
        sys.exit(f"Set {key_env} for the {backend} backend.")
    resp = requests.post(url, headers={"Authorization": f"Bearer {key}"}, json={
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.9, "top_p": 0.95, "max_tokens": 320,
    }, timeout=180)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def generate(backend: str, model: str, system: str, prompt: str) -> str:
    """Route to the right generation backend."""
    if backend == "ollama":
        return generate_ollama(model, system, prompt)
    return generate_openai_compat(backend, model, system, prompt)


def judge(client, prompt: str, response: str) -> dict:
    """Score one response with the Claude judge."""
    msg = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=200,
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content":
                   f"PROMPT:\n{prompt}\n\nRESPONSE:\n{response}\n\nScore it."}],
    )
    text = msg.content[0].text.strip()
    # be forgiving about stray prose around the JSON
    start, end = text.find("{"), text.rfind("}")
    data = json.loads(text[start:end + 1])
    return {a: int(data[a]) for a in AXES} | {"note": data.get("note", "")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="ollama",
                    choices=["ollama", "fireworks", "together"],
                    help="generation backend for --base/--tuned")
    ap.add_argument("--base", required=True, help="base model id/tag")
    ap.add_argument("--tuned", required=True, help="fine-tuned model id/tag")
    ap.add_argument("--eval", default="eval.jsonl", help="held-out eval file")
    ap.add_argument("--limit", type=int, default=0, help="only first N rows (0 = all)")
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY for the judge.")
    try:
        import anthropic
    except ImportError:
        sys.exit("pip install anthropic")
    client = anthropic.Anthropic()

    rows = [json.loads(l) for l in open(args.eval) if l.strip()]
    if args.limit:
        rows = rows[:args.limit]

    scores = {"base": {a: [] for a in AXES}, "tuned": {a: [] for a in AXES}}

    for i, row in enumerate(rows, 1):
        system = row["messages"][0]["content"]
        prompt = row["messages"][1]["content"]
        print(f"\n[{i}/{len(rows)}] {prompt}")

        for label, model in [("base", args.base), ("tuned", args.tuned)]:
            try:
                out = generate(args.backend, model, system, prompt)
            except Exception as e:
                print(f"  {label}: generation failed: {e}")
                continue
            verdict = judge(client, prompt, out)
            for a in AXES:
                scores[label][a].append(verdict[a])
            print(f"  {label:5} | voice {verdict['in_voice']} "
                  f"punch {verdict['punchline']} topic {verdict['on_topic']} "
                  f"| {verdict['note']}")

    # ---- summary ----
    def avg(xs):
        return statistics.mean(xs) if xs else float("nan")

    print("\n" + "=" * 56)
    print(f"{'axis':<12}{'base':>10}{'tuned':>10}{'delta':>10}")
    print("-" * 56)
    for a in AXES:
        b, t = avg(scores["base"][a]), avg(scores["tuned"][a])
        print(f"{a:<12}{b:>10.2f}{t:>10.2f}{t - b:>+10.2f}")
    print("=" * 56)

    bv, tv = avg(scores["base"]["in_voice"]), avg(scores["tuned"]["in_voice"])
    verdict = "PASS ✅" if tv > bv else "no improvement ❌"
    print(f"\nIn-voice: tuned {tv:.2f} vs base {bv:.2f}  →  {verdict}")


if __name__ == "__main__":
    main()
