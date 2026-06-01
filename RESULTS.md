# Results — did the fine-tune work?

**Short answer: yes, modestly.** Tuned beats base on ~13 of 20 held-out prompts. Full outputs in `eval_results.md`.

## Setup
- **Base:** Qwen2.5-7B-Instruct + the Trevor-Noah system prompt.
- **Tuned:** same model + our LoRA adapter (118 training rows, QLoRA r=16, 3 epochs).
- Both judged on the **20 held-out prompts** the model never trained on, generated locally (M4 Pro / MPS).

## What the fine-tune added (consistent, observable)
1. **Act-out / delivery cues** — `(Winks)`, `[pause for dramatic effect]`, `(Act out confused)`, `[grinning]`. The bracket convention from the dataset. Tuned uses these reliably; base only occasionally. (Examples: #9, #14, #16, #18, #20.)
2. **Complete bit structure + closing turn.** Base frequently rambles and gets truncated mid-sentence at the token limit (#1, #5, #8, #10, #13). Tuned more often lands a clean closing punchline within budget (#4, #6, #16, #18, #20).
3. **Signature rhythm** — repetition for emphasis ("Millions. Millions."; "not rocket science, my friends, not rocket science").

## Where it's a wash or base wins
- Base sometimes has more varied/creative setups.
- Tuned occasionally **invents Trevor backstory** (e.g., #10 "when I first moved to New York as a hospital intern") — a minor hallucination cost.

## The takeaway (for the writeup)
With a strong system prompt, **prompting alone gets ~70% of the persona**; the **fine-tune adds structural discipline** — act-outs, comedic rhythm, and reliably *finishing the joke*. Classic context-engineering-vs-fine-tuning tradeoff. The gap would widen with more data (this is a light fine-tune: 118 rows, 3 epochs).

## Method note
Eyeball comparison (side-by-side), not numeric. For 1–5 scores on in-voice/punchline/on-topic, run `evaluate.py` with an `ANTHROPIC_API_KEY` (LLM judge).

## How to use the model
```bash
cd "comedian-finetune"
./.venv-infer/bin/python chat_local.py     # interactive: type a headline, get a bit
#   '/base' toggles tuned <-> base ;  'quit' to exit
```
Other entry points: `infer_local.py` (3 sample prompts, base vs tuned) · `eval_local.py` (full 20-prompt report → eval_results.md).
