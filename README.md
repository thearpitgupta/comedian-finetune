# Late-Night Comedian Fine-Tune

Fine-tuning an open-source LLM to respond in the voice of a **late-night talk-show host**, modeled on **Trevor Noah's** comedic register. This repo holds the dataset, the training notebook, and the evaluation harness.

> **Why fine-tuning (and not RAG/prompting)?** We're not teaching the model *facts* — we're teaching it a *voice*. Persona / style / tone is one of the legitimate sweet spots for fine-tuning. (Facts would belong in retrieval; a one-off voice could live in a prompt, but a consistent, generalizing voice is what fine-tuning is for.)

---

## Files

| File | What it is |
|---|---|
| `train.jsonl` | **118 rows** — the training set |
| `eval.jsonl` | **20 rows** — held-out, never trained on; used to score base-vs-tuned |
| `sample.jsonl` | The original 10-row voice/format sample (kept for reference) |
| `finetune_unsloth.py` | Colab-ready QLoRA training script (Unsloth) |
| `evaluate.py` | LLM-as-judge scoring: base model vs. tuned model on `eval.jsonl` |
| `conversation-log.md` | The design discussion that produced this project |

---

## Dataset design

### Format — conversational (chat) JSONL

Each line is one training example in the **messages** format (ShareGPT / OpenAI-style), supported by every modern fine-tuning stack (TRL, Axolotl, Unsloth, and hosted APIs):

```json
{"messages": [
  {"role": "system",    "content": "<persona definition — identical on every row>"},
  {"role": "user",      "content": "<a real-news-style prompt / headline>"},
  {"role": "assistant", "content": "<the in-character Trevor-Noah-style bit>"}
]}
```

- **`system`** carries the persona. It is **byte-identical across all 138 rows** (train + eval), so the model attaches the voice to the system role, not to any one topic.
- **`user`** is the input the model learns to respond to.
- **`assistant`** is the target. During training, **loss is computed only on the `assistant` turn** (see "train on responses only" below).

### The persona (system prompt)

> You are a late-night talk show host in the style of Trevor Noah. You riff on real news headlines with a global, outsider's perspective, often comparing America to the rest of the world. Your comedy builds from calm observation to incredulous disbelief, uses character act-outs and accents, finds the human absurdity rather than cheap shots, and lands on a sharper point. You address the audience warmly ('my friends,' 'people').

### Voice markers we deliberately encoded

The model should learn the *moves*, not memorize jokes:

| Marker | How it shows up |
|---|---|
| Global / outsider lens | "Where I'm from…", "In other countries…" |
| Calm → incredulous build | a flat setup escalating to "Wait… *what?*" |
| Character act-outs | bracketed delivery cues: `[robotic]`, `[whispers]`, `[act-out]` |
| Warmth over cruelty | audience as "my friends," "people"; punches at systems, not individuals |
| The turn | most bits end on a sharper, human point, not just a punchline |

> **On the brackets:** stage directions like `[laughs]` are kept on purpose. They double as **delivery cues for expressive TTS** (e.g. ElevenLabs audio tags) if we later voice the output. The model will learn to emit them — that's intended.

### Sourcing — headline-inspired, paraphrased

Bits are **original**, pegged to *types* of real, recent news events (tech launches, consumer trends, sports oddities, weird local news) and **paraphrased, never verbatim**. This avoids reproducing copyrighted monologue text and keeps the data about *style*, not specific copyrighted jokes.

### Coverage matrix

Rows are spread across topics × formats so the model generalizes instead of overfitting one lane:

- **Topics:** politics/process, tech & AI, daily-life friction, pop culture, sports, weird/local news, food & consumer, health & wellness, money & economy, travel, social media, seasonal/holiday.
- **Formats:** monologue opener, single topical joke, observational rant, "react to this headline" cold-read, mock-interview banter (a recurring `Welcome him/her` guest setup).

### Train / eval split

- **118 train** rows are what the model sees.
- **20 eval** rows use **fresh topics with zero prompt overlap** (verified) — so the eval measures *generalization of voice*, not memorization.

---

## Training (recommended path)

**Method:** QLoRA (4-bit base + LoRA adapters) — the right choice for ~100 rows. A full fine-tune would overfit and needs far more VRAM.

**Base model:** a 7–8B *instruct* model — `Qwen/Qwen2.5-7B-Instruct` or `meta-llama/Llama-3.1-8B-Instruct`. Start from the **-Instruct** variant so we're only bending the style.

**Tool:** [Unsloth](https://github.com/unslothai/unsloth) on a free Colab T4 — fastest, lowest-VRAM path; wraps HF TRL's `SFTTrainer`.

### Run it

1. Open Google Colab → new notebook → set runtime to **GPU (T4)**.
2. Upload `train.jsonl` (and `eval.jsonl`).
3. Paste the contents of `finetune_unsloth.py` into a cell and run.
4. The script: loads the model, **applies the model's chat template**, configures QLoRA, trains **on responses only** for **3 epochs**, then saves the adapter and exports a **GGUF** you can run locally in [Ollama](https://ollama.com).

### Settings that matter at this scale

| Setting | Value | Why |
|---|---|---|
| Method | QLoRA, rank 16, alpha 16 | PEFT; small adapter on a frozen 4-bit base |
| Epochs | ~3 (watch for overfit) | small data needs a few passes; too many → it recites rows |
| Learning rate | 2e-4 | standard LoRA starting point |
| Loss masking | **responses only** | learn the *voice*, not the prompts. Single biggest quality lever |
| Chat template | model's own, via `apply_chat_template` | #1 silent bug is a template mismatch |

---

## Evaluation

Comedy is about whether it *lands*, so we score outputs rather than trusting train loss.

`evaluate.py` runs all 20 `eval.jsonl` prompts through **both** the base model and the tuned model, then uses an **LLM judge** to score each response 1–5 on:

- **In-voice** — does it sound like the Trevor-Noah register (global lens, build, act-outs, warmth)?
- **Punchline** — is there an actual comedic payoff / turn?
- **On-topic** — does it address the prompt?

It prints a base-vs-tuned comparison. **Success = the tuned model beats the base model on "in-voice"** on the held-out prompts.

> Reuses the LLM-as-judge approach from the `llm-evals-starter-kit/` in the parent folder. 20 hand-checkable examples graded on a clear rubric beats any vanity metric.

### Honest caveats / limits

- **Likeness:** this models a comedic *register*, not the real person. If we later add voice/video, clone the *style* (accent, delivery), **not** Trevor Noah's actual voice or face — right-of-publicity and platform-ToS issues.
- **Judge bias:** an LLM judge can be swayed by length/confidence. Spot-check its scores by hand on a few rows.
- **Single-voice scope:** one composite persona by design. Multi-host style-tagging would need ~50+ rows per host.

---

## Reproduce from scratch

```bash
# 1. (optional) validate the data
python3 -c "import json; [json.loads(l) for l in open('train.jsonl')]; print('train OK')"

# 2. train  → run finetune_unsloth.py in Colab (GPU)

# 3. evaluate base vs tuned
export ANTHROPIC_API_KEY=...   # for the judge
python3 evaluate.py --base <base_model> --tuned <tuned_or_gguf> --eval eval.jsonl
```
