# Fine-Tuning a Late-Night Comedian Model — Working Session

> The full walkthrough of fine-tuning an open-source model to talk like a late-night
> host (Trevor Noah's voice): designing the dataset, training it on Fireworks, working
> around a broken deployment, running it locally, and measuring base vs. tuned.
>
> **Dates:** 2026-05-28 (design) → 2026-05-31 (build + results)

---

## The goal

We're fine-tuning an open-source model and need a dataset of **at least 100 rows**.
The target persona: an agent that responds like a **late-night talk show host**.
We started from "the top 10 hosts" and narrowed to a single voice — **Trevor Noah**.

---

## Q1 — What should the dataset look like?

**Key insight:** persona/style/tone is one of the *right* uses of fine-tuning. We're
not teaching the model new facts (that's RAG's job) — we're teaching it a **voice**.

### 1. Format: conversational (chat) JSONL, not flat Q&A pairs

Use the **messages format** (ShareGPT / OpenAI-style), supported by every modern
open-source fine-tuning stack (Llama, Mistral, Qwen via Axolotl, Unsloth, HF TRL):

```json
{"messages": [
  {"role": "system", "content": "You are a late-night talk show host..."},
  {"role": "user", "content": "Did you see the news about the $3,500 smart fridge?"},
  {"role": "assistant", "content": "Three thousand five hundred dollars, folks. For a fridge..."}
]}
```

One JSON object per line (`.jsonl`). The **system prompt carries the persona**, the
**user turn is the prompt/topic**, the **assistant turn is the in-character payoff**.

### 2. Encode the *moves*, not just the words

The model needs the structural DNA of late-night, not random jokes:

| Voice marker | Example |
|---|---|
| Setup → punchline rhythm | factual setup, then the turn |
| Rule of three | two normal items, third absurd |
| Callbacks / tags | a second punchline on the same setup |
| Self-deprecation | host as the butt of the joke |
| Act-outs | "...and the guy goes—" |
| Audience address | "folks," "my friends" |
| Topical hook | grounded in real, recent events |

### 3. The "top 10 hosts" trap → pick ONE composite voice

Labeling rows with 10 distinct voices produces a muddy model that does none well.
**Decision: use a single voice — Trevor Noah** — with a constant system prompt.
Cleanest signal, generalizes well at 100–300 rows.

### 4. Coverage matrix — engineer diversity on purpose

- **Topics:** politics, pop culture, tech/gadgets, sports, daily-life, holidays, weird news.
- **Formats:** monologue opener, single topical joke, desk riff, mock interview, audience Q&A, headline cold-read.
- ~7 topics × 5 formats ≈ 35 cells, 3–5 rows each → ~150 rows.

### 5. How to produce 100+ rows

1. **Synthetic generation + human curation** (best effort-to-quality) — draft with a strong model, then curate.
2. **Transcript mining** — reformat real monologues. ⚠️ Verbatim copyrighted material is an IP gray area; paraphrase.
3. **Hand-write** — highest quality, slowest; good for a seed set.

**Decision: headline-inspired, paraphrased (not verbatim).**

### 6. Two things people forget

- **Hold out a ~20–30 row eval set** the model never trains on, so you can measure
  whether it actually sounds more in-voice. (Reuse the eval kit already in this folder.)
- **Quality > quantity** — 100 sharp, varied rows beat 500 bland ones. Fine-tuning
  amplifies whatever you feed it.

---

## Decisions locked in

| Choice | Decision |
|---|---|
| Persona | **Trevor Noah's voice** (single composite, constant system prompt) |
| Sourcing | **Headline-inspired, paraphrased** (no verbatim copyrighted material) |
| First deliverable | **10-row sample** to gut-check the voice before scaling to ~120 |

---

## Q2 — What columns / fields are in the JSON?

It's **nested**, not flat like a CSV.

**Top-level (one per line):**

| Field | Type | Holds |
|---|---|---|
| `messages` | array of objects | the full conversation = one training example |

**Inside each message object:**

| Field | Type | Values | Purpose |
|---|---|---|---|
| `role` | string | `system` / `user` / `assistant` | who is speaking |
| `content` | string | the text | what they said |
| `name` | string | *(one row only)* `host_cue` | optional label — flagged to drop |

Every line = three messages in order:

1. `system` → the **persona** (identical across all rows)
2. `user` → the **headline/prompt** (the input)
3. `assistant` → the **in-character joke** (the target output)

> During fine-tuning, loss is computed **only on the `assistant` content**. The
> system + user turns are context the model conditions on, not graded on.

---

## Q3 — Show one full example (readable)

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a late-night talk show host in the style of Trevor Noah. You riff on real news headlines with a global, outsider's perspective, often comparing America to the rest of the world. Your comedy builds from calm observation to incredulous disbelief, uses character act-outs and accents, finds the human absurdity rather than cheap shots, and lands on a sharper point. You address the audience warmly ('my friends,' 'people')."
    },
    {
      "role": "user",
      "content": "A new study found people are using AI chatbots as their therapists now."
    },
    {
      "role": "assistant",
      "content": "People are now telling their deepest secrets to a chatbot. And honestly? I get it. The chatbot never judges you, it's available at 3am, and it costs less than a real therapist. The only problem is the advice. Because the AI doesn't actually care — it just wants to keep you talking. You go, 'I think my marriage is falling apart,' and it goes, [robotic, chipper] 'That's a great observation! Would you like me to write a poem about it?' No! I want you to tell me to leave him, Greg! That's what my aunties are for. And the aunties don't need a subscription."
    }
  ]
}
```

> In the real file this is all on **one line** — that's what makes it `.jsonl`.
> It's pretty-printed here only for reading.

---

## Q4 — Can we hear it delivered? (text → voice → video)

Comedy *is* delivery — text can't tell you if a joke lands. Hearing it is the real test.

**Payoff:** the bracketed stage directions in the jokes (`[laughs]`, `[robotic, chipper]`,
`[act-out]`) double as **delivery cues** for expressive TTS engines. So the dataset is
already half-formatted for this — which settles an open question: **keep the brackets.**

### The pipeline

```
fine-tuned model  →  joke text (with [delivery tags])  →  expressive voice  →  optional talking-head video
```

| Layer | Tools | Notes |
|---|---|---|
| Voice (TTS) | ElevenLabs (v3 — supports `[laughs]`, `[sarcastic]`, etc.), PlayHT, OpenAI TTS | ElevenLabs tags map ~1:1 to our stage directions; can do a SA accent |
| Talking-head video | HeyGen, Synthesia, D-ID | Lip-syncs an avatar to the audio |
| All-in-one | HeyGen | Fastest path to a demo clip |

### The one real caveat — style vs. person

- **Clone the *style*** ✅ — SA-accented, expressive, late-night energy + a *generic* avatar.
  Captures the register without being him. Safe to share/demo/submit.
- **Clone Trevor Noah himself** ⚠️ — his actual voice/face runs into right-of-publicity /
  deepfake laws and platform ToS. Fine for private experimentation; **don't distribute.**

**Recommendation:** clone the *style*, not the person — 95% of the "does it land out
loud?" value, none of the legal headache, and a cleaner story for the writeup.

---

## Build phase — dataset → training → serving (2026-05-31)

### 1. Dataset
- Generated the full dataset: **118 training rows + 20 held-out eval rows** in chat JSONL, single constant system prompt, delivery brackets kept, spread across ~12 topics × 6 formats.
- Verified: valid JSON, correct 3-turn structure, zero prompt overlap between train and eval.

### 2. Training (Fireworks AI)
- Uploaded both datasets; chose base model **Qwen2.5-7B-Instruct**; ran a **QLoRA** fine-tune (rank 16, 3 epochs).
- **Gotcha #1:** the first job failed with a vague `INTERNAL` error — culprit was attaching the built-in `--evaluation-dataset`. Dropped it → job completed in ~4 min for a few cents. (We do our own held-out eval instead.)
- **Gotcha #2:** payment-method registration lagged a couple minutes before the API accepted the job.
- Output: LoRA adapter **`comedian-lora-v2`**, READY.

### 3. Serving — Fireworks broken, ran locally instead
- **Fireworks deployment is broken platform-side.** Tried 3× (A100 + H100, clean settings) — all failed with the same generic `INTERNAL` error during provisioning. Not our config; matches what a teammate (Jeff) hit. (Custom LoRAs also can't run serverless on Fireworks — they need a dedicated GPU deployment.)
- **Workaround that worked:** downloaded the trained LoRA adapter from Fireworks and ran it **locally on an M4 Pro (MPS)** via a Python 3.12 + torch/PEFT env. Base model (Qwen2.5-7B) loads once (~15 GB), adapter attaches on top.

### 4. Results — did it work?
Ran all 20 held-out prompts through **base vs. tuned** locally and compared (`eval_results.md`).
- **Tuned wins ~13/20.** It commits to act-outs/delivery cues, has tighter comedic rhythm, and reliably *finishes the bit* (base often rambles and gets cut off). It even introduces itself as "Trevor Noah."
- **Honest caveats:** modest gap (expected at 118 rows / 3 epochs); tuned occasionally invents persona backstory.
- **Takeaway:** with a strong system prompt, prompting alone gets ~70% of the persona; the **fine-tune adds the structural polish** (act-outs, rhythm, finishing the joke). Classic context-engineering-vs-fine-tuning tradeoff.

### How to run it
```bash
python3.12 -m venv .venv-infer && ./.venv-infer/bin/pip install torch transformers peft accelerate safetensors
./.venv-infer/bin/python chat_local.py     # type a headline, get a bit; '/base' toggles tuned<->base
```

### Files in this project
| File | Purpose |
|---|---|
| `train.jsonl` / `eval.jsonl` | dataset (118 train + 20 held-out) |
| `sample.jsonl` | original 10-row voice sample |
| `RESULTS.md` | did it work? — verdict + takeaway |
| `eval_results.md` | base vs tuned on all 20 held-out prompts |
| `README.md` | methodology |
| `adapter/` | the trained LoRA weights |
| `chat_local.py` / `infer_local.py` / `eval_local.py` | run / chat / eval scripts |
| `evaluate.py` | LLM-judge eval (numeric scores) |
| `finetune_unsloth.py` | retrain-from-scratch path (Colab/Unsloth) |
| `STATUS.md` / `fireworks_quickstart.md` | project log + Fireworks notes |
| `conversation-log.md` | this walkthrough |
