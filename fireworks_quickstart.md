# Fireworks fine-tune — copy-paste quickstart

Verified against `firectl` v1.7.21. The CLI is already installed on this machine.

> **You must run step 1 yourself** — `signin` is interactive and bills to your account.
> In Claude Code you can run a command in this session by typing `!` first, e.g. `! firectl signin`.

## 0. (already done)
```bash
brew install fw-ai/firectl/firectl   # ✅ installed, v1.7.21
```

## 1. Sign in  (interactive — you run this)
```bash
firectl signin
```

## 2. Upload the datasets
```bash
# positional args:  <dataset-name>  <path>
firectl create dataset comedian-train train.jsonl
firectl create dataset comedian-eval  eval.jsonl
```

## 3. Launch the LoRA fine-tune
```bash
firectl create supervised-fine-tuning-job \
  --base-model accounts/fireworks/models/qwen2p5-7b-instruct \
  --dataset comedian-train \
  --evaluation-dataset comedian-eval \
  --output-model comedian-lora \
  --epochs 3 \
  --lora-rank 16
```
Notes:
- LoRA is the default (don't pass `--full-parameter`). We bump rank 8 → 16.
- If the base-model ID errors, list exact IDs with `firectl list models` and swap it
  in (e.g. a Llama 3.1 8B instruct ID instead of Qwen).
- Watch progress: `firectl get supervised-fine-tuning-job <job-id>`

## 4. Deploy the tuned model (serverless LoRA)
```bash
firectl create deployment accounts/<your-account>/models/comedian-lora
```
This returns a model ID + an OpenAI-compatible endpoint.

## 5. Score base vs. tuned  (run in terminal — needs the judge key)
```bash
export FIREWORKS_API_KEY=...      # for generation
export ANTHROPIC_API_KEY=...      # for the LLM judge
python3 evaluate.py \
  --backend fireworks \
  --base  accounts/fireworks/models/qwen2p5-7b-instruct \
  --tuned accounts/<your-account>/models/comedian-lora \
  --eval  eval.jsonl
```
> Requires the `--backend fireworks` flag — ask Claude to add it to `evaluate.py`
> (a ~5-line OpenAI-compatible swap; works for Together too).

## Cost / sanity
- 118 rows × 3 epochs on a 7B LoRA is small — training is typically cents-to-a-few-dollars
  and finishes in minutes. Check `firectl billing` and the dashboard if unsure.
- This **replaces** the Unsloth/Colab path — pick one, not both. Same dataset feeds either.
