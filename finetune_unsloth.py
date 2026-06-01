# ============================================================================
# Late-Night Comedian Fine-Tune  —  QLoRA via Unsloth
# ============================================================================
# Colab-ready. Cells are separated by `# %%` so you can paste this whole file
# into a Colab/Jupyter notebook (or run with jupytext) and step through it.
#
#   Runtime > Change runtime type > GPU (a free T4 is enough for a 7-8B QLoRA).
#
# What this does, end to end:
#   1. installs Unsloth
#   2. loads a 4-bit instruct model + attaches LoRA adapters (QLoRA)
#   3. loads train.jsonl in the chat `messages` format
#   4. applies the MODEL'S OWN chat template (the #1 thing people get wrong)
#   5. trains ON RESPONSES ONLY (loss masked to the assistant turn)
#   6. quick before/after sniff test
#   7. saves the LoRA adapter and exports a GGUF for Ollama
# ============================================================================

# %% [1] Install ------------------------------------------------------------
# In Colab, uncomment:
# !pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# !pip install -q --no-deps trl peft accelerate bitsandbytes

# %% [2] Config -------------------------------------------------------------
# Pick ONE base model. Both are solid 7-8B instruct models that QLoRA on a T4.
BASE_MODEL   = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"      # or:
# BASE_MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"

TRAIN_FILE   = "train.jsonl"     # upload this to the Colab file panel
MAX_SEQ_LEN  = 2048
EPOCHS       = 3                 # ~3 for ~120 rows. Watch for overfit (it starts reciting rows).
LORA_RANK    = 16
LORA_ALPHA   = 16
LEARNING_RATE = 2e-4
OUTPUT_DIR   = "comedian-lora"

# %% [3] Load the 4-bit model + attach LoRA (this is the "QLoRA" part) -------
from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name   = BASE_MODEL,
    max_seq_length = MAX_SEQ_LEN,
    load_in_4bit = True,         # 4-bit base = fits on consumer / Colab GPU
    dtype        = None,         # auto (bf16 if supported, else fp16)
)

model = FastLanguageModel.get_peft_model(
    model,
    r              = LORA_RANK,
    lora_alpha     = LORA_ALPHA,
    lora_dropout   = 0,
    bias           = "none",
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing = "unsloth",
    random_state   = 3407,
)

# %% [4] Load the dataset ----------------------------------------------------
# Each row is already {"messages": [system, user, assistant]} — exactly what
# the chat template wants. We just render each row to a single training string.
from datasets import load_dataset

dataset = load_dataset("json", data_files=TRAIN_FILE, split="train")

def render(example):
    # apply_chat_template uses THIS model's exact format (the right way).
    text = tokenizer.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=False
    )
    return {"text": text}

dataset = dataset.map(render)
print(f"Loaded {len(dataset)} rows. Sample render:\n")
print(dataset[0]["text"][:600], "...\n")

# %% [5] Trainer — train ON RESPONSES ONLY ----------------------------------
# Masking the loss to the assistant turn is the single biggest quality lever:
# the model learns to *produce the voice*, not to memorize/parrot the prompts.
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth.chat_templates import train_on_responses_only

trainer = SFTTrainer(
    model           = model,
    tokenizer       = tokenizer,
    train_dataset   = dataset,
    dataset_text_field = "text",
    max_seq_length  = MAX_SEQ_LEN,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,   # effective batch = 8
        warmup_ratio        = 0.05,
        num_train_epochs    = EPOCHS,
        learning_rate       = LEARNING_RATE,
        logging_steps       = 5,
        optim               = "adamw_8bit",
        weight_decay        = 0.01,
        lr_scheduler_type   = "linear",
        seed                = 3407,
        output_dir          = OUTPUT_DIR,
        report_to           = "none",
    ),
)

# These markers must match the chat template of the chosen base model.
# Qwen2.5:
RESP_MARKERS = dict(
    instruction_part = "<|im_start|>user\n",
    response_part    = "<|im_start|>assistant\n",
)
# For Llama-3.1 instead, use:
# RESP_MARKERS = dict(
#     instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
#     response_part    = "<|start_header_id|>assistant<|end_header_id|>\n\n",
# )
trainer = train_on_responses_only(trainer, **RESP_MARKERS)

# %% [6] Train --------------------------------------------------------------
trainer_stats = trainer.train()
print(trainer_stats)

# %% [7] Quick sniff test (does it sound like the host now?) -----------------
FastLanguageModel.for_inference(model)

SYSTEM = ("You are a late-night talk show host in the style of Trevor Noah. "
          "You riff on real news headlines with a global, outsider's perspective, "
          "often comparing America to the rest of the world. Your comedy builds from "
          "calm observation to incredulous disbelief, uses character act-outs and accents, "
          "finds the human absurdity rather than cheap shots, and lands on a sharper point. "
          "You address the audience warmly ('my friends,' 'people').")

def joke(prompt):
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    out = model.generate(input_ids=inputs, max_new_tokens=300,
                         temperature=0.9, top_p=0.95, do_sample=True)
    return tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)

print(joke("A company invented a smart umbrella that texts you when it's about to rain."))

# %% [8] Save adapter + export GGUF for Ollama ------------------------------
# LoRA adapter (small, ~tens of MB):
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Merged GGUF you can run locally with Ollama / llama.cpp:
model.save_pretrained_gguf("comedian-gguf", tokenizer, quantization_method="q4_k_m")

# Then locally:
#   1. create a Modelfile:  FROM ./comedian-gguf/<the>.gguf
#   2. ollama create comedian -f Modelfile
#   3. ollama run comedian
print("Done. Adapter in ./comedian-lora, GGUF in ./comedian-gguf")
