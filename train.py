#!/usr/bin/env python
# coding: utf-8
"""
Fine-tune SpaceOm (3 B Qwen2.5-VL backbone) on mixed caption + VQA data
using LoRA (PEFT).  Works on a free Colab T4 / A10 with 16 GB VRAM.
"""

import os, json, torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoProcessor, 
    Qwen2_5_VLForConditionalGeneration,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model
from PIL import Image

# ========= CONFIG ========= #
MODEL_ID      = "remyxai/SpaceOm"
DATA_JSON     = Path("data_preprocess/train_all.json")     # captions + VQA merged
IMAGE_ROOT    = Path("data/bbox_global")                    # images paths are stored relative to this root
OUTPUT_DIR    = "spaceom_lora"
BATCH_SIZE    = 1                               # fits on 16 GB with bnb.int8
EPOCHS        = 3
LR            = 2e-5
MAX_NEW_TOK   = 0                               # no generation during training
MAX_TOKENS    = 1024                            # padding / trunc
# ========================== #

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading model & processor …")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    load_in_8bit=True,          # bitsandbytes (saves vRAM)
)
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

# ----- LoRA adapter -----
peft_cfg = LoraConfig(
    r=128,
    lora_alpha=256,
    target_modules=["q_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_cfg)
model.print_trainable_parameters()

# ========= DATASET ========= #
# ========= DATASET ========= #
class SpaceOmJsonDataset(Dataset):
    """
    Each entry:
    {
      "conversations":[
        { "role":"user",
          "content":[{"type":"image","image":"train/…/3_action.jpg"},
                     {"type":"text","text":"<prompt>"}] },
        { "role":"assistant","content":[{"type":"text","text":"<answer>"}] }
      ]
    }
    """

    def __init__(self, json_path: Path, processor, image_root: Path):
        self.items      = json.load(open(json_path))
        self.processor  = processor
        self.image_root = image_root.resolve()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item       = self.items[idx]
        conv       = item["conversations"]
        user_msg   = conv[0]          # first (and only) user turn
        assistant  = conv[1]          # first assistant turn (answer)

        # ---------- load image(s) ----------
        imgs = []
        for c in user_msg["content"]:
            if c["type"] != "image":
                continue

            raw_path = Path(c["image"])

            # ▸ If JSON already stores an absolute path OR a path that
            #   already starts with the IMAGE_ROOT folder, do **not**
            #   prefix again.
            if raw_path.is_absolute() or str(raw_path).startswith(str(self.image_root)):
                img_path = raw_path.resolve()
            else:
                img_path = (self.image_root / raw_path).resolve()

            try:
                img = Image.open(img_path).convert("RGB")
            except FileNotFoundError as e:
                raise RuntimeError(f"❌ Image not found: {img_path}") from e

            # Down-scale to max-width 512 px (SpaceOm pre-training size)
            if img.width > 512:
                h = int(img.height * 512 / img.width)
                img = img.resize((512, h), Image.Resampling.LANCZOS)

            imgs.append(img)

        # ---------- build chat template ----------
        prompt_chat = [
            user_msg,                   # already contains images + text
            assistant                   # ground-truth answer
        ]
        text_input = self.processor.apply_chat_template(
            prompt_chat, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text_input],
            images=imgs,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=MAX_TOKENS,
        )

        # we only need input_ids & pixel_values
        return {
            "input_ids"    : inputs["input_ids"].squeeze(0),
            "pixel_values" : inputs["pixel_values"],
        }


# ========= TRAINING ========= #
training_args = TrainingArguments(
    output_dir        = OUTPUT_DIR,
    per_device_train_batch_size = BATCH_SIZE,
    gradient_accumulation_steps = 4,
    learning_rate     = LR,
    num_train_epochs  = EPOCHS,
    bf16              = True,
    logging_steps     = 20,
    save_steps        = 500,
    save_total_limit  = 2,
    remove_unused_columns = False,
    fp16              = False,  # we use bf16
    dataloader_pin_memory = True,
    report_to         = "none",
)

def collate_fn(batch):
    return {
        k: torch.stack([x[k] for x in batch]).to(device)
        for k in batch[0]
    }

trainer = Trainer(
    model           = model,
    args            = training_args,
    train_dataset   = train_ds,
    data_collator   = collate_fn,
)

if __name__ == "__main__":
    trainer.train()
    # Save LoRA adapters only
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
