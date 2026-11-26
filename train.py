"""Fine-tune NLLB-200 (distilled 600M) with LoRA on English / Virac / Pandan parallel data.

Usage:
    python train.py                  # prepares the corpus if needed, then trains
    python train.py --epochs 20 --out adapter
Outputs: adapter/ (LoRA weights + tokenizer) and results/loss_curve.{csv,png}
"""
import argparse
import math
import os
import random

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, EarlyStoppingCallback,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments, set_seed)

import common as C


def build(df, split, tok):
    feats = []
    for _, r in df[df.split == split].iterrows():
        for s, t in C.DIRECTIONS:  # all 6 directions per sentence triple
            tok.src_lang, tok.tgt_lang = C.NLLB[s], C.NLLB[t]
            e = tok(C.prep_src(r[C.COL[s]], t), text_target=r[C.COL[t]], max_length=C.MAX_LEN, truncation=True)
            feats.append(dict(input_ids=e["input_ids"], attention_mask=e["attention_mask"], labels=e["labels"]))
    random.Random(C.SEED).shuffle(feats)
    return Dataset.from_list(feats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--out", default=C.ADAPTER_DIR)
    ap.add_argument("--ckpt", default="ckpt")
    a = ap.parse_args()

    random.seed(C.SEED); torch.manual_seed(C.SEED); set_seed(C.SEED)
    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    if not os.path.exists(f"{C.DATA_DIR}/bikol_clean_corpus.csv"):
        C.prepare_corpus()
    df = C.load_corpus()

    tok = AutoTokenizer.from_pretrained(C.BASE_MODEL)
    train_ds, val_ds = build(df, "train", tok), build(df, "validation", tok)
    print(len(train_ds), "train /", len(val_ds), "validation examples")

    model = AutoModelForSeq2SeqLM.from_pretrained(C.BASE_MODEL)
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM, r=C.LORA_R, lora_alpha=C.LORA_ALPHA, lora_dropout=C.LORA_DROPOUT,
        target_modules=["q_proj", "v_proj"], bias="none"))
    model.print_trainable_parameters()

    steps_per_epoch = math.ceil(len(train_ds) / C.BATCH)
    args = Seq2SeqTrainingArguments(
        output_dir=a.ckpt, num_train_epochs=a.epochs, learning_rate=C.LR, lr_scheduler_type="linear",
        warmup_steps=int(0.1 * a.epochs * steps_per_epoch), optim="adamw_torch",
        per_device_train_batch_size=C.BATCH, per_device_eval_batch_size=C.BATCH,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=2, load_best_model_at_end=True,
        metric_for_best_model="eval_loss", greater_is_better=False, logging_strategy="epoch",
        fp16=torch.cuda.is_available(), report_to="none", seed=C.SEED)
    trainer = Seq2SeqTrainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=DataCollatorForSeq2Seq(tok, model=model, label_pad_token_id=-100),
        callbacks=[EarlyStoppingCallback(5)])
    trainer.train()
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    mb = sum(os.path.getsize(f"{a.out}/{f}") for f in os.listdir(a.out) if f.endswith(".safetensors")) / 1e6
    print(f"adapter saved to {a.out}/ ({mb:.1f} MB)")

    # train vs validation loss per epoch (overfitting check)
    h = {}
    for r in trainer.state.log_history:
        e = int(round(r.get("epoch", 0)))
        if "loss" in r:
            h.setdefault(e, {})["train"] = r["loss"]
        if "eval_loss" in r:
            h.setdefault(e, {})["validation"] = r["eval_loss"]
    loss = pd.DataFrame(h).T.sort_index()
    loss.to_csv(f"{C.RESULTS_DIR}/loss_curve.csv")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    loss.plot(marker="o", xlabel="epoch", ylabel="loss", title="LoRA fine-tuning loss")
    plt.savefig(f"{C.RESULTS_DIR}/loss_curve.png", dpi=150)


if __name__ == "__main__":
    main()
