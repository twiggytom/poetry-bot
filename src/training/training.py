# training script that loads in config data and runs the lora fine tuning job on the base model

# script executed in colab using a L100 (T4 may not support fp16)

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")
DRIVE_OUTPUTS = "/content/drive/MyDrive/poetry-lora-outputs"
LOCAL_CHECKPOINTS = "/content/checkpoints" if os.path.isdir("/content") \
    else os.path.join(PROJECT_ROOT, "checkpoints")

IGNORE_INDEX = -100   # label value the loss skips

# model.loader in the config; vision-language models like Ministral 3 aren't causal LMs to HF
LOADERS = {"causal_lm": AutoModelForCausalLM, "image_text_to_text": AutoModelForImageTextToText}

# generated loss curves
TRAIN_COLOR, EVAL_COLOR = "#2a78d6", "#eb6834"
INK, MUTED_INK, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
SURFACE = "#fcfcfb"


# ---------------------------------------------------------------- data

def load_poems(path: str) -> list[dict]:
    with open(os.path.join(PROJECT_ROOT, path), encoding="utf-8") as fh:
        return json.load(fh)


def encode(tokenizer, poem: dict, max_length: int) -> dict:
    """Token ids for one example, with the prompt masked out of the labels."""
    user = [{"role": "user", "content": poem["prompt"]}]
    full = user + [{"role": "assistant", "content": poem["text"]}]

    # Render to text, then tokenize: apply_chat_template(tokenize=True) returns
    # a dict rather than a token list in newer transformers releases.
    prompt_text = tokenizer.apply_chat_template(user, tokenize=False, add_generation_prompt=True)
    full_text = tokenizer.apply_chat_template(full, tokenize=False)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    input_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    assert input_ids[:len(prompt_ids)] == prompt_ids, \
        f"{poem['id']}: prompt tokens are not a prefix of the full example"

    labels = [IGNORE_INDEX] * len(prompt_ids) + input_ids[len(prompt_ids):]
    input_ids, labels = input_ids[:max_length], labels[:max_length]
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}


def resolve_config(name: str) -> str:
    """A config path, or a model name like qwen2.5-7b for config/train_qwen2.5-7b.yaml."""
    if os.path.isfile(name):
        return name
    path = os.path.join(CONFIG_DIR, f"train_{name}.yaml")
    if os.path.isfile(path):
        return path
    available = sorted(f[len("train_"):-len(".yaml")] for f in os.listdir(CONFIG_DIR)
                       if f.startswith("train_") and f.endswith(".yaml"))
    raise SystemExit(f"No training config {name!r}. Available: {', '.join(available)}")


# ---------------------------------------------------------------- training

class EvaluateOnFinalStep(TrainerCallback):
    """Evaluate and checkpoint on the last step too.

    Total steps rarely land on a multiple of eval_steps, so without this the
    final stretch of training would never be a candidate for the best adapter.
    """

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= state.max_steps:
            control.should_evaluate = True
            control.should_save = True
        return control


def precision_flags(precision: str) -> dict:
    precision = precision.lower()
    if precision not in ("bf16", "fp16", "fp32"):
        raise ValueError(f"precision must be bf16, fp16 or fp32, not {precision!r}")
    if precision != "fp32" and not torch.cuda.is_available():
        raise SystemExit(f"precision {precision} needs a GPU. In Colab: "
                         "Runtime > Change runtime type > GPU.")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise SystemExit("This GPU does not support bf16 (e.g. a T4). Use an L4 or A100 "
                         "runtime, or set precision: fp16 in the training config.")
    return {"bf16": precision == "bf16", "fp16": precision == "fp16"}


def total_optimizer_steps(n_examples: int, cfg: dict) -> int:
    """Matches Trainer's count on a single device (a partial final batch still steps)."""
    batches = math.ceil(n_examples / cfg["per_device_batch_size"])
    return math.ceil(batches / cfg["gradient_accumulation_steps"]) * cfg["num_epochs"]


# ---------------------------------------------------------------- plots

def style(ax, title: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=12, pad=10)
    ax.set_xlabel("optimizer step", color=MUTED_INK)
    ax.set_ylabel("loss", color=MUTED_INK)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED_INK, labelsize=9)
    ax.ticklabel_format(axis="y", useOffset=False)


def save_plots(log_history: list[dict], best_step: int | None, out_dir: str) -> None:
    train = [(e["step"], e["loss"]) for e in log_history if "loss" in e]
    evals = [(e["step"], e["eval_loss"]) for e in log_history if "eval_loss" in e]
    best = next(((s, l) for s, l in evals if s == best_step), None)

    def plot_train(ax):
        if train:
            ax.plot(*zip(*train), color=TRAIN_COLOR, linewidth=2, label="training loss")

    def plot_eval(ax):
        if evals:
            ax.plot(*zip(*evals), color=EVAL_COLOR, linewidth=2, marker="o", markersize=5,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label="validation loss")
        if best:
            ax.plot(*best, linestyle="none", marker="o", markersize=10, color=EVAL_COLOR,
                    markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
            ax.annotate(f"best: step {best[0]}, loss {best[1]:.4f}", best,
                        xytext=(0, 14), textcoords="offset points", ha="center",
                        fontsize=9, color=INK)

    for name, title, draw in (
        ("train_loss.png", "Training loss", [plot_train]),
        ("eval_loss.png", "Validation loss", [plot_eval]),
        ("loss_curves.png", "Training vs validation loss", [plot_train, plot_eval]),
    ):
        fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=SURFACE)
        for fn in draw:
            fn(ax)
        style(ax, title)
        if len(draw) > 1:
            ax.legend(frameon=False, labelcolor=MUTED_INK, fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, name), dpi=150)
        plt.close(fig)

    with open(os.path.join(out_dir, "log_history.json"), "w", encoding="utf-8") as fh:
        json.dump(log_history, fh, indent=2)


# ---------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True,
                        help="model name (e.g. qwen2.5-7b for config/train_qwen2.5-7b.yaml) "
                             "or a path to a training config")
    parser.add_argument("--output-root", default=DRIVE_OUTPUTS,
                        help="existing folder the run folder is created in")
    parser.add_argument("--checkpoint-dir", default=LOCAL_CHECKPOINTS,
                        help="scratch space for intermediate checkpoints")
    args = parser.parse_args()

    # Fail before spending GPU time if the results would have nowhere to go.
    if not os.path.isdir(args.output_root):
        raise SystemExit(
            f"Output folder not found: {args.output_root}\n"
            "Mount Google Drive in a notebook cell first:\n"
            "    from google.colab import drive; drive.mount('/content/drive')")

    config_path = resolve_config(args.config)
    with open(config_path, encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    model_cfg, data_cfg, lora_cfg, train_cfg = (
        config["model"], config["data"], config["lora"], config["training"])
    loader = model_cfg.get("loader", "causal_lm")
    if loader not in LOADERS:
        raise SystemExit(f"model.loader must be one of {', '.join(LOADERS)}, not {loader!r}")
    print(f"Config: {config_path}")

    # Timestamp first so run folders still sort chronologically.
    model_tag = os.path.splitext(os.path.basename(config_path))[0].removeprefix("train_")
    run_name = datetime.now().strftime("run_%Y%m%d-%H%M%S") + f"_{model_tag}"
    run_dir = os.path.join(args.output_root, run_name)
    checkpoint_dir = os.path.join(args.checkpoint_dir, run_name)

    set_seed(train_cfg["seed"])
    flags = precision_flags(train_cfg["precision"])

    # Tokenizer and data.
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["base_model"])
    if model_cfg.get("chat_template"):
        # Saved with the tokenizer below, so inference formats prompts the same way.
        tokenizer.chat_template = model_cfg["chat_template"]
    if not tokenizer.chat_template:
        raise SystemExit(f"{model_cfg['base_model']} has no chat template; "
                         "set model.chat_template in the config")
    max_length = model_cfg["max_seq_length"]
    train_set = [encode(tokenizer, p, max_length) for p in load_poems(data_cfg["train_file"])]
    val_set = [encode(tokenizer, p, max_length) for p in load_poems(data_cfg["val_file"])]
    print(f"Training on {len(train_set)} poems, validating on {len(val_set)}")

    # Base model with LoRA adapters.
    dtype = torch.bfloat16 if flags["bf16"] else torch.float16 if flags["fp16"] else torch.float32
    model = LOADERS[loader].from_pretrained(model_cfg["base_model"], dtype=dtype)
    model.config.use_cache = False   # incompatible with gradient checkpointing
    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()

    total_steps = total_optimizer_steps(len(train_set), train_cfg)
    warmup_steps = math.ceil(total_steps * train_cfg["warmup_ratio"])
    print(f"{total_steps} optimizer steps, {warmup_steps} warmup, "
          f"eval every {train_cfg['eval_steps']}\n")

    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        num_train_epochs=train_cfg["num_epochs"],
        per_device_train_batch_size=train_cfg["per_device_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=float(train_cfg["learning_rate"]),
        lr_scheduler_type=train_cfg["lr_scheduler"],
        warmup_steps=warmup_steps,
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="steps",
        eval_steps=train_cfg["eval_steps"],
        save_strategy="steps",
        save_steps=train_cfg["save_steps"],
        save_total_limit=train_cfg["save_total_limit"],
        load_best_model_at_end=True,   # the best checkpoint is kept despite save_total_limit
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=train_cfg["logging_steps"],
        seed=train_cfg["seed"],
        report_to="none",
        **flags,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=val_set,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True,
                                             label_pad_token_id=IGNORE_INDEX,
                                             pad_to_multiple_of=8),
        callbacks=[EvaluateOnFinalStep()],
    )
    trainer.train()

    # load_best_model_at_end has restored the lowest-val-loss weights.
    state = trainer.state
    best_step = int(state.best_model_checkpoint.rsplit("-", 1)[1]) \
        if state.best_model_checkpoint else None
    print(f"\nBest validation loss {state.best_metric:.4f} at step {best_step}")

    os.makedirs(run_dir)
    trainer.model.save_pretrained(run_dir)
    tokenizer.save_pretrained(run_dir)
    shutil.copy(config_path, os.path.join(run_dir, "train.yaml"))
    with open(os.path.join(run_dir, "training_summary.json"), "w", encoding="utf-8") as fh:
        json.dump({
            "config": os.path.basename(config_path),
            "base_model": model_cfg["base_model"],
            "best_step": best_step,
            "best_eval_loss": state.best_metric,
            "total_steps": state.global_step,
            "train_examples": len(train_set),
            "val_examples": len(val_set),
        }, fh, indent=2)

    plots_dir = os.path.join(run_dir, "loss_plots")
    os.makedirs(plots_dir)
    save_plots(state.log_history, best_step, plots_dir)

    print(f"Saved best adapter and tokenizer to {run_dir}")
    print(f"Saved loss plots to {plots_dir}")


if __name__ == "__main__":
    main()
