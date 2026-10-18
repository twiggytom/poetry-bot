# inference script to be run in Colab

# works with CPU or GPU

# loads in adapter and base model weights -> asks user for prompt -> streams poem back

# works with any run from training.py (Qwen2.5-1.5B, Qwen2.5-7B, Ministral 3 8B): the base
# model, loader and chat template all come from the train.yaml saved in the run folder

# issues surrounding repetition that may be a result of inference script/settings

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
from datetime import datetime

import torch
import yaml
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
    StoppingCriteria,
    StoppingCriteriaList,
    TextStreamer,
)

# This file lives at <project root>/src/inference/inference.py.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_CONFIG = os.path.join(PROJECT_ROOT, "config", "inference.yaml")
RULE = "-" * 72

# Hard token cap, as a backstop for poems with no line or sentence breaks.
# Poems in the training data average ~1.4 tokens per word.
TOKENS_PER_WORD_CAP = 1.6
OVERSHOOT_WORDS = 100
# Code identifiers the base model associates with a bare "." token; they occur
# in none of the training poems, so they are never a legitimate continuation.
# (Qwen vocabulary; tokens missing from another model's vocabulary are skipped.)
SUPPRESSED_TOKENS = ("_Statics", "Statics")

# model.loader in the run's train.yaml; must match LOADERS in training.py
LOADERS = {"causal_lm": AutoModelForCausalLM, "image_text_to_text": AutoModelForImageTextToText}


def find_run_dir(output_root: str, run: str | None, model: str | None = None) -> str:
    """The named run (a folder name or a full path), else the newest run_* folder.

    With `model` (e.g. qwen2.5-7b), only runs trained from config/train_<model>.yaml count.
    """
    if run:
        path = run if os.path.isabs(run) else os.path.join(output_root, run)
        if not os.path.isfile(os.path.join(path, "adapter_config.json")):
            raise SystemExit(f"No adapter found in {path}")
        return path
    if not os.path.isdir(output_root):
        raise SystemExit(
            f"Output folder not found: {output_root}\n"
            "Mount Google Drive in a notebook cell first:\n"
            "    from google.colab import drive; drive.mount('/content/drive')")
    # run_YYYYMMDD-HHMMSS[_model] names sort chronologically.
    pattern = f"run_*_{model}" if model else "run_*"
    runs = sorted(d for d in glob.glob(os.path.join(output_root, pattern))
                  if os.path.isfile(os.path.join(d, "adapter_config.json")))
    if not runs:
        which = f"{model} runs" if model else "run folders"
        raise SystemExit(f"No {which} with an adapter in {output_root}")
    return runs[-1]


def poem_end_id(tokenizer) -> int:
    """The token the chat template closes the assistant turn with.

    Training ended every poem with it, so it is what the model learned to stop on:
    <|im_end|> for Qwen's template, </s> for Ministral's.
    """
    probe = "POEM_END_PROBE"
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "x"}, {"role": "assistant", "content": probe}], tokenize=False)
    tail = tokenizer(text.split(probe, 1)[1], add_special_tokens=False)["input_ids"]
    special = set(tokenizer.all_special_ids)
    return next((i for i in tail if i in special), tokenizer.eos_token_id)


def pick_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    return torch.float32


# ---------------------------------------------------------------- length control

class PoemProgress:
    """Reads the poem generated so far: word count and whether it is at a break."""

    def __init__(self, tokenizer, prompt_length: int):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length

    def read(self, input_ids: torch.Tensor) -> tuple[int, bool, bool]:
        generated = input_ids[0, self.prompt_length:]
        if len(generated) == 0:
            return 0, False, False
        words = len(self.tokenizer.decode(generated, skip_special_tokens=True).split())
        last = self.tokenizer.decode(generated[-1:])
        at_line_end = "\n" in last
        at_sentence_end = last.rstrip().endswith((".", "!", "?"))
        return words, at_line_end, at_sentence_end


class NudgeTowardEnding(LogitsProcessor):
    """Past min_words, make ending the poem more likely at each line break."""

    def __init__(self, progress: PoemProgress, end_id: int, length: dict):
        self.progress, self.end_id = progress, end_id
        self.min_words = length["min_words"]
        self.span = max(1, length["max_words"] - length["min_words"])
        self.strength = length["ending_strength"]

    def __call__(self, input_ids, scores):
        words, at_line_end, _ = self.progress.read(input_ids)
        if at_line_end and words >= self.min_words:
            scores[:, self.end_id] += self.strength * (words - self.min_words) / self.span
        return scores


class StopAtBreakPastMax(StoppingCriteria):
    """Past max_words, stop as soon as a line or sentence is complete."""

    def __init__(self, progress: PoemProgress, max_words: int):
        self.progress, self.max_words = progress, max_words

    def __call__(self, input_ids, scores, **kwargs):
        words, at_line_end, at_sentence_end = self.progress.read(input_ids)
        done = words >= self.max_words and (at_line_end or at_sentence_end)
        return torch.full((input_ids.shape[0],), done, dtype=torch.bool, device=input_ids.device)


# ---------------------------------------------------------------- repetition control

class RecentWordPenalty(LogitsProcessor):
    """Lower a word token's logit by `penalty` per use in the last `window` poem tokens.

    Only tokens containing a letter are penalised, so punctuation and line
    breaks -- which every poem repeats constantly -- stay untouched. Counting
    over a recent window, not the whole poem, keeps common words like "the"
    from being squeezed out of a long poem.
    """

    def __init__(self, word_mask: torch.Tensor, prompt_length: int, penalty: float, window: int):
        self.word_mask, self.prompt_length = word_mask, prompt_length
        self.penalty, self.window = penalty, window

    def __call__(self, input_ids, scores):
        recent = input_ids[0, self.prompt_length:][-self.window:]
        if len(recent) == 0 or self.penalty == 0:
            return scores
        vocab = scores.shape[-1]   # the model pads its output layer past the tokenizer's size
        if self.word_mask.shape[0] != vocab:
            padded = torch.zeros(vocab, dtype=torch.bool)
            n = min(vocab, self.word_mask.shape[0])
            padded[:n] = self.word_mask[:n]
            self.word_mask = padded
        self.word_mask = self.word_mask.to(scores.device)
        counts = torch.bincount(recent, minlength=vocab)[:vocab]
        scores[0] -= self.penalty * counts.to(scores.dtype) * self.word_mask
        return scores


def word_token_mask(tokenizer) -> torch.Tensor:
    """True for vocabulary entries containing an ASCII letter, excluding special tokens.

    Raw byte-level token strings keep ASCII letters as-is (a leading space is
    the separate symbol "Ġ"), so no decoding is needed.
    """
    mask = torch.zeros(len(tokenizer), dtype=torch.bool)
    for token, idx in tokenizer.get_vocab().items():
        if re.search(r"[A-Za-z]", token):
            mask[idx] = True
    mask[tokenizer.all_special_ids] = False
    return mask


# ---------------------------------------------------------------- generation

class PoemWriter:
    def __init__(self, run_dir: str, settings: dict):
        with open(os.path.join(run_dir, "train.yaml"), encoding="utf-8") as fh:
            train_config = yaml.safe_load(fh)
        self.max_seq_length = train_config["model"]["max_seq_length"]
        self.settings = settings

        model_cfg = train_config["model"]
        self.tokenizer = AutoTokenizer.from_pretrained(run_dir)
        if model_cfg.get("chat_template"):
            # Already saved with the tokenizer; set again so it is the exact training format.
            self.tokenizer.chat_template = model_cfg["chat_template"]
        loader = LOADERS[model_cfg.get("loader", "causal_lm")]
        base = loader.from_pretrained(
            model_cfg["base_model"], dtype=pick_dtype(),
            device_map="auto" if torch.cuda.is_available() else None)
        self.model = PeftModel.from_pretrained(base, run_dir)
        self.model.eval()

        # Stop on the token training ended poems with, or the tokenizer's own eos
        # (for Qwen these differ: <|im_end|> vs <|endoftext|>).
        self.end_id = poem_end_id(self.tokenizer)
        self.stop_ids = list(dict.fromkeys([self.end_id, self.tokenizer.eos_token_id]))
        vocab = self.tokenizer.get_vocab()
        self.suppressed = [vocab[t] for t in SUPPRESSED_TOKENS if t in vocab]
        self.word_mask = word_token_mask(self.tokenizer)

    def encode(self, prompt: str) -> torch.Tensor:
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False,
                                                  add_generation_prompt=True)
        ids = self.tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
        return ids.to(self.model.device)

    @torch.no_grad()
    def write(self, prompt: str, seed: int) -> str:
        length, sampling, rep = (self.settings["length"], self.settings["sampling"],
                                 self.settings["repetition"])
        input_ids = self.encode(prompt)
        prompt_length = input_ids.shape[1]
        # Training never saw an example longer than max_seq_length tokens.
        room = self.max_seq_length - prompt_length
        if room <= 0:
            return "(prompt is too long: no room left to write a poem)"
        cap = int((length["max_words"] + OVERSHOOT_WORDS) * TOKENS_PER_WORD_CAP)
        progress = PoemProgress(self.tokenizer, prompt_length)

        torch.manual_seed(seed)
        output = self.model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=min(cap, room),
            do_sample=True,
            temperature=sampling["temperature"],
            top_p=sampling["top_p"],
            min_p=sampling["min_p"],
            top_k=0,
            no_repeat_ngram_size=rep["no_repeat_ngram_size"],
            suppress_tokens=self.suppressed or None,
            logits_processor=LogitsProcessorList([
                RecentWordPenalty(self.word_mask, prompt_length,
                                  rep["frequency_penalty"], rep["window"]),
                NudgeTowardEnding(progress, self.end_id, length),
            ]),
            stopping_criteria=StoppingCriteriaList([
                StopAtBreakPastMax(progress, length["max_words"])]),
            eos_token_id=self.stop_ids,
            pad_token_id=self.tokenizer.pad_token_id,
            streamer=TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True),
        )
        return self.tokenizer.decode(output[0, prompt_length:], skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompt", help="write one poem for this prompt and exit; "
                                         "omit to type prompts one after another")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--run", help="run folder name (or full path); overrides the config")
    parser.add_argument("--model", help="use the newest run of this model, e.g. qwen2.5-7b or "
                                        "ministral3-8b (ignored when a run is named)")
    parser.add_argument("--no-save", action="store_true",
                        help="don't append poems to generations.jsonl")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        settings = yaml.safe_load(fh)
    length = settings["length"]
    if not 0 < length["min_words"] <= length["max_words"]:
        raise SystemExit("config: need 0 < min_words <= max_words")

    run_dir = find_run_dir(settings["output_root"], args.run or settings["run"], args.model)
    device = "GPU" if torch.cuda.is_available() else "CPU (no GPU found; this will be slow)"
    with open(os.path.join(run_dir, "train.yaml"), encoding="utf-8") as fh:
        base_model = yaml.safe_load(fh)["model"]["base_model"]
    print(f"Loading adapter from {run_dir} ({base_model}) on {device}")
    writer = PoemWriter(run_dir, settings)
    log_path = os.path.join(run_dir, "generations.jsonl")
    save = settings["save_generations"] and not args.no_save

    def handle(prompt: str) -> None:
        seed = settings["sampling"]["seed"]
        if seed is None:
            seed = random.randrange(2**31)
        print(RULE)
        poem = writer.write(prompt, seed)
        print(f"{RULE}\n({len(poem.split())} words, seed {seed})")
        if not save:
            return
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "time": datetime.now().isoformat(timespec="seconds"),
                "prompt": prompt,
                "poem": poem,
                "seed": seed,
                "settings": {k: settings[k] for k in ("length", "sampling", "repetition")},
            }, ensure_ascii=False) + "\n")
        print(f"(saved to {log_path})")

    if args.prompt:
        handle(args.prompt)
        return

    print("\nType a prompt and press Enter. An empty line or 'quit' exits.")
    while True:
        try:
            prompt = input("\nPrompt> ").strip()
        except EOFError:
            break
        if not prompt or prompt.lower() in ("quit", "exit"):
            break
        handle(prompt)


if __name__ == "__main__":
    main()
