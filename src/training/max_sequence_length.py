# picks a max sequence length for the poems in the dataset (used in training)

# ensures 95-99% of poems fit within the limit, but the outliers are dropped from dataset

from __future__ import annotations

import argparse
import json
import math
import os

from transformers import AutoTokenizer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

MODEL_ID = "Qwen/Qwen2.5-1.5B"
SPLITS = ("training", "val", "test")
COVERAGE_BAND = (0.95, 0.99)
ROUND_TO = 64


def sequence_length(tokenizer, poem: dict) -> int:
    """Token count of one full training example: prompt in, poem out."""
    messages = [
        {"role": "user", "content": poem["prompt"]},
        {"role": "assistant", "content": poem["text"]},
    ]
    # Render to text, then tokenize: apply_chat_template(tokenize=True) returns
    # a dict rather than a token list in newer transformers releases.
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def percentile(sorted_values: list[int], q: float) -> int:
    """Nearest-rank percentile: the smallest value covering at least q of the data."""
    rank = max(1, math.ceil(q * len(sorted_values)))
    return sorted_values[rank - 1]


def coverage(sorted_values: list[int], limit: int) -> float:
    return sum(1 for v in sorted_values if v <= limit) / len(sorted_values)


def choose_max_length(sorted_values: list[int]) -> int:
    low, high = COVERAGE_BAND
    p99 = percentile(sorted_values, high)
    candidates = [n for n in range(ROUND_TO, p99 + ROUND_TO, ROUND_TO)
                  if low <= coverage(sorted_values, n) <= high]
    return max(candidates) if candidates else p99


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datadir", default=DATA_DIR)
    parser.add_argument("--max-length", type=int,
                        help="use this length instead of choosing one from the data")
    parser.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    splits: dict[str, list[dict]] = {}
    lengths: dict[str, list[int]] = {}
    for split in SPLITS:
        with open(os.path.join(args.datadir, f"processed_{split}.json"), encoding="utf-8") as fh:
            splits[split] = json.load(fh)
        lengths[split] = [sequence_length(tokenizer, p) for p in splits[split]]

    pooled = sorted(n for split in SPLITS for n in lengths[split])
    print(f"Token lengths of {len(pooled)} poems (prompt + poem, {MODEL_ID} chat template)\n")
    print(f"  min {pooled[0]:>6}    mean {sum(pooled) / len(pooled):>8.1f}    max {pooled[-1]:>6}")
    for q in (0.50, 0.90, 0.95, 0.97, 0.98, 0.99):
        print(f"  p{round(q * 100):<3} {percentile(pooled, q):>6}")

    max_length = args.max_length or choose_max_length(pooled)
    how = "given" if args.max_length else "chosen"
    print(f"\nOptimal max sequence length ({how}): {max_length}"
          f"  -- covers {coverage(pooled, max_length):.1%} of poems\n")

    kept: dict[str, list[dict]] = {}
    print(f"{'split':<10} {'before':>7} {'dropped':>8} {'after':>7}")
    for split in SPLITS:
        kept[split] = [p for p, n in zip(splits[split], lengths[split]) if n <= max_length]
        before, after = len(splits[split]), len(kept[split])
        print(f"{split:<10} {before:>7} {before - after:>8} {after:>7}")

    if args.dry_run:
        print("\n(dry run -- nothing written)")
        return

    print()
    for split in SPLITS:
        path = os.path.join(args.datadir, f"processed_{split}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(kept[split], fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        print(f"Wrote {len(kept[split]):>4} poems to {path}")


if __name__ == "__main__":
    main()
