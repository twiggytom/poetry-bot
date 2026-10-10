# attaches generated prompts from LLM to id in data



from __future__ import annotations

import argparse
import json
import os


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

SPLITS = ("training", "val", "test")


def load(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def combine(poems: list[dict], prompts: list[dict], split: str) -> list[dict]:
    """Poems in their original order, each with its prompt placed after the id.

    Fails loudly if any poem lacks a prompt, any prompt lacks a poem, or an id
    repeats -- a silent mismatch would pair a poem with the wrong request.
    """
    by_id: dict[str, str] = {}
    for row in prompts:
        assert row["id"] not in by_id, f"{split}: duplicate prompt id {row['id']}"
        assert (row.get("prompt") or "").strip(), f"{split}: empty prompt for {row['id']}"
        by_id[row["id"]] = row["prompt"]

    poem_ids = [p["id"] for p in poems]
    assert len(poem_ids) == len(set(poem_ids)), f"{split}: duplicate poem id"
    missing = set(poem_ids) - set(by_id)
    extra = set(by_id) - set(poem_ids)
    assert not missing, f"{split}: poems without a prompt: {sorted(missing)[:5]}"
    assert not extra, f"{split}: prompts without a poem: {sorted(extra)[:5]}"

    out = []
    for poem in poems:
        rest = {k: v for k, v in poem.items() if k not in ("id", "prompt")}
        out.append({"id": poem["id"], "prompt": by_id[poem["id"]], **rest})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datadir", default=DATA_DIR)
    parser.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = parser.parse_args()

    results: dict[str, list[dict]] = {}
    for split in SPLITS:
        poems = load(os.path.join(args.datadir, f"{split}.json"))
        prompts = load(os.path.join(args.datadir, f"{split}_prompts.json"))
        results[split] = combine(poems, prompts, split)
        print(f"{split:<10} {len(results[split]):>5} poems matched to prompts")

    if args.dry_run:
        print("\n(dry run -- nothing written)")
        return

    print()
    for split in SPLITS:
        path = os.path.join(args.datadir, f"processed_{split}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(results[split], fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        print(f"Wrote {len(results[split]):>4} poems to {path}")


if __name__ == "__main__":
    main()
