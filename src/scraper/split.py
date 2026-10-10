# splits scraped poems into train, val, test

# prevents splitting poems by the same poet across splits, leaks information and inflates performance

from __future__ import annotations

import argparse
import json
import os
import random
import re
import unicodedata

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

SPLITS = ("train", "val", "test")
RATIOS = {"train": 0.80, "val": 0.10, "test": 0.10}
FILENAMES = {"train": "training.json", "val": "val.json", "test": "test.json"}


def poet_key(name: str) -> str:
    """Identity of a poet for grouping: letters only, lowercased, accents folded.

    Collapses the handful of spelling variants in the corpus -- 'Kate DeBolt'
    and 'Kate Debolt', 'Sam Sax' and 'sam sax' -- which would otherwise leak one
    poet across two splits.
    """
    folded = unicodedata.normalize("NFKD", name or "")
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    return re.sub(r"[^a-z]", "", folded.lower())


def group_by_poet(poems: list[dict]) -> dict[str, list[dict]]:
    """Poems bucketed by poet, each bucket keeping its original file order."""
    groups: dict[str, list[dict]] = {}
    for poem in poems:
        groups.setdefault(poet_key(poem["poet"]), []).append(poem)
    return groups


def assign(groups: dict[str, list[dict]], seed: int) -> dict[str, list[str]]:
    """Greedily pack whole poets into the three splits.

    Groups are shuffled for reproducible randomness, then placed largest first;
    each one goes to whichever split is proportionally furthest behind its
    quota. Largest-first matters: leaving the nine-poem poets until the end
    would overshoot whichever split happened to receive them.
    """
    total = sum(len(v) for v in groups.values())
    targets = {s: max(total * RATIOS[s], 1e-9) for s in SPLITS}

    keys = sorted(groups)                       # stable starting order
    random.Random(seed).shuffle(keys)           # ...then a reproducible shuffle
    keys.sort(key=lambda k: len(groups[k]), reverse=True)

    counts = {s: 0 for s in SPLITS}
    chosen: dict[str, list[str]] = {s: [] for s in SPLITS}
    for key in keys:
        size = len(groups[key])
        # The split whose quota stays least full once this poet is added.
        best = min(SPLITS, key=lambda s: ((counts[s] + size) / targets[s], s))
        chosen[best].append(key)
        counts[best] += size
    return chosen


def build(poems: list[dict], seed: int) -> tuple[dict[str, list[dict]], dict[str, list[str]]]:
    groups = group_by_poet(poems)
    chosen = assign(groups, seed)

    # Emit each split in the order the poems appear in the source file.
    position = {id(p): i for i, p in enumerate(poems)}
    out: dict[str, list[dict]] = {}
    for split in SPLITS:
        rows = [poem for key in chosen[split] for poem in groups[key]]
        rows.sort(key=lambda p: position[id(p)])
        out[split] = rows
    return out, chosen


def verify(poems: list[dict], out: dict[str, list[dict]], chosen: dict[str, list[str]]) -> None:
    """Fail loudly rather than write a split that leaks or loses poems."""
    sizes = {s: len(out[s]) for s in SPLITS}
    assert sum(sizes.values()) == len(poems), (
        f"poem count changed: {sum(sizes.values())} != {len(poems)}")

    # Identity is the record itself, not (url, title): a page can carry two
    # companion poems under one title, and they are distinct poems.
    seen = [id(p) for s in SPLITS for p in out[s]]
    assert len(seen) == len(set(seen)), "a poem appears in more than one split"
    assert set(seen) == {id(p) for p in poems}, "a poem went missing"

    poets = {s: set(chosen[s]) for s in SPLITS}
    for a in SPLITS:
        for b in SPLITS:
            if a < b:
                overlap = poets[a] & poets[b]
                assert not overlap, f"poet(s) in both {a} and {b}: {sorted(overlap)[:5]}"

    assert set().union(*poets.values()) == set(group_by_poet(poems)), "poets lost"


def report(poems: list[dict], out: dict[str, list[dict]], chosen: dict[str, list[str]]) -> None:
    total = len(poems)
    print(f"{'split':<8} {'poems':>6} {'share':>8} {'target':>8} {'poets':>7}")
    for split in SPLITS:
        n = len(out[split])
        print(f"{split:<8} {n:>6} {n / total:>7.1%} {RATIOS[split]:>7.0%} {len(chosen[split]):>7}")
    print(f"{'total':<8} {total:>6} {1:>7.0%} {'':>8} {sum(len(chosen[s]) for s in SPLITS):>7}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=os.path.join(DATA_DIR, "poems.json"))
    parser.add_argument("--outdir", default=DATA_DIR)
    parser.add_argument("--seed", type=int, default=42,
                        help="shuffle seed; the same seed always gives the same split")
    parser.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as fh:
        poems = json.load(fh)
    print(f"Read {len(poems)} poems from {args.input}\n")

    out, chosen = build(poems, args.seed)
    verify(poems, out, chosen)
    report(poems, out, chosen)

    if args.dry_run:
        print("\n(dry run -- nothing written)")
        return

    os.makedirs(args.outdir, exist_ok=True)
    print()
    for split in SPLITS:
        path = os.path.join(args.outdir, FILENAMES[split])
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(out[split], fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        print(f"Wrote {len(out[split]):>4} poems to {path}")


if __name__ == "__main__":
    main()
