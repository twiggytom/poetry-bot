# assigns unique IDs to each poem for merging

from __future__ import annotations

import argparse
import json
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

FILENAMES = ("training.json", "val.json", "test.json")
PREFIX = "poem_"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datadir", default=DATA_DIR)
    parser.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = parser.parse_args()

    splits: dict[str, list[dict]] = {}
    for name in FILENAMES:
        with open(os.path.join(args.datadir, name), encoding="utf-8") as fh:
            splits[name] = json.load(fh)

    total = sum(len(poems) for poems in splits.values())
    width = max(4, len(str(total)))

    counter = 0
    for name in FILENAMES:
        numbered = []
        for poem in splits[name]:
            counter += 1
            rest = {k: v for k, v in poem.items() if k != "id"}
            numbered.append({"id": f"{PREFIX}{counter:0{width}d}", **rest})
        splits[name] = numbered

    ids = [p["id"] for poems in splits.values() for p in poems]
    assert len(ids) == len(set(ids)) == total, "ids are not unique"

    for name in FILENAMES:
        poems = splits[name]
        span = f"{poems[0]['id']} .. {poems[-1]['id']}" if poems else "(empty)"
        print(f"{name:<14} {len(poems):>5} poems  {span}")

    if args.dry_run:
        print("\n(dry run -- nothing written)")
        return

    for name in FILENAMES:
        path = os.path.join(args.datadir, name)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(splits[name], fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
