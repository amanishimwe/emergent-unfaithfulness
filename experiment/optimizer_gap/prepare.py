"""Write the rebuilt preference files from a FaithConflict JSON list.

The original Tulu mixture is not rewritten. Training reads it from the hub.
This script writes the added faithfulness pairs, and the KTO view of those pairs.

Usage, from the repo root, after third_party/Emergent-unfaithfulness is cloned:

    python -m experiment.optimizer_gap.prepare \
        --faithconflict third_party/Emergent-unfaithfulness/data/faithconflict.json \
        --out experiment/optimizer_gap/data
"""

import argparse
import json
from pathlib import Path

from experiment.optimizer_gap.pairs import kto_rows, rebuilt_pairs


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--faithconflict", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    records = json.loads(args.faithconflict.read_text())
    pairs = rebuilt_pairs(records)
    kto = kto_rows(pairs)
    pairs_path = args.out / "rebuilt_pairs.jsonl"
    kto_path = args.out / "rebuilt_kto.jsonl"
    write_jsonl(pairs_path, pairs)
    write_jsonl(kto_path, kto)
    print(f"{len(records)} documents")
    print(f"{len(pairs)} preference pairs -> {pairs_path}")
    print(f"{len(kto)} KTO rows -> {kto_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
