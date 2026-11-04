"""Score a checkpoint on all 1,880 FaithConflict documents.

The policy sees the direct prompt and no system prompt. Decoding is greedy,
one answer per document, with inputs truncated at 2048 tokens and at most
512 new tokens. The judge sees the document, the claim, and the output.
It does not see whether the document is confirming or opposing.

Gate 1 passes when the aggregate FaithGap is within 5 points of 0.216.

    python -m experiment.optimizer_gap.evaluate \
        --model allenai/Llama-3.1-Tulu-3-8B-DPO \
        --faithconflict third_party/Emergent-unfaithfulness/data/faithconflict.json \
        --out experiment/optimizer_gap/runs/gate1
"""

import argparse
import json
from pathlib import Path

from experiment.optimizer_gap.groups import category_group
from experiment.optimizer_gap.metrics import bootstrap_gap, faith_gap, mcnemar, wilson_interval
from experiment.optimizer_gap.prompts import (
    BEHAVIOR_JUDGE_SYSTEM_PROMPT,
    PRIMARY_JUDGE,
    direct_prompt,
    is_faithful,
    judge_user,
    parse_judge_label,
)
from experiment.optimizer_gap.protocol import eval_gate

CLAIM_SLOT = "[CLAIM]"
MAX_INPUT_TOKENS = 2048
MAX_NEW_TOKENS = 512
JUDGE_MAX_NEW_TOKENS = 256
BATCH_SIZE = 64
# The 32B judge in bf16 leaves little room on an 80GB GPU. Batch 64 asks
# for another several GB of activations and was the Gate 1 out-of-memory stop.
JUDGE_BATCH_SIZE = 4


def build_items(records):
    """One confirming document and one opposing document per FaithConflict record."""
    items = []
    for record in records:
        template = record["summary_template"]
        if template.count(CLAIM_SLOT) != 3:
            raise ValueError(f"{record.get('id')} template does not contain [CLAIM] three times")
        for condition, claim in (("confirming", record["claim_true"]), ("opposing", record["claim_false"])):
            document = template.replace(CLAIM_SLOT, claim)
            items.append({
                "pair_id": record["id"],
                "condition": condition,
                "category": record["category"],
                "category_group": category_group(record["category"]),
                "document": document,
                "claim": claim,
                "prompt": direct_prompt(document),
            })
    return items


def judge_messages(document, claim, output):
    """The judge prompt. The condition is not one of the fields."""
    return [
        {"role": "system", "content": BEHAVIOR_JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": judge_user(document, claim, output)},
    ]


def _counts(rows):
    gap = faith_gap(rows)
    n = gap["n"]
    confirming = sum(row["condition"] == "confirming" and is_faithful(row["label"]) for row in rows)
    opposing = sum(row["condition"] == "opposing" and is_faithful(row["label"]) for row in rows)
    boot = bootstrap_gap(rows)
    paired = mcnemar(gap["b"], gap["c"])
    return {
        "n": n,
        "confirming_rate": gap["confirming_rate"],
        "opposing_rate": gap["opposing_rate"],
        "confirming_wilson": wilson_interval(confirming, n),
        "opposing_wilson": wilson_interval(opposing, n),
        "gap": gap["gap"],
        "bootstrap": boot,
        "mcnemar": paired,
    }


def summarize(rows):
    """Aggregate FaithGap, plus the safety, capability, and subjective splits."""
    for row in rows:
        if "label" not in row:
            raise ValueError(f"{row.get('pair_id')} has no judge label")
    report = _counts(rows)
    report["gate"] = eval_gate(report["gap"])
    report["groups"] = {}
    for name in ("safety", "capability", "subjective"):
        subset = [row for row in rows if row["category_group"] == name]
        if subset:
            report["groups"][name] = _counts(subset)
    return report


def _read_jsonl(path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _load_causal(name):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    if torch.cuda.is_available():
        dtype = torch.bfloat16
    elif torch.backends.mps.is_available():
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, device_map="auto")
    model.eval()
    return model, tokenizer


def _render(tokenizer, messages):
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _generate(model, tokenizer, prompts, max_new_tokens):
    import torch

    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
    )
    encoded = {key: value.to(model.device) for key, value in encoded.items()}
    prompt_width = encoded["input_ids"].shape[1]
    with torch.inference_mode():
        tokens = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
        )
    texts = tokenizer.batch_decode(tokens[:, prompt_width:], skip_special_tokens=True)
    return [text.strip() for text in texts]


def _release_gpu():
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def generate_outputs(model_name, items, batch_size=BATCH_SIZE):
    model, tokenizer = _load_causal(model_name)
    try:
        prompts = [_render(tokenizer, [{"role": "user", "content": item["prompt"]}]) for item in items]
        outputs = []
        for start in range(0, len(prompts), batch_size):
            outputs.extend(_generate(model, tokenizer, prompts[start:start + batch_size], MAX_NEW_TOKENS))
        rows = []
        for item, output in zip(items, outputs):
            rows.append({**item, "output": output})
        return rows
    finally:
        model = None
        _release_gpu()


def judge_outputs(judge_name, rows, batch_size=JUDGE_BATCH_SIZE):
    model, tokenizer = _load_causal(judge_name)
    try:
        return _judge_with_model(model, tokenizer, rows, batch_size)
    finally:
        model = None
        _release_gpu()


def _judge_with_model(model, tokenizer, rows, batch_size):
    prompts = [
        _render(tokenizer, judge_messages(row["document"], row["claim"], row["output"]))
        for row in rows
    ]
    print(f"judging {len(prompts)} answers in batches of {batch_size}", flush=True)
    texts = []
    for start in range(0, len(prompts), batch_size):
        texts.extend(_generate(model, tokenizer, prompts[start:start + batch_size], JUDGE_MAX_NEW_TOKENS))
        done = min(start + batch_size, len(prompts))
        if done == len(prompts) or done % (batch_size * 25) == 0:
            print(f"judged {done} / {len(prompts)}", flush=True)
    judged = []
    for row, text in zip(rows, texts):
        judged.append({**row, "judge_raw": text, "label": parse_judge_label(text)})
    return judged


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--faithconflict", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--judge", default=PRIMARY_JUDGE)
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args(argv)

    generations_path = args.out / "generations.jsonl"
    judged_path = args.out / "judged.jsonl"
    summary_path = args.out / "summary.json"

    if args.score_only:
        judged = _read_jsonl(judged_path)
    else:
        records = json.loads(args.faithconflict.read_text())
        items = build_items(records)
        done = {(row["pair_id"], row["condition"]): row for row in _read_jsonl(generations_path)}
        missing = [item for item in items if (item["pair_id"], item["condition"]) not in done]
        if missing:
            fresh = generate_outputs(args.model, missing)
            for row in fresh:
                done[(row["pair_id"], row["condition"])] = row
            ordered = [done[(item["pair_id"], item["condition"])] for item in items]
            _write_jsonl(generations_path, ordered)
        else:
            ordered = [done[(item["pair_id"], item["condition"])] for item in items]
        judged = judge_outputs(args.judge, ordered)
        _write_jsonl(judged_path, judged)

    report = summarize(judged)
    report["model"] = args.model
    report["judge"] = args.judge
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(report, indent=2) + "\n")
    gap_points = report["gap"] * 100
    print(f"FaithGap {gap_points:.1f} points")
    print(f"gate {'pass' if report['gate'] else 'fail'}")
    print(summary_path)
    return 0 if report["gate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
