"""Launch commands for the Tulu-3 pipeline.

SFT uses the published Llama-3.1-8B recipe: effective batch 128, 2 epochs,
learning rate 5e-6, seed 123. DPO uses the published 8B recipe: dpo_norm,
beta 5, learning rate 5e-7, 1 epoch, effective batch 128.

Current open-instruct parses `--loss_type` and `--beta`. The paper's
experiment1 script still passes the older `--dpo_loss_type` and `--dpo_beta`.
Confirm `--help` on the pinned commit before launching. The check is
`flags_present`.

SimPO is the same script and the same data. Its own hyperparameters are
beta 2 and gamma/beta 0.5. Sharing DPO's beta of 5 would not be SimPO.
The learning rate stays at 5e-7 so the loss is the intended change. If the
SimPO gap then differs from DPO by more than 5 points, protocol.simpo_needs_lr_followup
says to rerun at 1e-6 before treating that difference as an optimizer effect.

KTO is not in open-instruct. The rows are the same pairs, one desirable and
one undesirable each, and the trainer is TRL. That is a second codebase, so
KTO is a weaker match than SimPO.
"""

import shlex

from experiment.optimizer_gap.protocol import (
    DPO_BETA,
    DPO_LR,
    SIMPO_BETA,
    SIMPO_FOLLOWUP_LR,
    SIMPO_GAMMA_BETA_RATIO,
)

SFT_MODEL = "meta-llama/Llama-3.1-8B"
SFT_DATA = "allenai/tulu-3-sft-mixture"
PREFERENCE_DATA = "allenai/llama-3.1-tulu-3-8b-preference-mixture"
EFFECTIVE_BATCH = 128
SFT_LR = 5e-6
SFT_EPOCHS = 2
SFT_SEED = 123
PREFERENCE_SEED = 42
PREFERENCE_EPOCHS = 1
MAX_SEQ_SFT = 4096
MAX_SEQ_PREFERENCE = 2048

# Second family. OLMo-2 7B is the other base model in the Tulu-3 recipe.
# Qwen is not, because this preference mixture was built for the Tulu chat template.
SCALE_FAMILY = "allenai/OLMo-2-1124-7B"

SAFETY_SUITE = "safety::olmo3"
CAPABILITY_SUITE = "tulu_3_dev"


def grad_accum(n_gpus, effective_batch=EFFECTIVE_BATCH):
    if n_gpus <= 0 or effective_batch % n_gpus != 0:
        raise ValueError(
            f"{n_gpus} GPUs do not divide an effective batch of {effective_batch}"
        )
    return effective_batch // n_gpus


def _accelerate(n_gpus, script, args):
    accum = grad_accum(n_gpus)
    flags = [
        "accelerate", "launch",
        "--mixed_precision", "bf16",
        "--num_machines", "1",
        "--num_processes", str(n_gpus),
        "--use_deepspeed",
        "--deepspeed_config_file", "configs/ds_configs/stage3_no_offloading_accelerate.conf",
        script,
    ]
    for key, value in args:
        flags.append(f"--{key}")
        if isinstance(value, (list, tuple)):
            flags.extend(str(part) for part in value)
        elif value is not None:
            flags.append(str(value))
    rendered = shlex.join(flags)
    return {"argv": flags, "command": rendered, "gradient_accumulation_steps": accum, "effective_batch": EFFECTIVE_BATCH}


def sft_command(n_gpus, output_dir):
    accum = grad_accum(n_gpus)
    return _accelerate(n_gpus, "open_instruct/finetune.py", [
        ("model_name_or_path", SFT_MODEL),
        ("tokenizer_name", SFT_MODEL),
        ("use_slow_tokenizer", None),
        ("max_seq_length", MAX_SEQ_SFT),
        ("per_device_train_batch_size", 1),
        ("gradient_accumulation_steps", accum),
        ("learning_rate", SFT_LR),
        ("lr_scheduler_type", "linear"),
        ("warmup_ratio", 0.03),
        ("weight_decay", 0.0),
        ("num_train_epochs", SFT_EPOCHS),
        ("output_dir", output_dir),
        ("dataset_mixer_list", [SFT_DATA, "1.0"]),
        ("checkpointing_steps", "epoch"),
        ("seed", SFT_SEED),
        ("exp_name", "optimizer-gap-sft-8b"),
    ])


def preference_command(optimizer, n_gpus, model, data, output_dir, learning_rate=None):
    """optimizer is dpo, simpo, or simpo_lr_followup. data is a hub id or a local jsonl."""
    if optimizer not in ("dpo", "simpo", "simpo_lr_followup"):
        raise ValueError(f"{optimizer} is not an open-instruct preference loss")
    accum = grad_accum(n_gpus)
    if optimizer == "dpo":
        loss = "dpo_norm"
        beta = DPO_BETA
        gamma = None
        lr = DPO_LR if learning_rate is None else learning_rate
    else:
        loss = "simpo"
        beta = SIMPO_BETA
        gamma = SIMPO_GAMMA_BETA_RATIO
        default_lr = SIMPO_FOLLOWUP_LR if optimizer == "simpo_lr_followup" else DPO_LR
        lr = default_lr if learning_rate is None else learning_rate
    args = [
        ("model_name_or_path", model),
        ("tokenizer_name", model),
        ("max_seq_length", MAX_SEQ_PREFERENCE),
        ("per_device_train_batch_size", 1),
        ("gradient_accumulation_steps", accum),
        ("learning_rate", lr),
        ("lr_scheduler_type", "linear"),
        ("warmup_ratio", 0.1),
        ("weight_decay", 0.0),
        ("num_epochs", PREFERENCE_EPOCHS),
        ("output_dir", output_dir),
        ("mixer_list", [data, "1.0"]),
        ("mixer_list_splits", "train"),
        ("loss_type", loss),
        ("beta", beta),
        # This commit checkpoints when activation_memory_budget is below 1.
        # It has no --gradient_checkpointing flag. 0.5 is the script's own
        # "turn it on" value.
        ("activation_memory_budget", 0.5),
        ("use_lora", "False"),
        ("seed", PREFERENCE_SEED),
        ("exp_name", f"optimizer-gap-{optimizer}"),
    ]
    if gamma is not None:
        args.append(("gamma_beta_ratio", gamma))
    command = _accelerate(n_gpus, "open_instruct/dpo_tune_cache.py", args)
    command["loss_type"] = loss
    command["beta"] = beta
    command["learning_rate"] = lr
    return command


def kto_command(model, data_jsonl, output_dir):
    return {
        "trainer": "trl.KTOTrainer",
        "model": model,
        "data": data_jsonl,
        "output_dir": output_dir,
        "num_train_epochs": PREFERENCE_EPOCHS,
        "learning_rate": DPO_LR,
        "per_device_train_batch_size": 1,
        "seed": PREFERENCE_SEED,
        "desirable_weight": 1.0,
        "undesirable_weight": 1.0,
        "note": "Same pairs as DPO, split into one desirable and one undesirable row. TRL, not open-instruct.",
    }


def olmes_command(model, suite):
    if suite not in (SAFETY_SUITE, CAPABILITY_SUITE):
        raise ValueError(f"unknown suite {suite}")
    return ["olmes", "--model", model, "--task", suite]


def flags_present(help_text, command):
    """True when every --flag in the command appears in the pinned script's --help."""
    missing = []
    argv = command["argv"]
    for token in argv:
        if token.startswith("--") and token not in help_text:
            missing.append(token)
    return missing
