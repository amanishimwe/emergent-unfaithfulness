#!/usr/bin/env bash
# Clone the two repositories this experiment trains against, and record the commit.
# Each checkout is then limited to the paths this experiment actually reads.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
dest="$root/third_party"
mkdir -p "$dest"

clone() {
  local url="$1"
  local name="$2"
  if [[ ! -d "$dest/$name/.git" ]]; then
    git clone "$url" "$dest/$name"
  fi
  git -C "$dest/$name" rev-parse HEAD
}

# FaithConflict is the document source. The paper's other experiments are not run.
faithconflict_paths() {
  echo /data/faithconflict.json
}

# SFT and DPO/SimPO, plus the Tulu-3 8B and OLMo-2 7B recipes.
# Paths below are the import closure of those two trainers.
open_instruct_paths() {
  cat <<'EOF'
/.gitignore
/LICENSE
/README.md
/configs/ds_configs/stage3_no_offloading_accelerate.conf
/configs/train_configs/olmo2/olmo2_1124_7b_dpo.yaml
/configs/train_configs/olmo2/olmo2_1124_7b_sft.yaml
/configs/train_configs/tulu3/tulu3_dpo_8b.yaml
/configs/train_configs/tulu3/tulu3_sft.yaml
/mason.py
/open_instruct/IFEvalG/instructions.py
/open_instruct/IFEvalG/instructions_registry.py
/open_instruct/IFEvalG/instructions_util.py
/open_instruct/__init__.py
/open_instruct/answer_utils.py
/open_instruct/context_window_checker.py
/open_instruct/data_types.py
/open_instruct/dataset_statistics.py
/open_instruct/dataset_transformation.py
/open_instruct/dpo_tune_cache.py
/open_instruct/dpo_utils.py
/open_instruct/finetune.py
/open_instruct/ground_truth_utils.py
/open_instruct/if_functions.py
/open_instruct/judge_utils.py
/open_instruct/launch_utils.py
/open_instruct/logger_utils.py
/open_instruct/math_utils.py
/open_instruct/model_utils.py
/open_instruct/olmo_core_callbacks.py
/open_instruct/olmo_core_hybrid.py
/open_instruct/olmo_core_utils.py
/open_instruct/padding_free_collator.py
/open_instruct/rubrics/prompts.py
/open_instruct/rubrics/run_utils.py
/open_instruct/tensor_utils.py
/open_instruct/tokenizer_utils.py
/open_instruct/utils.py
/pyproject.toml
/requirements.txt
/scripts/train/olmo2/dpo_7b.sh
/scripts/train/olmo2/finetune_7b.sh
/scripts/train/tulu3/dpo_8b.sh
/scripts/train/tulu3/finetune_8b.sh
/uv.lock
EOF
}

limit_checkout() {
  local name="$1"
  shift
  git -C "$dest/$name" sparse-checkout set --no-cone --stdin <<<"$*"
}

{
  echo "faithconflict $(clone https://github.com/uiuc-conversational-ai-lab/Emergent-unfaithfulness.git Emergent-unfaithfulness)"
  echo "open-instruct $(clone https://github.com/allenai/open-instruct.git open-instruct)"
} | tee "$dest/PINS.txt"

limit_checkout Emergent-unfaithfulness "$(faithconflict_paths)"
limit_checkout open-instruct "$(open_instruct_paths)"
