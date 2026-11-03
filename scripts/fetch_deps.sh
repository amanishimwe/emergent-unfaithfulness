#!/usr/bin/env bash
# Clone the two repositories this experiment trains against, and record the commit.
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

{
  echo "faithconflict $(clone https://github.com/uiuc-conversational-ai-lab/Emergent-unfaithfulness.git Emergent-unfaithfulness)"
  echo "open-instruct $(clone https://github.com/allenai/open-instruct.git open-instruct)"
} | tee "$dest/PINS.txt"
