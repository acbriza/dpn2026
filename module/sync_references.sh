#!/usr/bin/env bash
# Sync the outputs of a tagged set of experiment configs into manuscript/references/.
#
# Usage: ./sync_references.sh <tag>
#
# Scans module/experiments/*.yml (top-level configs only, not the copies each
# report script leaves behind inside its own output dir) for files named
# *_<tag>.yml, resolves each one's output directory using the same
# classification_type/stage/model.code/tag logic the *report.py scripts use
# to build it, and rsyncs that directory into manuscript/references/ at the
# matching path (relative to experiments/). The matched top-level config
# files themselves are copied flat into manuscript/references/.
set -euo pipefail

if [[ $# -ne 1 || -z "$1" ]]; then
    echo "Usage: $0 <tag>" >&2
    exit 1
fi
TAG="$1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENTS_DIR="$SCRIPT_DIR/experiments"
REFERENCES_DIR="$SCRIPT_DIR/../manuscript/references"

mkdir -p "$REFERENCES_DIR"

# Resolve a config's output directory, relative to experiments/, using the
# same fields each *report.py script uses to build `outputdir`.
resolve_outputdir() {
    python3 - "$1" <<'EOF'
import sys, yaml
from pathlib import Path

with open(sys.argv[1]) as f:
    config = yaml.safe_load(f)

experiment = config["experiment"]
classification_type = experiment["classification_type"]
stage = experiment["stage"]
tag = experiment["tag"]

if stage == "eda":
    rel = Path(classification_type) / config["reporting"]["output_subdir"]
elif stage == "selection":
    rel = Path(classification_type) / stage / tag
else:
    rel = Path(classification_type) / stage / config["model"]["code"] / tag

print(rel)
EOF
}

shopt -s nullglob
configs=("$EXPERIMENTS_DIR"/*"_${TAG}.yml")
shopt -u nullglob

if [[ ${#configs[@]} -eq 0 ]]; then
    echo "No configs matching *_${TAG}.yml found in $EXPERIMENTS_DIR" >&2
    exit 1
fi

for config_path in "${configs[@]}"; do
    config_name="$(basename "$config_path")"
    config_tag="$(python3 -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['experiment']['tag'])" "$config_path")"
    if [[ "$config_tag" != "$TAG" ]]; then
        echo "Skipping $config_name: experiment.tag ('$config_tag') does not match requested tag ('$TAG')" >&2
        continue
    fi

    rel_outputdir="$(resolve_outputdir "$config_path")"
    src="$EXPERIMENTS_DIR/$rel_outputdir"
    dst="$REFERENCES_DIR/$rel_outputdir"

    if [[ ! -d "$src" ]]; then
        echo "Skipping $config_name: output dir $src does not exist" >&2
        continue
    fi

    echo "$config_name -> $rel_outputdir"
    mkdir -p "$dst"
    rsync -av --delete "$src/" "$dst/"

    rsync -av "$config_path" "$REFERENCES_DIR/"
done
