#!/usr/bin/env bash
# Sync the outputs of a tagged set of experiment configs into manuscript/references/.
#
# Usage: ./sync_references.sh <tag>
#
# Scans module/experiments/*.yml (top-level configs only, not the copies each
# report script leaves behind inside its own output dir) for files named
# *_<tag>.yml, resolves each one's output directory using the same
# classification_type/stage/model.code/tag logic the *report.py scripts use
# to build it, and syncs that directory into manuscript/references/<stage>/
# (classification_type, model.code, and tag are all dropped from the
# destination path; the source config copy that already lives inside each
# stage's own output dir travels along with it, so the top-level configs
# aren't copied separately). Since the tag isn't part of the destination
# path, re-running with a different tag replaces the prior tag's synced
# outputs — manuscript/references/ always reflects exactly one tag per stage.
#
# counterfactuals gets special handling: the split{N}/nofiltering/<patient>/
# nesting is flattened directly under counterfactuals/, and only patient
# folders containing a .png (i.e. a counterfactual was actually found) are
# copied. The per-split global CSVs (which sit in split{N}/, not in
# nofiltering/) are flattened alongside them; their filenames already carry
# the split index so there's no collision.
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

# Resolve a config's source output dir (relative to experiments/) and its
# destination dir (relative to references/, with classification_type and
# model.code dropped), using the same fields each *report.py script uses to
# build `outputdir`. Prints: stage, src_rel, dst_rel (one per line).
resolve_dirs() {
    python3 - "$1" <<'EOF'
import sys
from pathlib import Path

import yaml

with open(sys.argv[1]) as f:
    config = yaml.safe_load(f)

experiment = config["experiment"]
classification_type = experiment["classification_type"]
stage = experiment["stage"]
tag = experiment["tag"]

if stage == "eda":
    output_subdir = config["reporting"]["output_subdir"]
    src_rel = Path(classification_type) / output_subdir
    dst_rel = Path(output_subdir)
elif stage == "selection":
    src_rel = Path(classification_type) / stage / tag
    dst_rel = Path(stage)
else:
    model_code = config["model"]["code"]
    src_rel = Path(classification_type) / stage / model_code / tag
    dst_rel = Path(stage)

print(stage)
print(src_rel)
print(dst_rel)
EOF
}

sync_counterfactuals() {
    local src="$1" dst="$2"

    mkdir -p "$dst"

    shopt -s nullglob
    local top_ymls=("$src"/*.yml)
    shopt -u nullglob
    if [[ ${#top_ymls[@]} -gt 0 ]]; then
        rsync -av "${top_ymls[@]}" "$dst/"
    fi

    # Patient dirs are named after zero-padded numeric patient codes; track
    # which ones currently have a valid counterfactual so stale ones (no
    # longer present, or no longer containing a .png) can be pruned below
    # without wiping and re-transferring the whole dst on every run.
    local valid_patients=()

    for split_dir in "$src"/split*/; do
        [[ -d "$split_dir" ]] || continue

        shopt -s nullglob
        local split_csvs=("$split_dir"*.csv)
        shopt -u nullglob
        if [[ ${#split_csvs[@]} -gt 0 ]]; then
            rsync -av "${split_csvs[@]}" "$dst/"
        fi

        local nofiltering_dir="$split_dir/nofiltering"
        [[ -d "$nofiltering_dir" ]] || continue

        for patient_dir in "$nofiltering_dir"/*/; do
            [[ -d "$patient_dir" ]] || continue
            shopt -s nullglob
            local pngs=("$patient_dir"*.png)
            shopt -u nullglob
            if [[ ${#pngs[@]} -gt 0 ]]; then
                local patient_code
                patient_code="$(basename "$patient_dir")"
                valid_patients+=("$patient_code")
                rsync -av --delete "$patient_dir" "$dst/$patient_code/"
            fi
        done
    done

    for existing_dir in "$dst"/*/; do
        [[ -d "$existing_dir" ]] || continue
        local existing_code
        existing_code="$(basename "$existing_dir")"
        [[ "$existing_code" =~ ^[0-9]+$ ]] || continue
        if ! printf '%s\n' "${valid_patients[@]:-}" | grep -qx "$existing_code"; then
            echo "Pruning stale patient dir $existing_code (no longer has a valid counterfactual)"
            rm -rf "$existing_dir"
        fi
    done
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

    readarray -t resolved < <(resolve_dirs "$config_path")
    stage="${resolved[0]}"
    src_rel="${resolved[1]}"
    dst_rel="${resolved[2]}"

    src="$EXPERIMENTS_DIR/$src_rel"
    dst="$REFERENCES_DIR/$dst_rel"

    if [[ ! -d "$src" ]]; then
        echo "Skipping $config_name: output dir $src does not exist" >&2
        continue
    fi

    echo "$config_name -> $dst_rel"
    if [[ "$stage" == "counterfactuals" ]]; then
        sync_counterfactuals "$src" "$dst"
    else
        mkdir -p "$dst"
        rsync -av --delete "$src/" "$dst/"
    fi
done
