#!/usr/bin/env bash
set -uo pipefail

if [[ $# -lt 8 ]]; then
    echo "usage: run-basalt.sh STAGE_ROOT STAGE DATASET CONFIG OUTPUT EXECUTABLE KEEP_FAILURE [ARGS...]" >&2
    exit 64
fi

stage_root=$1
stage_directory=$2
source_dataset=$3
source_config=$4
host_output=$5
basalt_executable=$6
keep_failure=$7
shift 7

case "$stage_directory" in
    "$stage_root"/*) ;;
    *)
        echo "refusing stage directory outside configured root: $stage_directory" >&2
        exit 64
        ;;
esac

if [[ ! -x "$basalt_executable" ]]; then
    echo "Basalt executable is unavailable: $basalt_executable" >&2
    exit 69
fi
if [[ ! -d "$source_dataset" ]]; then
    echo "EuRoC source dataset is unavailable: $source_dataset" >&2
    exit 66
fi

mkdir -p "$stage_directory/dataset" "$host_output"
rsync -a --delete "$source_dataset/" "$stage_directory/dataset/"

staged_config=""
if [[ "$source_config" != "-" ]]; then
    if [[ ! -f "$source_config" ]]; then
        echo "Basalt runtime config is unavailable: $source_config" >&2
        exit 66
    fi
    staged_config="$stage_directory/basalt-config.json"
    cp "$source_config" "$staged_config"
fi

command=(
    "$basalt_executable"
    "$@"
    --dataset-path "$stage_directory/dataset"
    --cam-calib "$stage_directory/dataset/calibration.json"
)
if [[ -n "$staged_config" ]]; then
    command+=(--config-path "$staged_config")
fi

{
    echo "backend: wsl"
    printf "command:"
    printf " %q" "${command[@]}"
    printf "\n"
} > "$stage_directory/run.log"

set +e
(
    cd "$stage_directory" || exit 70
    "${command[@]}"
) > "$stage_directory/basalt.stdout.log" 2> "$stage_directory/basalt.stderr.log"
returncode=$?
set -e
echo "returncode: $returncode" >> "$stage_directory/run.log"

cp "$stage_directory/basalt.stdout.log" "$host_output/basalt.stdout.log"
cp "$stage_directory/basalt.stderr.log" "$host_output/basalt.stderr.log"
cp "$stage_directory/run.log" "$host_output/run.log"

if [[ $returncode -eq 0 && -f "$stage_directory/trajectory.csv" ]]; then
    cp "$stage_directory/trajectory.csv" "$host_output/trajectory.csv"
    rm -rf -- "$stage_directory"
    exit 0
fi

if [[ $returncode -eq 0 ]]; then
    echo "Basalt completed without trajectory.csv" >> "$host_output/basalt.stderr.log"
    returncode=65
fi
if [[ "$keep_failure" != "1" ]]; then
    rm -rf -- "$stage_directory"
fi
exit "$returncode"
