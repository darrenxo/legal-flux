#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export LEGAL_BENCHMARK_SUBSET=full
export LEGAL_BENCHMARK_RUN_TAG="${LEGAL_BENCHMARK_RUN_TAG:-delta-bf16-three-dataset-full-v1}"
export LEGAL_BENCHMARK_DATASETS="${LEGAL_BENCHMARK_DATASETS:-annocaselaw realistic_ljp_facts il_tur_cjpe}"

bash "${SCRIPT_DIR}/submit_legal_benchmarks.sh"
