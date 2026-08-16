#!/usr/bin/env bash

legal_flux_start_vllm() {
  if (( $# < 2 )); then
    echo "Usage: legal_flux_start_vllm PORT LOG_FILE [vLLM arguments...]" >&2
    return 2
  fi

  local port="$1"
  local log_file="$2"
  shift 2

  : "${LEGAL_FLUX_WORK_ROOT:?LEGAL_FLUX_WORK_ROOT must be set}"
  : "${HF_HOME:?HF_HOME must be set}"
  : "${LEGAL_FLUX_MODEL_NAME:?LEGAL_FLUX_MODEL_NAME must be set}"

  local container="${LEGAL_FLUX_VLLM_CONTAINER:-${LEGAL_FLUX_WORK_ROOT}/containers/vllm-openai-v0.21.0.sif}"
  if [[ ! -r "$container" ]]; then
    echo "vLLM container is missing: ${container}" >&2
    echo "Run scripts/cluster/setup_delta_eval.sh before submitting GPU jobs." >&2
    return 1
  fi
  if ! command -v apptainer >/dev/null 2>&1; then
    echo "Apptainer is unavailable on this node." >&2
    return 1
  fi

  local expected_version="${LEGAL_FLUX_VLLM_VERSION:-0.21.0}"
  local version_output
  if ! version_output="$(
    apptainer exec --cleanenv "$container" /bin/bash -c '
      for metadata in \
        /usr/local/lib/python*/dist-packages/vllm-*.dist-info/METADATA \
        /usr/local/lib/python*/site-packages/vllm-*.dist-info/METADATA; do
        if [ -f "$metadata" ]; then
          sed -n "s/^Version: //p" "$metadata" | head -n 1
          exit 0
        fi
      done
      exit 1
    '
  )"; then
    echo "Could not read vLLM package metadata from ${container}." >&2
    return 1
  fi
  if [[ "$version_output" != "$expected_version" ]]; then
    echo "Expected vLLM ${expected_version}, but ${container} reported:" >&2
    echo "$version_output" >&2
    return 1
  fi

  mkdir -p "$(dirname "$log_file")"
  local -a container_env=(
    --env "HF_HOME=${HF_HOME}"
    --env "TOKENIZERS_PARALLELISM=false"
  )
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    container_env+=(--env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}")
  fi

  apptainer exec \
    --nv \
    --cleanenv \
    --bind "${LEGAL_FLUX_WORK_ROOT}:${LEGAL_FLUX_WORK_ROOT}" \
    "${container_env[@]}" \
    "$container" \
    /bin/bash -c \
      'export LD_LIBRARY_PATH="/usr/local/cuda/compat:${LD_LIBRARY_PATH:-}"; exec vllm serve "$@"' \
    legalflux-vllm \
    "$LEGAL_FLUX_MODEL_NAME" \
    --host 127.0.0.1 \
    --port "$port" \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.90 \
    --reasoning-parser qwen3 \
    --language-model-only \
    "$@" \
    > "$log_file" 2>&1 &
  LEGAL_FLUX_VLLM_PID=$!
}


legal_flux_wait_for_vllm() {
  if (( $# != 3 )); then
    echo "Usage: legal_flux_wait_for_vllm PID BASE_URL LOG_FILE" >&2
    return 2
  fi

  local server_pid="$1"
  local base_url="$2"
  local log_file="$3"
  local timeout_seconds="${LEGAL_FLUX_VLLM_READY_TIMEOUT_SECONDS:-1800}"
  local started_at=$SECONDS
  local deadline=$((started_at + timeout_seconds))

  until curl --silent --fail --max-time 5 "${base_url}/models" >/dev/null; do
    if ! kill -0 "$server_pid" 2>/dev/null; then
      echo "vLLM exited before becoming ready. Last 200 log lines:" >&2
      tail -n 200 "$log_file" >&2 || true
      return 1
    fi
    if (( SECONDS >= deadline )); then
      echo "vLLM did not become ready within ${timeout_seconds} seconds." >&2
      echo "Last 200 log lines:" >&2
      tail -n 200 "$log_file" >&2 || true
      return 1
    fi
    sleep 10
  done

  echo "vLLM is ready after $((SECONDS - started_at)) seconds."
}
