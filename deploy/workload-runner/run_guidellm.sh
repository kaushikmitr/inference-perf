#!/bin/bash
# guidellm-based saturation sweep for the 10× Qwen3-32B cluster.
# Runs IN-CLUSTER as a kubectl Job per (scenario, c) — same network path as
# the previous inference-perf benchmarks, no external-LB hop.
#
# Replaces the inference-perf multi-stage runner (which has known hangs).
# Each scenario × concurrency-level combination is its own guidellm Job,
# producing one JSON report per (scenario, c) in GCS. With --random-seed=42
# across invocations, the synthetic 80-prefix conversation pool is byte-
# identical between runs, so vLLM's prefix cache reuses across runs the same
# way multi-stage would have.
#
# Workload mapping (from inference-perf code-generation.yaml):
#   num_conversations=80          → prefix_count=80
#   shared+dynamic system prompt  → prefix_tokens=58000  (3K shared + 55K mean)
#   input_tokens_per_turn         → prompt_tokens=1500
#   output_tokens_per_turn        → output_tokens=800
#   turns_per_conversation        → turns=3              (matches our RPS=3)
#   tool_call_latency_sec         → DROPPED              (intentionally)
#
# Prerequisites:
#   * guidellm image pushed to GAR (build with Dockerfile.guidellm).
#   * gmp-test-sa service account has GCS write permission (already true).
#
# Launch:
#   nohup bash deploy/workload-runner/run_guidellm.sh > /tmp/guidellm.log 2>&1 &

set -euo pipefail

INTERNAL_IP="http://35.240.207.135:80"   # direct vLLM (bypass EPP)
EXTERNAL_IP="http://34.124.184.34:80"    # through EPP
MODEL="Qwen/Qwen3-32B"
DEPLOYMENT_NAME="predicted-latency-based-scheduling-epp"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GCS_BUCKET="${GCS_BUCKET:-kaushikmitra-llm-ig-benchmark}"
GCS_OUTPUT_DIR="${GCS_OUTPUT_DIR:-guidellm-runs}"

CONCURRENCY_LEVELS="${CONCURRENCY_LEVELS:-10 20 30 40 50 60 70 80}"
TURNS_PER_CONV="${TURNS_PER_CONV:-3}"
NUM_CONVERSATIONS="${NUM_CONVERSATIONS:-80}"
PROMPT_TOKENS="${PROMPT_TOKENS:-1500}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-800}"
PREFIX_TOKENS="${PREFIX_TOKENS:-58000}"
# RANDOM_SEED is set per-Job (each (scenario, c-level) gets its own seed
# from the Job's submit timestamp). Different prefixes AND different user
# prompts across c-levels → no cross-c cache contamination: c=20's TTFT is
# NOT artificially lowered by cache warmed during c=10. Trade-off: each Job
# cold-starts vLLM's prefix cache from scratch; we lose the "warm cache
# across c-levels within a scenario" effect that multi-stage had.

SCENARIOS=(
  "run=code-generation-guidellm-epp-penalty64k-2 url=$EXTERNAL_IP skip_config=false penalty=64000"
  "run=code-generation-guidellm-epp-penalty96k-2 url=$EXTERNAL_IP skip_config=false penalty=96000"
  "run=code-generation-guidellm-baseline-2       url=$INTERNAL_IP skip_config=true"
)

apply_epp_config() {
  local penalty=$1
  export MAX_TOKENS_IN_FLIGHT_PENALTY="$penalty"
  export WEIGHT_PREFIX_CACHE=1 WEIGHT_QUEUE=0 WEIGHT_KV_UTIL=1
  export AFFINITY_GATE_TAU=0.8 AFFINITY_GATE_TAU_GLOBAL=0.99
  export MAX_CONCURRENCY=8192 HEADROOM=3
  export FLOW_CONTROL_GATE="- flowControl"
  echo "[1/3] Applying EndpointPickerConfig (penalty=$penalty)..."
  envsubst < "$SCRIPT_DIR/server-config-no-predicted-latency.yaml" | kubectl apply -f -
  echo "[2/3] Restarting Deployment $DEPLOYMENT_NAME..."
  kubectl rollout restart "deployment/$DEPLOYMENT_NAME"
  kubectl rollout status "deployment/$DEPLOYMENT_NAME" --timeout=10m
  echo "      Waiting 30s for endpoints to stabilize..."
  sleep 30
}

run_one() {
  local run_name=$1 url=$2 conc=$3 seed=$4
  local suffix
  suffix=$(date +%s)
  local job_name="guidellm-${suffix}"
  local max_requests=$((NUM_CONVERSATIONS * TURNS_PER_CONV))

  echo "    → c=$conc  max_requests=$max_requests  seed=$seed  target=$url  job=$job_name"

  # Allowlist mode: only substitute the named vars. Without this, envsubst would
  # also eat ${DATA} inside the bash heredoc body, leaving --data empty at runtime.
  RUN_NAME="$run_name" CONC="$conc" TARGET_URL="$url" MODEL="$MODEL" \
    MAX_REQUESTS="$max_requests" TURNS="$TURNS_PER_CONV" \
    PREFIX_COUNT="$NUM_CONVERSATIONS" PREFIX_TOKENS="$PREFIX_TOKENS" \
    PROMPT_TOKENS="$PROMPT_TOKENS" OUTPUT_TOKENS="$OUTPUT_TOKENS" \
    RANDOM_SEED="$seed" GCS_BUCKET="$GCS_BUCKET" \
    GCS_OUTPUT_DIR="$GCS_OUTPUT_DIR" SUFFIX="$suffix" \
    envsubst '$SUFFIX $RUN_NAME $CONC $TARGET_URL $MODEL $MAX_REQUESTS $TURNS $PREFIX_COUNT $PREFIX_TOKENS $PROMPT_TOKENS $OUTPUT_TOKENS $RANDOM_SEED $GCS_BUCKET $GCS_OUTPUT_DIR' \
    < "$SCRIPT_DIR/guidellm-bench-job.yaml" | kubectl apply -f -

  echo "      waiting for $job_name to complete..."
  kubectl wait --for=condition=complete "job/$job_name" --timeout=2h || {
    echo "      WARN: job did not complete cleanly; collecting logs and moving on"
    kubectl logs -l "job-name=$job_name" --tail=50 || true
  }

  kubectl delete job "$job_name" --wait=false >/dev/null 2>&1 || true
  echo "      uploaded to gs://$GCS_BUCKET/$GCS_OUTPUT_DIR/${run_name}-c${conc}-${suffix}.json"
}

for scenario in "${SCENARIOS[@]}"; do
  RUN_NAME="" BASE_URL="" SKIP_CONFIG="false" MAX_TOKENS_IN_FLIGHT_PENALTY="64000"
  for pair in $scenario; do
    key="${pair%%=*}"; val="${pair#*=}"
    case "$key" in
      run)         RUN_NAME="$val" ;;
      url)         BASE_URL="$val" ;;
      skip_config) SKIP_CONFIG="$val" ;;
      penalty)     MAX_TOKENS_IN_FLIGHT_PENALTY="$val" ;;
    esac
  done

  echo "############################################################"
  echo "SCENARIO: $RUN_NAME  target=$BASE_URL  penalty=$MAX_TOKENS_IN_FLIGHT_PENALTY"
  echo "############################################################"

  if [ "$SKIP_CONFIG" = "false" ]; then
    apply_epp_config "$MAX_TOKENS_IN_FLIGHT_PENALTY"
  else
    echo "[SKIP] EPP reconfigure not applicable (baseline path)."
  fi

  echo "[3/3] Submitting guidellm Jobs across c-levels: $CONCURRENCY_LEVELS  (per-c seed)"
  for conc in $CONCURRENCY_LEVELS; do
    run_one "$RUN_NAME" "$BASE_URL" "$conc" "$(date +%s)"
    sleep 1   # ensure $(date +%s) advances → unique seed per c-level
  done

  echo "Finished $RUN_NAME."
done

echo "All scenarios complete. Results in gs://$GCS_BUCKET/$GCS_OUTPUT_DIR/"
