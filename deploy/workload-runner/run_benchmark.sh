#!/bin/bash
set -euo pipefail

# Internal = optimized-baseline-vllm-lb (direct vLLM, bypasses EPP)
INTERNAL_IP="http://35.240.207.135:80"
# External = precise-prefix-cache-aware-epp-lb (through gateway/EPP)
EXTERNAL_IP="http://34.124.184.34:80"

# Model + tokenizer (vLLM serves Qwen/Qwen3-32B in this cluster)
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-32B}"

# EPP deployment to roll-restart whenever its plugin config changes
DEPLOYMENT_NAME="predicted-latency-based-scheduling-epp"

# GCS bucket for reports
GCS_BUCKET="${GCS_BUCKET:-kaushikmitra-llm-ig-benchmark}"

# Multi-stage single-job mode. All c-levels run as stages of ONE inference-perf
# job per scenario, sharing num_conversations and the seeded blueprints — so the
# shared system_prompt and per-conv prompts persist across stages, and vLLM's
# prefix cache can warm up cross-stage.
#
# Stages and num_conversations live in the workload yaml under deploy/workload-runner/workloads/.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKLOAD_DIR="$SCRIPT_DIR/workloads"

# Available workloads (each is a yaml file under ./workloads/, sized for 10x Qwen3-32B):
#   interactive-chat   code-generation   deep-research
#   reasoning          batch-summarization-rag   batch-synthetic-data-generation
#
# Per-scenario keys (workload= is required, the rest match the original shape):
#   run=<name>                     — report file prefix
#   workload=<catalog-folder>      — which workload-catalog entry to use
#   url=$INTERNAL_IP|$EXTERNAL_IP  — target base URL
#   skip_config=true|false         — skip EPP reconfigure + rollout
#   w_prefix w_queue w_kv w_pred   — scheduling weights for server-config template
#   tau tau_global                 — affinity-gate thresholds
#   slo_tpot slo_ttft              — request SLO headers (ms; 0 = unset)
#   stream=true|false              — streaming completions
#   shed=nonsheddable|sheddable    — gateway inference objective
#   flow=true|false                — toggles flowControl gate in EPP config
#   max_concurrency=<int>          — concurrency-detector PER ENDPOINT cap, in TOKENS
#                                    (concurrencyMode=tokens). Default 8192 = vLLM
#                                    max_num_batched_tokens. Trigger threshold =
#                                    max_concurrency × (1 + headroom).
#   headroom=<float>               — multiplier above max_concurrency. Trigger
#                                    threshold = max_concurrency × (1 + headroom).
#                                    Default 3 → allows up to 4× max_num_batched_tokens
#                                    of in-flight prefill work. With inflight-load-producer
#                                    (includeOutputTokens=false) and the EPP image's
#                                    cache discount, the counter measures actual
#                                    prefill work (uncached input tokens) directly.
#   penalty=<int>                  — maxTokensInFlightPenalty for prefix-cache-affinity
#                                    filter (in tokens, default 64000). Higher values
#                                    let affinity hold longer before load-balancing
#                                    counter-pressure kicks in.
#   output=<gcs-subpath>           — report directory under gs://$GCS_BUCKET

SCENARIOS=(
  #"run=code-generation-saturation-sweep-epp-1 workload=code-generation url=$EXTERNAL_IP skip_config=false w_prefix=1 w_queue=0 w_kv=1 w_pred=0 tau=0.8 tau_global=0.99 slo_tpot=0 slo_ttft=0 stream=true shed=nonsheddable flow=true output=workload-catalog-runs max_concurrency=13"
  #"run=code-generation-saturation-sweep-epp-tokens-2 workload=code-generation url=$EXTERNAL_IP skip_config=false w_prefix=1 w_queue=0 w_kv=1 w_pred=0 tau=0.8 tau_global=0.99 slo_tpot=0 slo_ttft=0 stream=true shed=nonsheddable flow=true output=workload-catalog-runs"
  #"run=code-generation-saturation-sweep-epp-prefix-filter-3 workload=code-generation url=$EXTERNAL_IP skip_config=false w_prefix=1 w_queue=0 w_kv=1 w_pred=0 tau=0.8 tau_global=0.99 slo_tpot=0 slo_ttft=0 stream=true shed=nonsheddable flow=true output=workload-catalog-runs"
  #"run=code-generation-saturation-sweep-baseline-3 workload=code-generation url=$INTERNAL_IP skip_config=true w_prefix=1 w_queue=0 w_kv=1 w_pred=0 tau=0.8 tau_global=0.99 slo_tpot=0 slo_ttft=0 stream=true shed=nonsheddable flow=true output=workload-catalog-runs"
  #"run=code-generation-saturation-sweep-epp-prefix-filter-penalty64k-1 workload=code-generation url=$EXTERNAL_IP skip_config=false w_prefix=1 w_queue=0 w_kv=1 w_pred=0 tau=0.8 tau_global=0.99 slo_tpot=0 slo_ttft=0 stream=true shed=nonsheddable flow=true output=workload-catalog-runs"
  #"run=code-generation-saturation-sweep-epp-prefix-filter-penalty64k-4 workload=code-generation url=$EXTERNAL_IP skip_config=false w_prefix=1 w_queue=0 w_kv=1 w_pred=0 tau=0.8 tau_global=0.99 slo_tpot=0 slo_ttft=0 stream=true shed=nonsheddable flow=true output=workload-catalog-runs penalty=64000"
  #"run=code-generation-saturation-sweep-epp-prefix-filter-penalty96k-3 workload=code-generation url=$EXTERNAL_IP skip_config=false w_prefix=1 w_queue=0 w_kv=1 w_pred=0 tau=0.8 tau_global=0.99 slo_tpot=0 slo_ttft=0 stream=true shed=nonsheddable flow=true output=workload-catalog-runs penalty=96000"
  #"run=code-generation-saturation-sweep-baseline-6 workload=code-generation url=$INTERNAL_IP skip_config=true w_prefix=1 w_queue=0 w_kv=1 w_pred=0 tau=0.8 tau_global=0.99 slo_tpot=0 slo_ttft=0 stream=true shed=nonsheddable flow=true output=workload-catalog-runs"
  #"run=code-generation-saturation-sweep-epp-prefix-filter-penalty128k workload=code-generation url=$EXTERNAL_IP skip_config=false w_prefix=1 w_queue=0 w_kv=1 w_pred=0 tau=0.8 tau_global=0.99 slo_tpot=0 slo_ttft=0 stream=true shed=nonsheddable flow=true output=workload-catalog-runs penalty=128000"
  #"run=code-generation-saturation-sweep-epp-prefix-filter-penalty160k workload=code-generation url=$EXTERNAL_IP skip_config=false w_prefix=1 w_queue=0 w_kv=1 w_pred=0 tau=0.8 tau_global=0.99 slo_tpot=0 slo_ttft=0 stream=true shed=nonsheddable flow=true output=workload-catalog-runs penalty=160000"
  #"run=code-generation-saturation-sweep-epp-prefix-filter-penalty32k workload=code-generation url=$EXTERNAL_IP skip_config=false w_prefix=1 w_queue=0 w_kv=1 w_pred=0 tau=0.8 tau_global=0.99 slo_tpot=0 slo_ttft=0 stream=true shed=nonsheddable flow=true output=workload-catalog-runs penalty=32000"
  # Multi-stage runs (num_conversations=80, c=10..80, prompts persist across stages)
  #"run=code-generation-multistage-epp-penalty64k workload=code-generation url=$EXTERNAL_IP skip_config=false w_prefix=1 w_queue=0 w_kv=1 w_pred=0 tau=0.8 tau_global=0.99 slo_tpot=0 slo_ttft=0 stream=true shed=nonsheddable flow=true output=workload-catalog-runs penalty=64000"
  "run=code-generation-multistage-epp-penalty96k workload=code-generation url=$EXTERNAL_IP skip_config=false w_prefix=1 w_queue=0 w_kv=1 w_pred=0 tau=0.8 tau_global=0.99 slo_tpot=0 slo_ttft=0 stream=true shed=nonsheddable flow=true output=workload-catalog-runs penalty=96000"
  "run=code-generation-multistage-baseline workload=code-generation url=$INTERNAL_IP skip_config=true w_prefix=1 w_queue=0 w_kv=1 w_pred=0 tau=0.8 tau_global=0.99 slo_tpot=0 slo_ttft=0 stream=true shed=nonsheddable flow=true output=workload-catalog-runs"

)

# --- helper: parse key=value pairs from a scenario string ---
parse_scenario() {
  RUN_NAME="" WORKLOAD=""
  WEIGHT_PREFIX_CACHE="" WEIGHT_QUEUE="" WEIGHT_KV_UTIL=""
  WEIGHT_PREDICTED_LATENCY="" BASE_URL="" SKIP_CONFIG="false"
  AFFINITY_GATE_TAU="" AFFINITY_GATE_TAU_GLOBAL="" OUTPUT_DIR=""
  SLO_TPOT_MS="0" SLO_TTFT_MS="0" STREAMING_MODE="false"
  SHEDDABLE="nonsheddable" FLOW_CONTROL="false"
  MAX_CONCURRENCY="8192"
  HEADROOM="3"
  MAX_TOKENS_IN_FLIGHT_PENALTY="64000"

  for pair in $1; do
    key="${pair%%=*}"
    val="${pair#*=}"
    case "$key" in
      run)              RUN_NAME="$val" ;;
      workload)         WORKLOAD="$val" ;;
      w_prefix)         WEIGHT_PREFIX_CACHE="$val" ;;
      w_queue)          WEIGHT_QUEUE="$val" ;;
      w_kv)             WEIGHT_KV_UTIL="$val" ;;
      w_pred)           WEIGHT_PREDICTED_LATENCY="$val" ;;
      url)              BASE_URL="$val" ;;
      skip_config)      SKIP_CONFIG="$val" ;;
      tau)              AFFINITY_GATE_TAU="$val" ;;
      tau_global)       AFFINITY_GATE_TAU_GLOBAL="$val" ;;
      output)           OUTPUT_DIR="$val" ;;
      slo_tpot)         SLO_TPOT_MS="$val" ;;
      slo_ttft)         SLO_TTFT_MS="$val" ;;
      stream)           STREAMING_MODE="$val" ;;
      shed)             SHEDDABLE="$val" ;;
      flow)             FLOW_CONTROL="$val" ;;
      max_concurrency)  MAX_CONCURRENCY="$val" ;;
      headroom)         HEADROOM="$val" ;;
      penalty)          MAX_TOKENS_IN_FLIGHT_PENALTY="$val" ;;
      *)                echo "WARNING: unknown key '$key'" >&2 ;;
    esac
  done

  export WEIGHT_PREFIX_CACHE WEIGHT_QUEUE WEIGHT_KV_UTIL WEIGHT_PREDICTED_LATENCY
  export BASE_URL AFFINITY_GATE_TAU AFFINITY_GATE_TAU_GLOBAL
  export OUTPUT_DIR SLO_TPOT_MS SLO_TTFT_MS STREAMING_MODE SHEDDABLE
  export MAX_CONCURRENCY HEADROOM MAX_TOKENS_IN_FLIGHT_PENALTY
}

cd "$SCRIPT_DIR"
export PREDICTED_LATENCY_PARAMS=""
export MODEL_NAME GCS_BUCKET

for scenario in "${SCENARIOS[@]}"; do
  parse_scenario "$scenario"

  if [ -z "$WORKLOAD" ]; then
    echo "ERROR: scenario missing workload= key: $scenario" >&2
    exit 1
  fi

  if [ "$FLOW_CONTROL" = "true" ]; then
    export FLOW_CONTROL_GATE="- flowControl"
  else
    export FLOW_CONTROL_GATE=""
  fi

  echo "################################################################"
  echo "STARTING SCENARIO: $RUN_NAME (workload=$WORKLOAD)"
  echo "Target: $BASE_URL | Dir: $OUTPUT_DIR"
  if [ "$SKIP_CONFIG" = "false" ]; then
    echo "Weights -> Prefix: $WEIGHT_PREFIX_CACHE | Queue: $WEIGHT_QUEUE | KV: $WEIGHT_KV_UTIL | Pred: $WEIGHT_PREDICTED_LATENCY"
  fi
  echo "Multi-stage: stages and num_conversations from $WORKLOAD.yaml"
  echo "################################################################"

  # --- PHASE 1: RECONFIGURE EPP (once per scenario) ---
  if [ "$SKIP_CONFIG" = "false" ]; then
    echo "[1/4] Applying new EndpointPickerConfig..."
    if [ "${WEIGHT_PREDICTED_LATENCY:-0}" = "0" ]; then
      echo "WEIGHT_PREDICTED_LATENCY is 0, using config without predicted-latency-scorer"
      envsubst < server-config-no-predicted-latency.yaml | kubectl apply -f -
    else
      envsubst < server-config.yaml | kubectl apply -f -
    fi

    echo "[2/4] Restarting Deployment ($DEPLOYMENT_NAME)..."
    kubectl rollout restart deployment/$DEPLOYMENT_NAME
    kubectl rollout status deployment/$DEPLOYMENT_NAME --timeout=10m

    echo "      Waiting 30s for endpoints to stabilize..."
    sleep 30
  else
    echo "[SKIP] Skipping server configuration and restart."
  fi

  # --- ONE MULTI-STAGE JOB PER SCENARIO ---
  export SUFFIX=$(date +%s)
  export REPORT_PREFIX="${RUN_NAME}-${SUFFIX}"

  # --- PHASE 2: BUILD CONFIG + SUBMIT JOB ---
  echo "[3/4] Building merged inference-perf config and submitting Job..."
  workload_yaml="$WORKLOAD_DIR/$WORKLOAD.yaml"
  if [ ! -f "$workload_yaml" ]; then
    echo "ERROR: workload not found: $workload_yaml" >&2
    exit 1
  fi
  CONFIG_FILE="$(mktemp -t inference-perf-config-${SUFFIX}.XXXXXX.yml)"
  envsubst < "$workload_yaml" > "$CONFIG_FILE"
  kubectl create configmap "inference-perf-config-${SUFFIX}" \
      --from-file=config.yml="$CONFIG_FILE" \
      --dry-run=client -o yaml | kubectl apply -f -
  envsubst < bench-job.yaml | kubectl apply -f -
  rm -f "$CONFIG_FILE"

  echo "      Waiting for completion (inference-perf-$SUFFIX)..."
  kubectl wait --for=condition=complete "job/inference-perf-${SUFFIX}" --timeout=4h

  # --- PHASE 3: CLEANUP ---
  echo "[4/4] Cleaning up..."
  kubectl delete job "inference-perf-${SUFFIX}"
  kubectl delete configmap "inference-perf-config-${SUFFIX}"

  echo "Finished $RUN_NAME."
  sleep 5
done
