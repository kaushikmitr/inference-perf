#!/usr/bin/env python3
"""
Flow Control Tuning Wizard

Calculates optimal concurrency and lookahead buffer limits for Gateway deployments.
Evaluates the three fundamental constraints of LLM serving:
  1. Compute SLA Constraint:   Empirically, via Little's Law (needs throughput + latency).
  2. Memory Capacity Constraint: Analytically, via CLT on KV cache footprint.
  3. Prefill Compute Constraint: TTFT-budget against the engine's chunked-prefill budget.

The system's true concurrency limit is the active bottleneck (min of the three).

Engine `step_time` is hard to know without measurement. The script supports:
  (a) explicit value via --step-time-sec, OR
  (b) estimate from the GPU FLOPs roofline (--num-params + --tp-size + GPU info).
"""

import argparse
import math
import sys
from typing import Tuple, Optional

# ==========================================
# UI Helpers
# ==========================================

def get_float_input(prompt: str, default: Optional[float] = None) -> Optional[float]:
    while True:
        val = input(prompt)
        if not val:
            return default
        try:
            return float(val)
        except ValueError:
            print("  [!] Please enter a valid number.")

def get_int_input(prompt: str, default: Optional[int] = None) -> Optional[int]:
    while True:
        val = input(prompt)
        if not val:
            return default
        try:
            return int(val)
        except ValueError:
            print("  [!] Please enter a valid integer.")

def print_header(title: str) -> None:
    print(f"\n{'-'*60}\n{title}\n{'-'*60}")

# ==========================================
# Core Mathematical Logic
# ==========================================

def calculate_compute_constraint(throughput: float, latency_sec: float) -> int:
    """Concurrency from Little's Law (L = lambda * W)."""
    return math.floor(throughput * latency_sec)

def calculate_memory_constraint(
    gpu_blocks: int, block_size: int, paged_attention_efficiency: float,
    shared_prefix: int, enable_prefix_caching: bool,
    isl_mean: float, isl_std: float, osl_mean: float, osl_std: float,
    correlation_coefficient: float, z_score: float
) -> Tuple[int, float, float]:
    """Max concurrency before KV exhaustion, via CLT on per-request VRAM footprint.
    Returns: (memory_limit, marginal_isl, coefficient_of_variation)"""
    effective_tokens = gpu_blocks * block_size * paged_attention_efficiency

    if enable_prefix_caching:
        available_tokens = max(0, effective_tokens - shared_prefix)
        marginal_isl = max(0, isl_mean - shared_prefix)
    else:
        available_tokens = effective_tokens
        marginal_isl = isl_mean

    isl_std_eff = isl_std if marginal_isl > 0 else 0.0

    # Mean cost over an autoregressive request's lifetime
    mu_footprint = marginal_isl + (osl_mean / 2.0)
    var_output = (osl_std**2 / 3.0) + (osl_mean**2 / 12.0)
    var_footprint = (isl_std_eff**2) + var_output + (correlation_coefficient * isl_std_eff * osl_std)
    sigma_footprint = math.sqrt(var_footprint)

    cv = sigma_footprint / mu_footprint if mu_footprint > 0 else 0.0

    a = mu_footprint
    b = z_score * sigma_footprint
    c = -available_tokens

    discriminant = (b**2) - (4 * a * c)
    if discriminant < 0 or a <= 0:
        raise ValueError("Workload variance is too high for available VRAM.")

    x = (-b + math.sqrt(discriminant)) / (2 * a)
    return int(x**2), marginal_isl, cv

def estimate_step_time(
    num_params_b: float,
    max_num_batched_tokens: int,
    tp_size: int = 1,
    peak_tflops: float = 989.0,
    utilization: float = 0.40,
) -> float:
    """Estimate engine scheduler step time from a FLOPs roofline.

    For chunked prefill (the binding case for TTFT), each step is dominated by
    a forward pass over `mnbt` tokens through a model of `num_params_b` billion
    parameters:

        FLOPs/step = 2 × num_params × mnbt
        achievable = TP × peak_TFLOPS × utilization
        t_step    = FLOPs/step / achievable

    Defaults are H100 SXM5 (989 TFLOPS FP16) at 40% achieved utilization, which
    is a reasonable middle-of-the-road number for vLLM. Adjust for your GPU.

    Calibration: 32B / TP=2 / H100 / 40% util → ~660ms (matches observed
    Qwen3-32B/TP=2 step time of 500–600ms within ~20%).
    """
    flops_per_step = 2.0 * (num_params_b * 1e9) * max_num_batched_tokens
    achievable_flops_per_sec = tp_size * (peak_tflops * 1e12) * utilization
    return flops_per_step / achievable_flops_per_sec

def calculate_prefill_constraint(
    max_num_batched_tokens: int,
    marginal_isl: float,
    osl_mean: float,
    max_num_partial_prefills: int = 1,
    ttft_budget_sec: float = 0.5,
    step_time_sec: float = 0.05,
) -> int:
    """TTFT-budgeted prefill-compute concurrency limit per replica.

    Derivation (chunked prefill, P = max_num_partial_prefills):

      With at most P requests in their prefill phase at once, the Nth queued
      request waits for (N-1)/P prior prefills to drain through the P slots.
      Each prefill drains at rate mnbt/isl per step, taking isl/mnbt steps.

          TTFT(N) ≈ (N / P) × (isl / mnbt) × t_step  (seconds)

      Solving for N at a target TTFT budget:

          N_prefill = TTFT_budget × P × mnbt / (isl × t_step)

      Multiply by (1 + osl/isl) to account for decode-stretched residence:
      requests in decode hold a slot but don't compete for prefill budget.

          N_max = N_prefill × (1 + osl/isl)

    Sanity check (Qwen3-32B/TP=2/H100, P=1, mnbt=8192, isl=9000, osl=500,
    t_step=550ms): TTFT=10s → N≈18, TTFT=60s → N≈110 (clipped by max_num_seqs).
    """
    isl_eff = max(1.0, marginal_isl)
    n_prefill = ttft_budget_sec * max_num_partial_prefills * max_num_batched_tokens / (isl_eff * step_time_sec)
    osl_factor = 1.0 + (osl_mean / isl_eff)
    return max(1, math.floor(n_prefill * osl_factor))

def calculate_lookahead_buffer(active_batch: int, max_num_batched_tokens: int, isl_mean: Optional[float]) -> int:
    """Sizes the engine's local queue for continuous batching, capped at 15% of active."""
    max_allowed_buffer = math.ceil(active_batch * 0.15)

    if isl_mean is None:
        return max(1, max_allowed_buffer)

    effective_isl = max(1.0, isl_mean)
    buffer_size = math.ceil(max_num_batched_tokens / effective_isl)
    return max(1, min(buffer_size, max_allowed_buffer))

# ==========================================
# Main
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Flow Control Tuning Wizard")

    group_compute = parser.add_argument_group("Compute SLA Constraints")
    group_compute.add_argument("--throughput", type=float, help="Mean throughput (RPS)")
    group_compute.add_argument("--latency-sec", type=float, dest="latency_sec", help="Mean end-to-end latency (seconds)")

    group_memory = parser.add_argument_group("Memory Capacity Constraints")
    group_memory.add_argument("--gpu-blocks", type=int, dest="gpu_blocks", help="Available KV blocks from engine logs")
    group_memory.add_argument("--block-size", type=int, dest="block_size", default=16, help="Tokens per KV block")
    group_memory.add_argument("--isl-mean", type=float, dest="isl_mean", help="Mean Input Sequence Length")
    group_memory.add_argument("--isl-std", type=float, dest="isl_std", help="StdDev of ISL")
    group_memory.add_argument("--osl-mean", type=float, dest="osl_mean", help="Mean Output Sequence Length")
    group_memory.add_argument("--osl-std", type=float, dest="osl_std", help="StdDev of OSL")

    group_engine = parser.add_argument_group("Engine Architecture")
    group_engine.add_argument("--shared-prefix", type=int, dest="shared_prefix", default=0, help="Static system prompt length")
    group_engine.add_argument("--enable-prefix-caching", action="store_true", help="Set if caching is ON")
    group_engine.add_argument("--max-num-batched-tokens", type=int, dest="max_num_batched_tokens", default=2048, help="Engine prefill budget")
    group_engine.add_argument("--max-num-partial-prefills", type=int, dest="max_num_partial_prefills", default=1,
                              help="vLLM scheduler cap on concurrent prefill phases (default 1, vLLM default)")

    group_step = parser.add_argument_group("Engine Step Time (provide one)")
    group_step.add_argument("--step-time-sec", type=float, dest="step_time_sec",
                            help="Measured engine scheduler step time in seconds (preferred — overrides estimate)")
    group_step.add_argument("--num-params-b", type=float, dest="num_params_b",
                            help="Model parameter count in billions (e.g. 32 for Qwen3-32B). Used to estimate step time.")
    group_step.add_argument("--tp-size", type=int, dest="tp_size", default=1, help="Tensor parallel size (default 1)")
    group_step.add_argument("--peak-tflops", type=float, dest="peak_tflops", default=989.0,
                            help="GPU peak FP16 TFLOPS (default 989 = H100 SXM5; A100=312, H200=989, B200=2250)")
    group_step.add_argument("--gpu-utilization", type=float, dest="gpu_utilization", default=0.40,
                            help="Achieved FLOPs utilization fraction (default 0.4; vLLM typically 0.3-0.5)")

    group_adv = parser.add_argument_group("Advanced Statistical Parameters")
    group_adv.add_argument("--z-score", type=float, dest="z_score", default=2.0, help="Statistical safety margin")
    group_adv.add_argument("--paged-attention-efficiency", type=float, dest="paged_attention_efficiency", default=0.90, help="VRAM fragmentation buffer")
    group_adv.add_argument("--correlation-coefficient", type=float, dest="correlation_coefficient", default=0.0, help="ISL/OSL correlation")
    group_adv.add_argument("--ttft-budget-sec", type=float, dest="ttft_budget_sec", default=0.5,
                           help="TTFT tolerance for prefill concurrency calc (default 0.5s; 0.2=strict, 1.0=tolerant)")

    args = parser.parse_args()
    interactive = len(sys.argv) == 1

    if interactive:
        print("=== LLM Capacity Tuning Wizard ===")
        print("Press Enter to accept defaults where available.\n")

        mode = input("Select calculation mode (compute/memory/both) [both]: ") or "both"

        if mode in ["compute", "both"]:
            print_header("Step 1: Compute SLA Constraint")
            args.throughput = get_float_input("  > Mean throughput (RPS): ")
            args.latency_sec = get_float_input("  > Mean end-to-end latency (SECONDS): ")

        if mode in ["memory", "both"]:
            print_header("Step 2: Memory Capacity Constraint")
            args.gpu_blocks = get_int_input("  > Total available KV cache blocks (# GPU blocks): ")
            args.block_size = get_int_input("  > Tokens per KV block [16]: ", default=16)
            args.isl_mean = get_float_input("  > Mean Input Sequence Length (tokens): ")

            val_isl = get_float_input("  > StdDev of Input (leave blank to assume exponential): ")
            args.isl_std = val_isl if val_isl is not None else args.isl_mean

            args.osl_mean = get_float_input("  > Mean Output Sequence Length (tokens): ")
            val_osl = get_float_input("  > StdDev of Output (leave blank to assume exponential): ")
            args.osl_std = val_osl if val_osl is not None else args.osl_mean

            print("\n  [Engine Context & Caching]")
            args.shared_prefix = get_int_input("  > Length of shared system prompt [0]: ", default=0)

            if args.shared_prefix > 0:
                print("  [!] WARNING: Enabling caching here without enabling it on the engine causes OOMs.")
                ans = input("  > Is prefix caching explicitly enabled on your engine? (y/n) [n]: ") or "n"
                args.enable_prefix_caching = ans.lower().startswith('y')
            else:
                args.enable_prefix_caching = False

            args.max_num_batched_tokens = get_int_input("  > Engine chunked prefill budget (--max-num-batched-tokens) [2048]: ", default=2048)
            args.max_num_partial_prefills = get_int_input("  > vLLM --max-num-partial-prefills [1]: ", default=1)

            print("\n  [Engine Step Time — pick one]")
            print("  Option A: provide measured step time directly")
            print("  Option B: leave blank and provide model+GPU info to estimate from FLOPs roofline")
            measured = get_float_input("  > Measured step time in seconds (blank to estimate): ")
            if measured is not None:
                args.step_time_sec = measured
            else:
                args.num_params_b = get_float_input("  > Model parameters in billions (e.g. 32 for Qwen3-32B): ")
                args.tp_size = get_int_input("  > Tensor parallel size [1]: ", default=1)
                args.peak_tflops = get_float_input("  > GPU peak FP16 TFLOPS [989=H100, 312=A100, 2250=B200]: ", default=989.0)
                args.gpu_utilization = get_float_input("  > Expected GPU utilization fraction [0.4]: ", default=0.4)

        print_header("Step 3: Statistical Safety")
        args.z_score = get_float_input("  > Z-score safety margin (e.g., 2.0 for 95% confidence) [2.0]: ", default=2.0)

        print_header("Step 4: TTFT Tolerance")
        args.ttft_budget_sec = get_float_input("  > TTFT budget for prefill saturation (seconds) [0.5]: ", default=0.5)

    # Fallbacks for CLI args
    if args.isl_std is None and args.isl_mean is not None: args.isl_std = args.isl_mean
    if args.osl_std is None and args.osl_mean is not None: args.osl_std = args.osl_mean

    # Resolve step time: explicit > FLOPs estimate > hard fallback
    step_time_source = "explicit"
    if args.step_time_sec is None:
        if args.num_params_b is not None and args.max_num_batched_tokens is not None:
            args.step_time_sec = estimate_step_time(
                args.num_params_b, args.max_num_batched_tokens,
                args.tp_size, args.peak_tflops, args.gpu_utilization,
            )
            step_time_source = f"FLOPs estimate ({args.num_params_b}B params, TP={args.tp_size}, {args.peak_tflops} TFLOPS, util={args.gpu_utilization})"
        else:
            args.step_time_sec = 0.05
            step_time_source = "hard fallback (50ms — likely WRONG for >13B models)"

    # Validation
    run_compute = args.throughput is not None and args.latency_sec is not None
    run_memory = all(v is not None for v in [args.gpu_blocks, args.isl_mean, args.isl_std, args.osl_mean, args.osl_std])

    if not run_compute and not run_memory:
        sys.exit("\nError: You must provide arguments for Compute, Memory, or both.")

    compute_limit, memory_limit, prefill_limit = None, None, None
    marginal_isl, cv = None, 0.0

    if run_compute:
        if args.latency_sec > 100:
            print("  [!] WARNING: Latency is high. Ensure it's in seconds, not ms.")
        compute_limit = calculate_compute_constraint(args.throughput, args.latency_sec)

    if run_memory:
        if args.shared_prefix > args.isl_mean:
            print("  [!] WARNING: Shared prefix is larger than mean ISL. Capping to mean ISL.")
            args.shared_prefix = int(args.isl_mean)
        try:
            memory_limit, marginal_isl, cv = calculate_memory_constraint(
                args.gpu_blocks, args.block_size, args.paged_attention_efficiency,
                args.shared_prefix, args.enable_prefix_caching,
                args.isl_mean, args.isl_std, args.osl_mean, args.osl_std,
                args.correlation_coefficient, args.z_score
            )
        except ValueError as e:
            sys.exit(f"\nError: {e}")

        prefill_limit = calculate_prefill_constraint(
            args.max_num_batched_tokens, marginal_isl, args.osl_mean,
            args.max_num_partial_prefills, args.ttft_budget_sec, args.step_time_sec,
        )

    # Bottleneck Resolution
    candidates = []
    if run_compute:                  candidates.append(("Compute (Latency SLAs)", compute_limit))
    if run_memory:                   candidates.append(("Memory (KV Cache)", memory_limit))
    if prefill_limit is not None:    candidates.append(("Prefill (max_num_batched_tokens)", prefill_limit))

    bottleneck, safe_active_batch = min(candidates, key=lambda x: x[1])
    if not run_compute:
        bottleneck += " (no SLA — heuristics only)"

    if safe_active_batch < 1:
        sys.exit("\nError: Safe active batch is < 1. Hardware cannot support this workload.")

    buffer_size = calculate_lookahead_buffer(safe_active_batch, args.max_num_batched_tokens, args.isl_mean)
    gateway_concurrency = safe_active_batch + buffer_size

    # Results
    print_header("TUNING WIZARD RESULTS")
    if run_memory:
        print(f"Step time used:           {args.step_time_sec*1000:.0f}ms ({step_time_source})")
    if run_compute:                  print(f"Calculated Compute Limit: {compute_limit} requests")
    if run_memory:                   print(f"Calculated Memory Limit:  {memory_limit} requests")
    if prefill_limit is not None:    print(f"Calculated Prefill Limit: {prefill_limit} requests  (TTFT={args.ttft_budget_sec}s, P={args.max_num_partial_prefills})")
    print(f"Active Bottleneck:        {bottleneck}")
    print("-" * 60)
    print(f"Engine Target (N_active): {safe_active_batch}")
    print(f"Lookahead Buffer (B):     {buffer_size}")
    print(f"Gateway Max Concurrency:  {gateway_concurrency}  (per endpoint, N_active + B)")

    # Heavy-Tail Warnings
    if (run_compute and compute_limit < 30) or (run_memory and cv > 0.5):
        print("\n[!] WARNING: HIGH VARIANCE / SMALL BATCH DETECTED")
        if run_compute and compute_limit < 30:
            print(f"  - Batch size ({compute_limit}) is small (N < 30).")
        if run_memory and cv > 0.5:
            print(f"  - Coefficient of Variation (CV = {cv:.2f}) > 0.5 indicates a heavy-tailed distribution.")
        print("  -> The Gaussian assumption of the CLT may underestimate peak VRAM usage.")
        print("  -> Consider increasing --z-score to 3.0+ or using P99 sequence lengths.")

    if prefill_limit is not None and bottleneck.startswith("Prefill"):
        print("\n[!] PREFILL-BOUND WORKLOAD")
        print(f"  - Binding constraint is the engine's prefill budget (max_num_batched_tokens={args.max_num_batched_tokens},")
        print(f"    max_num_partial_prefills={args.max_num_partial_prefills}, t_step={args.step_time_sec*1000:.0f}ms).")
        print(f"  - Raise --ttft-budget-sec if higher TTFT is acceptable for higher batch utilization.")
        print("  -> Other levers: increase --max-num-batched-tokens or --max-num-partial-prefills,")
        print("     add replicas, shorten avg ISL, or move to P/D disaggregation.")

    if step_time_source.startswith("hard fallback"):
        print("\n[!] STEP TIME UNKNOWN — prefill estimate may be off by 10× or more.")
        print("    Provide --step-time-sec OR --num-params-b for a defensible number.")

    # Output Configurations
    recommended_headroom = 0.0
    if bottleneck.startswith("Compute") and args.enable_prefix_caching:
        recommended_headroom = 0.1

    print_header("CONFIGURATION SNIPPETS")
    print("1. Gateway Configuration (EndpointPickerConfig YAML):")
    print(f"""apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
featureGates:
- flowControl
plugins:
- name: my-concurrency-detector
  type: concurrency-detector
  parameters:
    maxConcurrency: {gateway_concurrency}     # per endpoint
    headroom: {recommended_headroom}                  # 10-20% only if compute-bound + cache enabled
saturationDetector:
  pluginRef: my-concurrency-detector
flowControl:
  maxRequests: 200
  maxBytes: "10Gi"
""")

    print("2. Model Server Configuration (Engine Target):")
    print(f"   Target Active Concurrency per replica: {safe_active_batch}")
    print(f"   Example for vLLM: vllm serve ... --max-num-seqs {safe_active_batch}")
    print(f"   Example for TGI:  text-generation-launcher ... --max-concurrent-requests {safe_active_batch}\n")

    # Reproducer
    if interactive:
        print_header("AUTOMATION REPRODUCER")
        print("To reproduce this calculation via CLI:")
        cmd = ["python3 tuning_wizard.py"]

        def add(flag: str, val) -> None:
            if val is not None:
                cmd.append(f"{flag} {val}")

        add("--throughput", args.throughput)
        add("--latency-sec", args.latency_sec)
        add("--gpu-blocks", args.gpu_blocks)
        if args.block_size != 16: add("--block-size", args.block_size)
        add("--isl-mean", args.isl_mean)
        add("--isl-std", args.isl_std)
        add("--osl-mean", args.osl_mean)
        add("--osl-std", args.osl_std)
        if args.shared_prefix: add("--shared-prefix", args.shared_prefix)
        if args.enable_prefix_caching: cmd.append("--enable-prefix-caching")
        if args.max_num_batched_tokens != 2048: add("--max-num-batched-tokens", args.max_num_batched_tokens)
        if args.max_num_partial_prefills != 1: add("--max-num-partial-prefills", args.max_num_partial_prefills)
        if step_time_source == "explicit": add("--step-time-sec", args.step_time_sec)
        elif args.num_params_b is not None:
            add("--num-params-b", args.num_params_b)
            if args.tp_size != 1: add("--tp-size", args.tp_size)
            if args.peak_tflops != 989.0: add("--peak-tflops", args.peak_tflops)
            if args.gpu_utilization != 0.40: add("--gpu-utilization", args.gpu_utilization)
        if args.z_score != 2.0: add("--z-score", args.z_score)
        if args.paged_attention_efficiency != 0.90: add("--paged-attention-efficiency", args.paged_attention_efficiency)
        if args.correlation_coefficient != 0.0: add("--correlation-coefficient", args.correlation_coefficient)
        if args.ttft_budget_sec != 0.5: add("--ttft-budget-sec", args.ttft_budget_sec)

        print(" \\\n  ".join(cmd) + "\n")

if __name__ == "__main__":
    main()
