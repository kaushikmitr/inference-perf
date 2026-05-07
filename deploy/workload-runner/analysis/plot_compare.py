import json, re, subprocess, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Multi-stage runs: each base_prefix is one inference-perf job with multiple
# stages (one per c-level). The script lists GCS for the latest <SUFFIX> per
# base_prefix and loads stage_0..stage_N reports.
RUNS = [
    ("baseline",                        "code-generation-multistage-baseline",          "#1f77b4", "o"),
    ("epp prefix-affinity penalty=64K", "code-generation-multistage-epp-penalty64k",    "#d62728", "s"),
    ("epp prefix-affinity penalty=96K", "code-generation-multistage-epp-penalty96k",    "#2ca02c", "^"),
]

BUCKET = "gs://kaushikmitra-llm-ig-benchmark/workload-catalog-runs"
GMP_PROJECT = "kaushikmitra-gke-dev"
GMP_URL = f"https://monitoring.googleapis.com/v1/projects/{GMP_PROJECT}/location/global/prometheus/api/v1/query"
SCRAPE_INTERVAL = 15
SCRAPE_BUFFER = 2
# c-levels matching the stages list in run_benchmark.sh's multi-stage mode.
# Must match the order stages were defined in (build_config emits stages in
# CONCURRENCY_LEVELS order).
CONC = [10, 20, 30, 40, 50, 60, 70, 80]
FAIL_RATE_THRESHOLD = 0.02
FAIL_COUNT_FLOOR = 3
MIN_DISPATCHED = 30


def discover(base_prefix):
    """Return the full prefix (base + latest suffix) for the most recent run, or None."""
    try:
        listing = subprocess.check_output(
            ["gsutil", "ls", f"{BUCKET}/{base_prefix}-*summary_lifecycle_metrics.json"],
            stderr=subprocess.DEVNULL,
        ).decode()
    except subprocess.CalledProcessError:
        return None
    # Filenames look like: <base_prefix>-<SUFFIX>summary_lifecycle_metrics.json
    pat = re.compile(rf"({re.escape(base_prefix)}-(\d+))summary_lifecycle_metrics\.json$")
    candidates = []
    for line in listing.splitlines():
        m = pat.search(line)
        if m:
            candidates.append((int(m.group(2)), m.group(1)))
    if not candidates:
        return None
    candidates.sort(reverse=True)  # latest suffix first
    return candidates[0][1]


def load(prefix, kind, stage):
    blob = f"{BUCKET}/{prefix}stage_{stage}_{kind}.json"
    return json.loads(subprocess.check_output(["gsutil", "cat", blob], stderr=subprocess.DEVNULL))


_gmp_token_cache = {"token": None}


def gmp_query(query, eval_time):
    """Run a PromQL query against GMP at eval_time (unix seconds). Returns float or 0.0."""
    if _gmp_token_cache["token"] is None:
        _gmp_token_cache["token"] = subprocess.check_output(
            ["gcloud", "auth", "application-default", "print-access-token"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    import urllib.request, urllib.parse
    params = urllib.parse.urlencode({"query": query, "time": str(eval_time)})
    req = urllib.request.Request(
        f"{GMP_URL}?{params}",
        headers={"Authorization": f"Bearer {_gmp_token_cache['token']}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            payload = json.loads(r.read())
        result = payload.get("data", {}).get("result", [])
        if not result:
            return 0.0
        return float(result[0]["value"][1])
    except Exception as e:
        print(f"  gmp_query failed: {e}", file=sys.stderr)
        return 0.0


def _stage_window(prefix, stage_durations, stage_idx):
    run_start = int(prefix.rsplit("-", 1)[-1])
    stage_start = run_start + int(sum(stage_durations[:stage_idx]))
    stage_end = stage_start + int(stage_durations[stage_idx])
    eval_time = stage_end + SCRAPE_INTERVAL + SCRAPE_BUFFER
    window = eval_time - stage_start
    return eval_time, window


def gmp_prefix_hit_pct(prefix, stage_durations, stage_idx):
    """Cluster-wide prefix cache hit % for a specific stage."""
    eval_time, window = _stage_window(prefix, stage_durations, stage_idx)
    q = (
        f"100 * sum(increase(vllm:prefix_cache_hits_total[{window}s])) "
        f"/ (sum(increase(vllm:prefix_cache_queries_total[{window}s])) > 0)"
    )
    return gmp_query(q, eval_time)


def gmp_queue_len(prefix, stage_durations, stage_idx):
    """Cluster-wide peak queue size (sum_max across pods) for a specific stage."""
    eval_time, window = _stage_window(prefix, stage_durations, stage_idx)
    q = f"sum(max_over_time(vllm:num_requests_waiting[{window}s]))"
    return gmp_query(q, eval_time)


def gmp_running_per_endpoint_mean(prefix, stage_durations, stage_idx):
    """Per-endpoint mean concurrent in-flight requests (averaged across pods)."""
    eval_time, window = _stage_window(prefix, stage_durations, stage_idx)
    q = f"avg(avg_over_time(vllm:num_requests_running[{window}s]))"
    return gmp_query(q, eval_time)


def gmp_running_per_endpoint_std(prefix, stage_durations, stage_idx):
    """Stddev across endpoints of per-pod time-averaged in-flight count.
    High value → load imbalanced across pods. 0 → perfectly balanced.
    """
    eval_time, window = _stage_window(prefix, stage_durations, stage_idx)
    q = f"stddev(avg_over_time(vllm:num_requests_running[{window}s]))"
    return gmp_query(q, eval_time)


def gather(label, base_prefix):
    prefix = discover(base_prefix)
    if not prefix:
        print(f"  [{label}] no reports found for base prefix '{base_prefix}'", file=sys.stderr)
        return []

    # Collect stage durations up front so we can query GMP for per-stage prefix-hit.
    stage_lifecycles = {}
    for stage_idx in range(len(CONC)):
        try:
            stage_lifecycles[stage_idx] = load(prefix, "lifecycle_metrics", stage_idx)
        except subprocess.CalledProcessError:
            break
    stage_durations = [
        stage_lifecycles[i]["benchmark_time_seconds"] for i in sorted(stage_lifecycles)
    ]

    rows = []
    for stage_idx, c in enumerate(CONC):
        if stage_idx not in stage_lifecycles:
            print(f"  [{label}] stage {stage_idx} (c={c}) not yet uploaded, skipping", file=sys.stderr)
            continue
        life = stage_lifecycles[stage_idx]
        try:
            prom = load(prefix, "prometheus_metrics", stage_idx)
        except subprocess.CalledProcessError:
            print(f"  [{label}] stage {stage_idx} (c={c}) prom report missing, skipping", file=sys.stderr)
            continue
        try:
            dispatched = life["load_summary"]["count"]
            failures = life["failures"]["count"]
            fail_rate = failures / dispatched if dispatched else 0.0
            if dispatched < MIN_DISPATCHED:
                print(
                    f"  drop [{label}] c={c}: only {dispatched} dispatched (< MIN_DISPATCHED={MIN_DISPATCHED})",
                    file=sys.stderr,
                )
                continue
            if fail_rate > FAIL_RATE_THRESHOLD and failures >= FAIL_COUNT_FLOOR:
                print(
                    f"  drop [{label}] c={c}: "
                    f"{failures}/{dispatched} failed = {fail_rate*100:.1f}% > "
                    f"{FAIL_RATE_THRESHOLD*100:.0f}%",
                    file=sys.stderr,
                )
                continue
            lat = life["successes"]["latency"]
            rows.append({
                "conc": c,
                "ttft_mean": lat["time_to_first_token"]["mean"],
                "ttft_p90":  lat["time_to_first_token"]["p90"],
                "tpot_mean": lat["time_per_output_token"]["mean"] * 1000,
                "tpot_p90":  lat["time_per_output_token"]["p90"] * 1000,
                "out_per_sec": life["successes"]["throughput"]["output_tokens_per_sec"],
                "kv_cache_mean": prom["successes"]["kv_cache_usage_percentage"]["mean"] * 100,
                "kv_cache_p90":  prom["successes"]["kv_cache_usage_percentage"]["p90"] * 100,
                "prefix_hit_pct": gmp_prefix_hit_pct(prefix, stage_durations, stage_idx),
                "queue_len_mean": gmp_queue_len(prefix, stage_durations, stage_idx),
                "running_req_mean": gmp_running_per_endpoint_mean(prefix, stage_durations, stage_idx),
                "running_req_std":  gmp_running_per_endpoint_std(prefix, stage_durations, stage_idx),
                "queue_time_mean": prom["successes"]["request_queue_time"]["mean"],
                "queue_time_p90":  prom["successes"]["request_queue_time"]["p90"],
                "queue_time_p99":  prom["successes"]["request_queue_time"]["p99"],
                "prompt_tokens_p50": life["successes"]["prompt_len"]["median"],
                "prompt_tokens_p90": life["successes"]["prompt_len"]["p90"],
                "gen_tokens_p50":    life["successes"]["output_len"]["median"],
                "gen_tokens_p90":    life["successes"]["output_len"]["p90"],
            })
        except Exception as e:
            print(f"  [{label}] c={c} stage {stage_idx}: {e}", file=sys.stderr)
    return rows


runs = [(label, gather(label, base), color, marker) for label, base, color, marker in RUNS]
runs = [r for r in runs if r[1]]

if not runs:
    sys.exit("no runs with data found")

fig, axes = plt.subplots(4, 3, figsize=(22, 20))
ax = axes.flatten()


def plot_pair(idx, key_lo, key_hi, ylabel, title):
    lo_label = key_lo.rsplit("_", 1)[-1]  # e.g. "mean", "p50"
    hi_label = key_hi.rsplit("_", 1)[-1] if key_hi else None
    for label, rows, color, marker in runs:
        x = [r["conc"] for r in rows]
        ax[idx].plot(x, [r[key_lo] for r in rows], f"{marker}-", label=f"{label} {lo_label}", color=color)
        if key_hi:
            ax[idx].plot(x, [r[key_hi] for r in rows], f"{marker}--", label=f"{label} {hi_label}", color=color, alpha=0.5)
    ax[idx].set_xlabel("concurrency_level (conv slots)")
    ax[idx].set_ylabel(ylabel); ax[idx].set_title(title)
    ax[idx].legend(fontsize=8); ax[idx].grid(alpha=0.3)


def plot_single(idx, key, ylabel, title):
    for label, rows, color, marker in runs:
        x = [r["conc"] for r in rows]
        ax[idx].plot(x, [r[key] for r in rows], f"{marker}-", label=label, color=color)
    ax[idx].set_xlabel("concurrency_level (conv slots)")
    ax[idx].set_ylabel(ylabel); ax[idx].set_title(title)
    ax[idx].legend(fontsize=8); ax[idx].grid(alpha=0.3)


plot_single(0, "ttft_mean", "TTFT mean (s)", "Time To First Token")
plot_single(1, "tpot_mean", "TPOT mean (ms)", "Time Per Output Token")
plot_single(2, "out_per_sec", "output tokens / sec", "Throughput (mean output tokens/s)")
plot_pair(3, "kv_cache_mean", "kv_cache_p90", "KV cache usage (%)", "KV Cache Usage")
plot_single(4, "prefix_hit_pct", "prefix cache hit (%)", "Prefix Cache Hit Rate")
plot_single(5, "queue_len_mean", "queue length (cluster, sum_max)",
            "Queue Size (vllm:num_requests_waiting)")
plot_pair(6, "prompt_tokens_p50", "prompt_tokens_p90", "input tokens / request",
          "Prompt Size (loadgen-tokenized)")
plot_pair(7, "gen_tokens_p50", "gen_tokens_p90", "output tokens / request",
          "Output Size (loadgen-tokenized)")
plot_single(8, "queue_time_mean", "queue wait mean (s)",
            "Request Queue Wait Time")
plot_pair(9, "running_req_mean", "running_req_std", "running requests / endpoint",
          "Per-Endpoint Concurrent In-Flight — avg(across pods) ± stddev(across pods)")

plt.suptitle(" vs ".join(label for label, _, _, _ in RUNS) + "   (multi-stage, num_conv=80)", fontsize=12)
plt.tight_layout()
import os
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "baseline_vs_epp_prefix_runs.png")
plt.savefig(out, dpi=120, bbox_inches="tight")
print(f"saved: {out}")

print()
print(f"{'conc':>5} | {'TTFT mean':>9} {'TTFT p90':>9} | {'TPOT mean':>9} | {'out_t/s':>8} | {'kv_mean':>8} | {'prefix%':>8} | {'qsize':>6} | {'prompt_p50':>10} | {'gen_p50':>8}")
for label, rows, *_ in runs:
    print(f"--- {label} ---")
    for r in rows:
        print(
            f"{r['conc']:5d} | "
            f"{r['ttft_mean']:7.1f}s  {r['ttft_p90']:7.1f}s | "
            f"{r['tpot_mean']:7.1f}ms | "
            f"{r['out_per_sec']:8.1f} | "
            f"{r['kv_cache_mean']:7.1f}% | "
            f"{r['prefix_hit_pct']:7.2f}% | "
            f"{r['queue_len_mean']:6.1f} | "
            f"{r['prompt_tokens_p50']:10.0f} | "
            f"{r['gen_tokens_p50']:8.0f}"
        )
