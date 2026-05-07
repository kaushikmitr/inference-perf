"""Plot guidellm saturation-sweep results, with vLLM Prometheus overlay.

Reads guidellm JSON reports from GCS (one per scenario × c-level), extracts
TTFT/TPOT/throughput etc. from the lifecycle metrics, and queries Google
Managed Prometheus for the same per-Job time window to fold in cluster
gauges (prefix-hit %, queue size, KV cache, running req).
"""
import json, re, subprocess, sys, urllib.request, urllib.parse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BUCKET = "gs://kaushikmitra-llm-ig-benchmark/guidellm-runs"
GMP_PROJECT = "kaushikmitra-gke-dev"
GMP_URL = f"https://monitoring.googleapis.com/v1/projects/{GMP_PROJECT}/location/global/prometheus/api/v1/query"
SCRAPE_INTERVAL = 15
SCRAPE_BUFFER = 2
FAIL_RATE_THRESHOLD = 0.02
FAIL_COUNT_FLOOR = 3
MIN_DISPATCHED = 30

RUNS = [
    ("baseline",                        "code-generation-guidellm-baseline",          "#1f77b4", "o"),
    ("epp prefix-affinity penalty=64K", "code-generation-guidellm-epp-penalty64k",    "#d62728", "s"),
    ("epp prefix-affinity penalty=96K", "code-generation-guidellm-epp-penalty96k",    "#2ca02c", "^"),
]

CONC = [10, 20, 30, 40, 50, 60, 70, 80]


def discover(base_prefix):
    """Return {c: full_blob_uri} (latest suffix per c) for files under base_prefix."""
    try:
        listing = subprocess.check_output(
            ["gsutil", "ls", f"{BUCKET}/{base_prefix}-c*.json"],
            stderr=subprocess.DEVNULL,
        ).decode()
    except subprocess.CalledProcessError:
        return {}
    out = {}
    pat = re.compile(rf"{re.escape(base_prefix)}-c(\d+)-(\d+)\.json$")
    for line in listing.splitlines():
        m = pat.search(line)
        if not m:
            continue
        c, suffix = int(m.group(1)), int(m.group(2))
        if c not in out or suffix > out[c][0]:
            out[c] = (suffix, line.strip())
    return {c: uri for c, (_, uri) in out.items()}


def load(blob_uri):
    return json.loads(subprocess.check_output(["gsutil", "cat", blob_uri], stderr=subprocess.DEVNULL))


_token = {"v": None}


def gmp_query(query, eval_time):
    if _token["v"] is None:
        _token["v"] = subprocess.check_output(
            ["gcloud", "auth", "application-default", "print-access-token"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    params = urllib.parse.urlencode({"query": query, "time": str(eval_time)})
    req = urllib.request.Request(
        f"{GMP_URL}?{params}",
        headers={"Authorization": f"Bearer {_token['v']}"},
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


def prom_overlay(start_time, end_time):
    """Query GMP for the time window of one benchmark Job."""
    eval_time = end_time + SCRAPE_INTERVAL + SCRAPE_BUFFER
    window = int(eval_time - start_time)
    return {
        "prefix_hit_pct": gmp_query(
            f"100 * sum(increase(vllm:prefix_cache_hits_total[{window}s])) "
            f"/ (sum(increase(vllm:prefix_cache_queries_total[{window}s])) > 0)",
            eval_time,
        ),
        "queue_len_max": gmp_query(
            f"sum(max_over_time(vllm:num_requests_waiting[{window}s]))",
            eval_time,
        ),
        "running_req_mean": gmp_query(
            f"avg(avg_over_time(vllm:num_requests_running[{window}s]))",
            eval_time,
        ),
        "running_req_std": gmp_query(
            f"stddev(avg_over_time(vllm:num_requests_running[{window}s]))",
            eval_time,
        ),
        "kv_cache_mean": gmp_query(
            f"avg(avg_over_time(vllm:kv_cache_usage_perc[{window}s])) * 100",
            eval_time,
        ),
    }


def gather(label, base_prefix):
    discovered = discover(base_prefix)
    if not discovered:
        print(f"  [{label}] no reports found for '{base_prefix}'", file=sys.stderr)
        return []
    rows = []
    for c in sorted(discovered):
        try:
            payload = load(discovered[c])
        except subprocess.CalledProcessError:
            print(f"  [{label}] c={c}: load failed", file=sys.stderr)
            continue
        if not payload.get("benchmarks"):
            continue
        b = payload["benchmarks"][0]
        m = b["metrics"]
        succ_count = m["request_totals"]["successful"]
        err_count = m["request_totals"]["errored"]
        dispatched = succ_count + err_count
        fail_rate = err_count / dispatched if dispatched else 0.0
        if dispatched < MIN_DISPATCHED:
            print(f"  drop [{label}] c={c}: only {dispatched} dispatched", file=sys.stderr)
            continue
        if fail_rate > FAIL_RATE_THRESHOLD and err_count >= FAIL_COUNT_FLOOR:
            print(f"  drop [{label}] c={c}: {err_count}/{dispatched} failed", file=sys.stderr)
            continue

        ttft = m["time_to_first_token_ms"]["successful"]
        tpot = m["time_per_output_token_ms"]["successful"]
        out_tps = m["output_tokens_per_second"]["successful"]
        prompt_ct = m["prompt_token_count"]["successful"]
        gen_ct = m["output_token_count"]["successful"]
        prom = prom_overlay(b["start_time"], b["end_time"])

        rows.append({
            "conc": c,
            "ttft_mean": ttft["mean"] / 1000.0,
            "ttft_p90": ttft["percentiles"]["p90"] / 1000.0,
            "tpot_mean": tpot["mean"],
            "tpot_p90": tpot["percentiles"]["p90"],
            "out_per_sec": out_tps["mean"],
            "prompt_tokens_p50": prompt_ct["percentiles"]["p50"],
            "prompt_tokens_p90": prompt_ct["percentiles"]["p90"],
            "gen_tokens_p50": gen_ct["percentiles"]["p50"],
            "gen_tokens_p90": gen_ct["percentiles"]["p90"],
            **prom,
        })
    return rows


runs = [(label, gather(label, base), color, marker) for label, base, color, marker in RUNS]
runs = [r for r in runs if r[1]]

if not runs:
    sys.exit("no runs with data found")

fig, axes = plt.subplots(4, 3, figsize=(22, 20))
ax = axes.flatten()


def plot_pair(idx, key_lo, key_hi, ylabel, title):
    lo_label = key_lo.rsplit("_", 1)[-1]
    hi_label = key_hi.rsplit("_", 1)[-1] if key_hi else None
    for label, rows, color, marker in runs:
        x = [r["conc"] for r in rows]
        ax[idx].plot(x, [r[key_lo] for r in rows], f"{marker}-",
                     label=f"{label} {lo_label}", color=color)
        if key_hi:
            ax[idx].plot(x, [r[key_hi] for r in rows], f"{marker}--",
                         label=f"{label} {hi_label}", color=color, alpha=0.5)
    ax[idx].set_xlabel("concurrency_level")
    ax[idx].set_ylabel(ylabel); ax[idx].set_title(title)
    ax[idx].legend(fontsize=8); ax[idx].grid(alpha=0.3)


def plot_single(idx, key, ylabel, title):
    for label, rows, color, marker in runs:
        x = [r["conc"] for r in rows]
        ax[idx].plot(x, [r[key] for r in rows], f"{marker}-", label=label, color=color)
    ax[idx].set_xlabel("concurrency_level")
    ax[idx].set_ylabel(ylabel); ax[idx].set_title(title)
    ax[idx].legend(fontsize=8); ax[idx].grid(alpha=0.3)


plot_single(0, "ttft_mean", "TTFT mean (s)", "Time To First Token")
plot_single(1, "tpot_mean", "TPOT mean (ms)", "Time Per Output Token")
plot_single(2, "out_per_sec", "output tokens / sec", "Throughput (output tokens/s)")
plot_single(3, "kv_cache_mean", "KV cache (%)", "KV Cache Usage (cluster avg)")
plot_single(4, "prefix_hit_pct", "prefix cache hit (%)", "Prefix Cache Hit Rate")
plot_single(5, "queue_len_max", "peak queue len (cluster sum_max)",
            "Queue Size (vllm:num_requests_waiting)")
plot_pair(6, "prompt_tokens_p50", "prompt_tokens_p90", "input tokens / request",
          "Prompt Size")
plot_pair(7, "gen_tokens_p50", "gen_tokens_p90", "output tokens / request",
          "Output Size")
plot_pair(9, "running_req_mean", "running_req_std", "running req / endpoint",
          "Per-Endpoint Concurrent In-Flight (avg ± std)")

plt.suptitle(" vs ".join(label for label, _, _, _ in RUNS) + "   (guidellm sweep)", fontsize=12)
plt.tight_layout()
import os
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "guidellm_runs.png")
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
            f"{r['queue_len_max']:6.1f} | "
            f"{r['prompt_tokens_p50']:10.0f} | "
            f"{r['gen_tokens_p50']:8.0f}"
        )
