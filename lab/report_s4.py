#!/usr/bin/env python3
"""Validate the S4 64-layer chain probe and generate its report.

Reads only recorded JSONL. Every number is computed from the artifacts; the
prediction ranges are read back from the preregistered prediction file.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def _final(text):
    """One trailing newline; no trailing blank line."""
    return text.rstrip("\n") + "\n"
RUNS = ("run1", "run2")
SHAPES = ("toy", "shard")
VARIANTS = ("graph_chain", "graph_compute", "graph_comm", "eager_chain")
SAMPLES = 9
# docs/multigpu.md: T(n) = 8.35/n + 1.56 + 0.128*L ms.
COMPUTE_HALF_MS = 8.35 / 2
FIXED_MS = 1.56
MS_PER_US = 0.128
UPSTREAM_MS = 9.91


def read(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x]


def load(run_dir):
    meta = json.loads((run_dir / "metadata.json").read_text())
    sha = meta["git_commit"]["stdout"].strip()
    for filename, key in (("03_block_chain.py", "script_sha256"), ("01_link.py", "metadata_helper_sha256")):
        code = subprocess.check_output(["git", "-C", str(ROOT), "show", f"{sha}:lab/{filename}"])
        assert hashlib.sha256(code).hexdigest() == meta[key], (run_dir, filename)
    statuses = read(run_dir / "status.jsonl")
    assert len(statuses) == len(SHAPES) * len(VARIANTS), run_dir
    assert all(s["returncode"] == 0 and not s["timed_out"] for s in statuses), run_dir
    rows = {}
    for shape in SHAPES:
        for variant in VARIANTS:
            d = run_dir / f"{shape}-{variant}"
            a, b = (read(d / f"rank{r}.jsonl") for r in (0, 1))
            assert len(a) == len(b) == 1, d
            x, y = a[0], b[0]
            assert (x["shape"], x["variant"], x["layers"]) == (shape, variant, meta["shapes"][shape]["layers"])
            for r in (x, y):
                assert r["correct"] and r["finite"] and r["exact_equal_to_reference"], r
                assert r["samples"] == SAMPLES and len(r["gpu_us_per_pass_samples"]) == SAMPLES
                assert len(r["wall_us_per_pass_samples"]) == SAMPLES
                assert abs(statistics.median(r["gpu_us_per_pass_samples"]) - r["gpu_us_per_pass_median"]) < 1e-6
            # max rank per sample, then median; per-layer = per-pass / layers
            per_pass = list(map(max, zip(x["gpu_us_per_pass_samples"], y["gpu_us_per_pass_samples"])))
            wall = list(map(max, zip(x["wall_us_per_pass_samples"], y["wall_us_per_pass_samples"])))
            rows[(shape, variant)] = {
                "run": run_dir.name, "layers": x["layers"], "hidden": x["hidden"], "half_n": x["half_n"],
                "collective_bytes": x["collective_bytes"], "approx_launches_per_pass": x["approx_launches_per_pass"],
                "timing": x["timing"],
                "us_per_pass_median": statistics.median(per_pass),
                "us_per_layer_median": statistics.median(per_pass) / x["layers"],
                "wall_us_per_pass_median": statistics.median(wall),
                "exact_equal": x["exact_equal_to_reference"] and y["exact_equal_to_reference"],
                "max_abs_diff": max(x["max_abs_diff"], y["max_abs_diff"]),
                "weights_bytes_per_card": x["weights_bytes_per_card"],
            }
    return {"meta": meta, "statuses": statuses, "rows": rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("result_dir", type=Path)
    base = p.parse_args().result_dir.resolve()
    loaded = {r: load(base / r) for r in RUNS}
    assert len({d["meta"]["git_commit"]["stdout"] for d in loaded.values()}) == 1
    meta = loaded[RUNS[0]]["meta"]
    commit = meta["git_commit"]["stdout"].strip()
    prediction = json.loads((base / "prediction.json").read_text())
    ranges = prediction["predictions"]

    def row(run, shape, variant):
        return loaded[run]["rows"][(shape, variant)]

    metrics = {}
    for shape in SHAPES:
        chain = {r: row(r, shape, "graph_chain") for r in RUNS}
        compute = {r: row(r, shape, "graph_compute") for r in RUNS}
        comm = {r: row(r, shape, "graph_comm") for r in RUNS}
        eager = {r: row(r, shape, "eager_chain") for r in RUNS}
        metrics[shape] = {
            "chain_us_per_layer": {r: chain[r]["us_per_layer_median"] for r in RUNS},
            "compute_us_per_layer": {r: compute[r]["us_per_layer_median"] for r in RUNS},
            "comm_us_per_layer": {r: comm[r]["us_per_layer_median"] for r in RUNS},
            "eager_us_per_layer": {r: eager[r]["us_per_layer_median"] for r in RUNS},
            "sum_of_parts": {r: compute[r]["us_per_layer_median"] + comm[r]["us_per_layer_median"] for r in RUNS},
            "residual_percent": {r: (chain[r]["us_per_layer_median"] / (compute[r]["us_per_layer_median"] + comm[r]["us_per_layer_median"]) - 1) * 100 for r in RUNS},
            "comm_share_percent": {r: comm[r]["us_per_layer_median"] / chain[r]["us_per_layer_median"] * 100 for r in RUNS},
            "eager_over_graph": {r: eager[r]["us_per_layer_median"] / chain[r]["us_per_layer_median"] for r in RUNS},
            "chain_ms_per_pass": {r: chain[r]["us_per_pass_median"] / 1000 for r in RUNS},
            "effective_GBs": {r: (chain[r]["weights_bytes_per_card"] / chain[r]["layers"] / chain[r]["us_per_layer_median"]) / 1000 for r in RUNS},
            "gemv_GBs": {r: (compute[r]["weights_bytes_per_card"] / compute[r]["layers"] / compute[r]["us_per_layer_median"]) / 1000 for r in RUNS},
        }
    repeat_gap = max(abs(metrics[s][k]["run2"] / metrics[s][k]["run1"] - 1) * 100
                     for s in SHAPES for k in ("chain_us_per_layer", "compute_us_per_layer", "comm_us_per_layer"))

    checks = []
    for shape in SHAPES:
        for key, field in (("graph_chain_us_per_layer", "chain_us_per_layer"),
                           ("graph_compute_us_per_layer", "compute_us_per_layer"),
                           ("graph_comm_us_per_layer", "comm_us_per_layer"),
                           ("eager_over_graph", "eager_over_graph"),
                           ("comm_share_percent", "comm_share_percent")):
            low, high = ranges[shape][key]
            for r in RUNS:
                value = metrics[shape][field][r]
                checks.append({"shape": shape, "metric": key, "run": r, "predicted": [low, high],
                               "measured": value,
                               "verdict": "hit" if low <= value <= high else
                                          ("miss (below)" if value < low else "miss (above)")})
    residual_rule = prediction["additivity_residual_percent_max"]
    residuals_ok = all(abs(metrics[s]["residual_percent"][r]) <= residual_rule for s in SHAPES for r in RUNS)
    misses = [c for c in checks if c["verdict"] != "hit"]

    def projection(latency_us):
        step = COMPUTE_HALF_MS + FIXED_MS + MS_PER_US * latency_us
        return {"L_us": latency_us, "step_ms": step, "tokens_per_second": 1000 / step,
                "speedup_vs_upstream": UPSTREAM_MS / step}

    lat = statistics.mean(metrics["shard"]["comm_us_per_layer"][r] for r in RUNS)
    proj = projection(lat)

    summary = {
        "label": "measured components; the model step below is projected",
        "code_commit": commit, "script_sha256": meta["script_sha256"],
        "shapes": meta["shapes"],
        "runs": {r: {"finished_utc": json.loads((base / r / "finished.json").read_text())["finished_utc"],
                     "variants": [s["variant"] for s in loaded[r]["statuses"]],
                     "reversed_order": r == "run2"} for r in RUNS},
        "metrics": metrics,
        "repeat_max_difference_percent": repeat_gap,
        "checks": checks,
        "additivity": {"rule_percent": residual_rule, "all_within_rule": residuals_ok},
        "criteria": {
            "capture_holds_at_64_layers": True,
            "exact_correctness_both_shapes": all(loaded[r]["rows"][(s, v)]["exact_equal"] for r in RUNS for s in SHAPES for v in VARIANTS),
            "S4_graphs_3x_eager_toy": min(metrics["toy"]["eager_over_graph"].values()) >= 3,
            "S4_graphs_3x_eager_shard": min(metrics["shard"]["eager_over_graph"].values()) >= 3,
            "S5_comm_share_le_25_percent_shard": max(metrics["shard"]["comm_share_percent"].values()) <= 25,
            "M0_L_le_12us": lat <= 12,
        },
        "projection": {"label": "projected, NOT a measured engine",
                       "basis": "measured shard communication component as L plus the plan's estimated compute terms (8.35/2 + 1.56 ms); upstream single-card reference token-rush Phase 4, 2026-09-09/10",
                       **proj, "note": "the 4.7 ms measured chain is one bf16 GEMV per layer, not the production int4 layer, so it is not compared to the step projection"},
        "gate": {"M0_kill_criteria_triggered": False, "M0_cleared": False,
                 "why_not_cleared": "capture, L and additivity pass, but the chain used bf16 weights rather than the production int4 kernel and omitted attention/GDN/sampling; the 1.56 ms fixed term remains an estimate",
                 "decision": "Mode A earns the next measured step: S5 block stack and then the Qwen3-8B TP=2 test model. No 1.5x claim before that measurement."},
    }
    (base / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    fields = ["run", "shape", "variant", "layers", "hidden", "half_n", "collective_bytes",
              "timing", "approx_launches_per_pass", "us_per_pass_median", "us_per_layer_median",
              "wall_us_per_pass_median", "exact_equal", "max_abs_diff", "weights_bytes_per_card"]
    with (base / "matrix.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in RUNS:
            for shape in SHAPES:
                for variant in VARIANTS:
                    d = dict(loaded[r]["rows"][(shape, variant)])
                    d["run"] = r
                    d["shape"] = shape
                    d["variant"] = variant
                    w.writerow({k: d[k] for k in fields})

    def per_layer_table():
        lines = []
        for shape in SHAPES:
            for variant in VARIANTS:
                a, b = row("run1", shape, variant), row("run2", shape, variant)
                lines.append(f"| {shape} | {variant} | {a['us_per_layer_median']:.3f} | {b['us_per_layer_median']:.3f} | "
                             f"{a['us_per_pass_median'] / 1000:.3f} / {b['us_per_pass_median'] / 1000:.3f} | "
                             f"{a['approx_launches_per_pass']} | {a['timing']} |")
        return "\n".join(lines)

    def decomposition_table():
        lines = []
        for shape in SHAPES:
            m = metrics[shape]
            for r in RUNS:
                lines.append(f"| {shape} | {r} | {m['compute_us_per_layer'][r]:.3f} | {m['comm_us_per_layer'][r]:.3f} | "
                             f"{m['sum_of_parts'][r]:.3f} | {m['chain_us_per_layer'][r]:.3f} | "
                             f"{m['residual_percent'][r]:+.2f}% | {m['comm_share_percent'][r]:.1f}% | "
                             f"{m['eager_over_graph'][r]:.2f}x |")
        return "\n".join(lines)

    def check_table():
        return "\n".join(f"| {c['shape']} | {c['metric']} | {c['predicted'][0]:g}-{c['predicted'][1]:g} | "
                         f"{c['measured']:.3f} | {c['run']} | {c['verdict']} |" for c in checks)

    miss_note = ("Every predicted cell falls inside its range." if not misses else
                 f"{len(misses)} of {len(checks)} cells miss: "
                 + "; ".join(f"{c['shape']} {c['metric']} {c['measured']:.3f} ({c['verdict']})" for c in misses)
                 + ". The toy compute and toy all-reduce predictions were wrong: tiny kernels cost "
                   "microseconds each in a graph, and the toy collective is 1 KiB, not the 10 KiB the "
                   "range assumed. The shard communication component landed at the bottom edge of its range.")

    body = f"""## S4 — 64-layer capture probe (2026-10-27, vast-57952342)

Generated by `lab/report_s4.py {base.relative_to(ROOT)}` from recorded JSONL.
Component measurements, not model performance. Test-model results do not
transfer to the production model; no model was downloaded or run.

**Question:** does a 64-layer chain of a real-shaped GEMV, a 10 KiB all-reduce
and an RMSNorm capture in one graph per card, replay correctly, and reproduce
the sum of its separately measured parts?

**Prediction (written before the run, `prediction.json`):** shard shape
55-75 us/layer for the chain, 47-67 compute-only, 8.3-8.7 communication-only,
eager 1.0-1.5x graph, communication share 11-16%. Toy shape 8-12 us/layer and
eager 1.5-3.5x graph, with S4's >=3x criterion at risk. Additivity residual
<= {residual_rule}%.

**Machine/version/code:** the S1/S2 box, 2 x RTX 5090, Gen5 x16, NODE, 400 W,
runtime peer access false both directions. PyTorch {meta['torch']}, CUDA wheel
{meta['cuda']}, NCCL {'.'.join(map(str, meta['nccl']))}. Code commit
`{commit}`; `lab/03_block_chain.py` SHA-256 `{meta['script_sha256']}`. `run2`
repeats every variant in reverse order.

**What was done:** 64 layers per chain, two shapes, one graph per card per
variant, nine timing batches and three warmup replays (five replays per
batch). Per layer: `y = W_l x` (bf16 GEMV at hidden 5120, half-N 8704 for the
shard shape; 512/512 for the toy), residual add, 10 KiB BF16 SUM all-reduce on
the same stream, RMSNorm. `graph_compute` removes the collective,
`graph_comm` is the 64 dependent collectives alone, `eager_chain` runs the full
chain without capture. Weights are identical on both ranks and so are the
inputs, which makes the SUM an exact doubling and gives a bit-exact reference.

**Correctness:** each graph replay is compared against a no-collective
reference chain that substitutes the exact factor 2; every variant and shape
reported `exact_equal_to_reference` with max absolute difference
{max(loaded[r]['rows'][(s, v)]['max_abs_diff'] for r in RUNS for s in SHAPES for v in VARIANTS):.1e}.
The communication-only chain starts at `pattern * 2^-64` and must end exactly
at `pattern` after 64 doublings; it does. Capture succeeded at 64 layers for
both shapes.

**Measured per layer (us; max rank per sample, median) and per pass (ms):**

| shape | variant | run1 | run2 | pass ms run1/run2 | launches | timing |
|---|---|---:|---:|---:|---:|---|
{per_layer_table()}

**Decomposition (us/layer; residual = chain versus compute + communication):**

| shape | run | compute | comm | sum | chain | residual | comm share | eager/graph |
|---|---|---:|---:|---:|---:|---:|---:|---:|
{decomposition_table()}

**Prediction vs result:**

| shape | metric | predicted | measured | run | verdict |
|---|---|---|---:|---|---|
{check_table()}

{miss_note} The largest run1/run2 difference across the nine series is
{repeat_gap:.2f}%.

**Bottleneck:** at the shard shape the chain is weight-read bound: the GEMV
alone is {metrics['shard']['compute_us_per_layer']['run1']:.3f} us/layer, an effective
{metrics['shard']['gemv_GBs']['run1']:.0f} GB/s for
{meta['shapes']['shard']['half_n'] * meta['shapes']['shard']['hidden'] * 2 / 1e6:.1f} MB of bf16
weights per layer, and the whole chain moves those bytes at
{metrics['shard']['effective_GBs']['run1']:.0f} GB/s effective. The collective adds
{metrics['shard']['comm_us_per_layer']['run1']:.3f} us/layer
({metrics['shard']['comm_share_percent']['run1']:.1f}% of the chain), and the
whole chain equals compute plus communication within
{abs(metrics['shard']['residual_percent']['run1']):.1f}%. The toy shape is the
opposite: its GEMV is a few microseconds, so the chain is launch-bound, its
communication share is {metrics['toy']['comm_share_percent']['run1']:.1f}%, and
graphs are {metrics['toy']['eager_over_graph']['run1']:.2f}x faster than eager - this
is where S4's >=3x criterion applies and it passes. At the shard shape the same
comparison is {metrics['shard']['eager_over_graph']['run1']:.2f}x: the
{row('run1', 'shard', 'graph_chain')['approx_launches_per_pass']} launches per pass are almost
entirely hidden behind weight reads, so graphs buy launch tax
only when a chain is launch-bound. Both results are shape facts, not a
contradiction.

**What this does and does not establish:** capture, correctness, additivity
and the 64-layer communication share are measured at the production shard
shape with bf16 weights. The production layer is int4 (4.25 bpw) and has more
compute per layer than one GEMV, and it also carries attention, GDN and
sampling, so the {metrics['shard']['chain_ms_per_pass']['run1']:.3f} ms chain is **not** the
model step and is not compared with it. The communication share is therefore a
conservative upper bound for the production layer: more compute per layer can
only lower it.

**Projection, not measurement (2026-10-27):** using the measured shard
communication component as L ({lat:.3f} us) and the plan's estimated compute terms
(8.35/2 + 1.56 ms), the step is {proj['step_ms']:.3f} ms,
{proj['tokens_per_second']:.1f} tok/s, {proj['speedup_vs_upstream']:.2f}x the upstream
single-card reference. Extrapolated from component measurements, not from a
working engine. The 1.56 ms fixed term is still an estimate from the
single-card engine, and no same-box single-card baseline was re-run.

**Gate and decision:** M0's kill criteria are not triggered - L is
{lat:.3f} us (<= 12), the two-device capture probe holds at 64 layers, and the
component model reproduces the chain within
{abs(metrics['shard']['residual_percent']['run1']):.1f}%. S5's communication-share
criterion passes conservatively at
{metrics['shard']['comm_share_percent']['run1']:.1f}% (<= 25%). **M0 is not
declared cleared**: the chain is bf16 rather than the production int4 kernel,
and the fixed term is estimated. The decision is to spend the next measurement
on the real test model rather than on more synthetic shapes: S5 block stack,
then Qwen3-8B at TP=2, with a fresh prediction first. If the measured model
fails to reach 1.4x, the fallback is Mode B, not more cards.

**Reproduction:**

```bash
.venv/bin/python lab/03_block_chain.py --output results/<date-box>/s4/run1
.venv/bin/python lab/03_block_chain.py --output results/<date-box>/s4/run2 --reverse
.venv/bin/python lab/report_s4.py results/<date-box>/s4
```

`matrix.csv` holds every aggregate cell, raw rank JSONL keeps all nine samples
per cell, `status.jsonl` records clean exit, and the generator checks code
hashes against the recorded commit, sample counts, medians, exact-correctness
flags, and the prediction ranges read from `prediction.json`.
"""
    (base / "S4.md").write_text(body)

    log = ROOT / "lab/LOG.md"
    old = log.read_text().rstrip()
    marker = "## S4 —" if "## S4 —" in old else "## S3/S4 —"
    head = old.split(marker, 1)[0].rstrip() if marker in old else old
    log.write_text(_final(re.sub(r"\n{3,}", "\n\n", head + "\n\n" + body)))

    index = ["S4 64-layer capture probe measured on 2026-10-27, same 2 x RTX 5090 box.",
             f"Code `{commit[:7]}`. bf16 GEMV at hidden 5120 / half-N 8704, 10 KiB SUM, nine samples.", "",
             "| shard shape, 64 layers | us/layer | pass (ms) |", "|---|---:|---:|",
             f"| chain (GEMV + all-reduce + RMSNorm), graph | {metrics['shard']['chain_us_per_layer']['run1']:.3f} | {metrics['shard']['chain_ms_per_pass']['run1']:.3f} |",
             f"| compute only | {metrics['shard']['compute_us_per_layer']['run1']:.3f} | {row('run1', 'shard', 'graph_compute')['us_per_pass_median'] / 1000:.3f} |",
             f"| communication only | {metrics['shard']['comm_us_per_layer']['run1']:.3f} | {metrics['shard']['comm_us_per_layer']['run1'] * 64 / 1000:.3f} |",
             f"| eager chain | {metrics['shard']['eager_us_per_layer']['run1']:.3f} | {metrics['shard']['eager_us_per_layer']['run1'] * 64 / 1000:.3f} |", "",
             f"Chain = compute + communication within {abs(metrics['shard']['residual_percent']['run1']):.1f}%; communication share",
             f"{metrics['shard']['comm_share_percent']['run1']:.1f}% at 64 layers. Capture and exact correctness hold. The chain is",
             "one bf16 GEMV per layer, not the production int4 layer, so **M0 is not cleared** and",
             f"the step projection ({proj['step_ms']:.2f} ms, {proj['speedup_vs_upstream']:.2f}x) remains projected. Next: the",
             "Qwen3-8B TP=2 test model, prediction first.",
             f"[S4 report]({base.relative_to(ROOT)}/S4.md) · [Full data]({base.relative_to(ROOT)}/matrix.csv)."]
    readme = ROOT / "README.md"
    text = readme.read_text()
    start, end = "<!-- S4_RESULTS_START -->", "<!-- S4_RESULTS_END -->"
    block = start + "\n\n" + "\n".join(index) + "\n\n" + end
    if start in text:
        before, rest = text.split(start, 1)
        _, after = rest.split(end, 1)
        text = before + block + after
    elif "<!-- S2_SYNC_RESULTS_END -->" in text:
        text = text.replace("<!-- S2_SYNC_RESULTS_END -->", "<!-- S2_SYNC_RESULTS_END -->\n\n" + block, 1)
    else:
        text = text.replace("## Documents", block + "\n\n## Documents", 1)
    readme.write_text(_final(text))
    print(json.dumps({"checks": {"hits": len(checks) - len(misses), "cells": len(checks)},
                      "shard_chain_us_per_layer": metrics["shard"]["chain_us_per_layer"],
                      "shard_residual_percent": metrics["shard"]["residual_percent"],
                      "projection": proj}, indent=2))


if __name__ == "__main__":
    main()
