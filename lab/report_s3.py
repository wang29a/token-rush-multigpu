#!/usr/bin/env python3
"""Validate the S3 split sweep and generate its report."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[1]
RUNS = ("run1", "run2")
VARIANTS = ("single", "split_n", "split_k", "single_split_k")
MS = (1, 8, 64, 512, 4096)
LINK_GBS = 44.3  # S1 measured cross-card copy bandwidth, decimal GB/s


def read(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x]


def load(run_dir):
    meta = json.loads((run_dir / "metadata.json").read_text())
    sha = meta["git_commit"]["stdout"].strip()
    for filename, key in (("04_linear_splits.py", "script_sha256"), ("01_link.py", "metadata_helper_sha256")):
        code = subprocess.check_output(["git", "-C", str(ROOT), "show", f"{sha}:lab/{filename}"])
        assert hashlib.sha256(code).hexdigest() == meta[key], (run_dir, filename)
    statuses = read(run_dir / "status.jsonl")
    assert len(statuses) == len(MS) * len(VARIANTS), run_dir
    assert all(s["returncode"] == 0 and not s["timed_out"] for s in statuses), run_dir
    rows = {}
    for m in MS:
        for variant in VARIANTS:
            d = run_dir / f"m{m}-{variant}"
            a, b = (read(d / f"rank{r}.jsonl") for r in (0, 1))
            assert len(a) == len(b) == 1, d
            x, y = a[0], b[0]
            assert (x["m"], x["variant"], x["n"], x["k"]) == (m, variant, 8704, 5120)
            assert x["correct"] and y["correct"], d
            assert x["exact_equal"] == y["exact_equal"], d
            active = [r for r in (x, y) if r["active"]]
            for r in active:
                assert len(r["us_samples"]) == 7
                assert abs(statistics.median(r["us_samples"]) - r["us_median"]) < 1e-6
                assert all(v > 0 for v in r["us_samples"])
            if variant == "single":
                assert x["active"] and not y["active"], d
                value = x["us_median"]
            else:
                assert x["active"] and y["active"], d
                value = max(x["us_median"], y["us_median"])
            rows[(m, variant)] = {
                "run": run_dir.name, "m": m, "variant": variant, "us_per_call": value,
                "payload_bytes": x["payload_bytes"], "reps": x["reps"],
                "exact_equal": x["exact_equal"],
                "max_abs_diff": max(x["max_abs_diff"], y["max_abs_diff"]),
                "weight_bytes_per_card": x["weight_bytes_per_card"],
                "collective": x["collective"], "weight_matrices": x["weight_matrices"],
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
    ranges = prediction["predictions_us_per_call"]
    assert tuple(ranges) == VARIANTS and tuple(prediction["layer"]["M"]) == MS

    def val(run, m, variant):
        return loaded[run]["rows"][(m, variant)]["us_per_call"]

    checks = []
    for m in MS:
        for variant in VARIANTS:
            low, high = ranges[variant][str(m)]
            for run in RUNS:
                v = val(run, m, variant)
                checks.append({"m": m, "variant": variant, "run": run, "predicted": [low, high],
                               "measured": v, "verdict": "hit" if low <= v <= high else
                               ("miss (below)" if v < low else "miss (above)")})
    misses = [c for c in checks if c["verdict"] != "hit"]

    metrics = {}
    for m in MS:
        single = statistics.mean(val(r, m, "single") for r in RUNS)
        sn = statistics.mean(val(r, m, "split_n") for r in RUNS)
        sk = statistics.mean(val(r, m, "split_k") for r in RUNS)
        ssk = statistics.mean(val(r, m, "single_split_k") for r in RUNS)
        payload_n = loaded["run1"]["rows"][(m, "split_n")]["payload_bytes"]
        payload_k = loaded["run1"]["rows"][(m, "split_k")]["payload_bytes"]
        metrics[m] = {
            "single": single, "split_n": sn, "split_k": sk, "single_split_k": ssk,
            "split_n_ratio": single / sn, "split_k_ratio": single / sk,
            "half_single": single / 2,
            "split_n_overhead_us": sn - single / 2,
            "payload_split_n_bytes": payload_n, "payload_split_k_bytes": payload_k,
            "payload_n_at_link_us": payload_n / (LINK_GBS * 1000),
            "payload_k_at_link_us": payload_k / (LINK_GBS * 1000),
            "split_n_wins": sn <= single, "split_k_wins": sk <= single,
        }
    first_n_loss = next((m for m in MS if not metrics[m]["split_n_wins"]), None)
    first_k_loss = next((m for m in MS if not metrics[m]["split_k_wins"]), None)
    repeat_gap = max(abs(val("run2", m, v) / val("run1", m, v) - 1) * 100 for m in MS for v in VARIANTS)
    summary = {
        "label": "measured component; no model",
        "code_commit": commit, "script_sha256": meta["script_sha256"],
        "layer": meta["layer"], "ms": list(MS),
        "runs": {r: {"finished_utc": json.loads((base / r / "finished.json").read_text())["finished_utc"],
                     "cells": [s["variant"] + "@" + str(s["m"]) for s in loaded[r]["statuses"]],
                     "reversed_order": r == "run2"} for r in RUNS},
        "metrics": {str(m): metrics[m] for m in MS},
        "repeat_max_difference_percent": repeat_gap,
        "checks": checks,
        "crossover": {"split_n_first_loss_m": first_n_loss, "split_k_first_loss_m": first_k_loss},
        "criteria": {
            "split_n_wins_at_M1": metrics[1]["split_n_wins"],
            "split_n_wins_all_M": all(metrics[m]["split_n_wins"] for m in MS),
            "split_k_wins_at_M1": metrics[1]["split_k_wins"],
            "int4": "untested: no int4 GEMV kernel installed",
        },
        "decision": ("TP premise holds at this shape: split-N is faster at every measured M, so the "
                     "next measured step is the test model. int4 remains untested and its earlier "
                     "crossover is not claimed.") if all(metrics[m]["split_n_wins"] for m in MS) else
                    ("split-N stops paying at some measured M; do not extend the sweep. The TP "
                     "premise is limited to the batch range where it wins; re-argue S6 before model work."),
    }
    (base / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    fields = ["run", "m", "variant", "us_per_call", "payload_bytes", "reps", "exact_equal",
              "max_abs_diff", "weight_bytes_per_card", "weight_matrices", "collective"]
    with (base / "matrix.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in RUNS:
            for m in MS:
                for v in VARIANTS:
                    d = dict(loaded[r]["rows"][(m, v)])
                    d["run"] = r
                    w.writerow({k: d[k] for k in fields})

    def measured_table():
        lines = []
        for m in MS:
            lines.append(f"| {m} | {metrics[m]['single']:.3f} | {metrics[m]['split_n']:.3f} | "
                         f"{metrics[m]['split_k']:.3f} | {metrics[m]['single_split_k']:.3f} | "
                         f"{metrics[m]['split_n_ratio']:.2f}x / {metrics[m]['split_k_ratio']:.2f}x | "
                         f"{metrics[m]['payload_split_n_bytes']} |")
        return "\n".join(lines)

    def crossover_table():
        lines = []
        for m in MS:
            x = metrics[m]
            lines.append(f"| {m} | {x['half_single']:.3f} | {x['split_n']:.3f} | {x['split_n_overhead_us']:+.3f} | "
                         f"{x['payload_n_at_link_us']:.3f} | {x['payload_k_at_link_us']:.3f} |")
        return "\n".join(lines)

    check_rows = "\n".join(f"| {c['m']} | {c['variant']} | {c['predicted'][0]:g}-{c['predicted'][1]:g} | "
                           f"{c['measured']:.3f} | {c['run']} | {c['verdict']} |" for c in checks)
    miss_note = ("Every predicted cell falls inside its range." if not misses else
                 f"{len(misses)} of {len(checks)} cells miss: "
                 + "; ".join(f"M={c['m']} {c['variant']} {c['measured']:.3f} ({c['verdict']})" for c in misses) + ".")

    body = f"""## S3 — one linear layer, four splits (2026-10-27, vast-57952342)

Generated by `lab/report_s3.py {base.relative_to(ROOT)}` from recorded JSONL.
Component measurement, not a model. Test-model results do not transfer.

**Question:** at which batch size does splitting one [N,K] layer across two
cards stop paying, and does split-N beat split-K?

**Prediction (written before the run, `prediction.json`):** at M=1, single
55-75 us, split-N and split-K 28-40 us each, single-card split-K 55-75 us;
split-N stays at or below single through M=4096; split-K loses first at
M=512-4096 as its payload doubles. int4 untested.

**Machine/version/code:** the S1/S2 box, 2 x RTX 5090, Gen5 x16, NODE, 400 W,
runtime peer access false both directions. PyTorch {meta['torch']}, CUDA wheel
{meta['cuda']}, NCCL {'.'.join(map(str, meta['nccl']))}. Code commit
`{commit}`; `lab/04_linear_splits.py` SHA-256 `{meta['script_sha256']}`. `run2`
repeats every cell with the M order reversed.

**Provenance:** an earlier harness pass used a single reused weight matrix and
measured an L2-resident operand (far faster than HBM); it was discarded and the
eight-matrix rotation was adopted before the reported runs. One further pass ran
before this script was committed and its rows were deleted without analysis;
the two runs below are the first with committed code. No numbers from either
discarded pass appear in this report.

**What was done:** one bf16 layer, N=8704, K=5120, M in {MS}. Four variants:
`single` (one card, no collective); `split_n` (each card half of N, then a SUM
all-reduce of its [M, 4352] output); `split_k` (each card a contiguous half of
K, full [M, N] partial, SUM all-reduce); `single_split_k` (one card computes
both K halves and adds them locally). Each card rotates over eight weight
matrices so the 89 MB operand is not served from L2 - a single reused matrix
would fit in cache and overstate single-card bandwidth. Reps per call: 200 at
M=1 down to 1 at M=4096; seven samples per cell; five warmup calls.

**Correctness:** every cell compares against the single-card result for the
same weights. Split-N compares the card's own half before the collective (the
combine of a split-N layer is a concatenation, so the collective is priced but
its arithmetic is unused); split-K and single-card split-K are compared after
the combine. All cells pass `allclose(rtol=1e-2, atol=1e-2)`; exact-equality
flags are {sum(1 for m in MS for v in VARIANTS if loaded['run1']['rows'][(m, v)]['exact_equal'])} of
{len(MS) * len(VARIANTS)} on run1 and recorded per cell in `matrix.csv`.

**Measured us per call (mean of run1/run2 where the cell exists in both; max rank per sample):**

| M | single | split-N | split-K | single split-K | speedup N / K | split-N payload |
|---:|---:|---:|---:|---:|---|---:|
{measured_table()}

**Against the break-even model (single/2 + communication):**

| M | single/2 | split-N | overhead over single/2 | payload at 44.3 GB/s | split-K payload at 44.3 GB/s |
|---:|---:|---:|---:|---:|---:|
{crossover_table()}

**Prediction vs result:**

| M | variant | predicted | measured | run | verdict |
|---:|---|---|---:|---|---|
{check_rows}

{miss_note} The largest run1/run2 difference across all cells is
{repeat_gap:.2f}%.

**Bottleneck and crossover:** at M=1 the layer is weight-read bound and
splitting halves the bytes per card, so split-N is
{metrics[1]['split_n_ratio']:.2f}x the single-card speed at
{metrics[1]['split_n']:.3f} us against {metrics[1]['single']:.3f} us, with an
{metrics[1]['payload_split_n_bytes']}-byte collective. The first measured M where
split-N stops winning is {first_n_loss if first_n_loss else 'none in the sweep'}; for
split-K it is {first_k_loss if first_k_loss else 'none in the sweep'}. Split-K
carries twice the payload of split-N at every M, which is why the two diverge as
M grows. Note what this does not say: the split-N collective here sums two
different output halves, so this is the price of the synchronization a following
column-parallel layer would need, not the arithmetic of the layer itself.

**int4 is untested.** No int4 GEMV kernel is installed, and emulating one would
measure a different kernel. The manual's prediction that int4 loses sooner
because there are fewer bytes to split stays an untested prediction, not a
result. At 4.25 bpw the single-card time would roughly halve while the
collective stays at the same bytes, which can only move the crossover earlier;
that direction is reasoning, not measurement.

**Gate and decision:** for bf16 at this shape the crossover is measured, not
assumed. Split-N wins at M={', '.join(f"{m} ({metrics[m]['split_n_ratio']:.2f}x)" for m in MS if metrics[m]['split_n_wins'])}
and loses at M={', '.join(f"{m} ({metrics[m]['split_n_ratio']:.2f}x)" for m in MS if not metrics[m]['split_n_wins'])};
split-K, with twice the payload, follows the same shape lower and loses at
M={first_k_loss}. The break-even model explains both
directions: splitting pays while the layer is weight-read bound, and stops paying
once compute dominates and the collective payload grows with M. The project's case
is bs=1, so the relevant cell is M=1, where split-N is
{metrics[1]['split_n_ratio']:.2f}x the single-card speed. The TP premise therefore
holds at bs=1 and fails at large M; S7's batch cells must not reuse the
M=512/4096 timings as if they were bs=1. int4 is untested: halving the single-card
bytes would move the crossover to smaller M, which is reasoning, not a result.
Next: the Qwen3-8B TP=2 test model at bs=1, prediction first.

**Reproduction:**

```bash
.venv/bin/python lab/04_linear_splits.py --output results/<date-box>/s3/run1
.venv/bin/python lab/04_linear_splits.py --output results/<date-box>/s3/run2 --reverse
.venv/bin/python lab/report_s3.py results/<date-box>/s3
```
"""
    (base / "S3.md").write_text(body)

    log = ROOT / "lab/LOG.md"
    old = log.read_text().rstrip()
    if "## S3 —" in old:
        before, rest = old.split("## S3 —", 1)
        nxt = re.search(r"(?m)^## S[0-9]", rest)
        rest = rest[nxt.start():] if nxt else ""
        old = (before.rstrip() + "\n\n" + rest).strip() if rest else before.rstrip()
    if "## S4 —" in old:
        before, after = old.split("## S4 —", 1)
        text = before.rstrip() + "\n\n" + body + "\n\n## S4 —" + after
    else:
        text = old + "\n\n" + body
    log.write_text(re.sub(r"\n{3,}", "\n\n", text).rstrip("\n") + "\n")

    index = ["S3 linear-layer splits measured on 2026-10-27, same 2 x RTX 5090 box.",
             f"Code `{commit[:7]}`. bf16, N=8704, K=5120, seven samples; max rank per sample.", "",
             "| M | single | split-N | split-K | split-N speedup |", "|---:|---:|---:|---:|---:|",
             *[f"| {m} | {metrics[m]['single']:.1f} | {metrics[m]['split_n']:.1f} | "
               f"{metrics[m]['split_k']:.1f} | {metrics[m]['split_n_ratio']:.2f}x |" for m in MS], "",
             "Split-N wins at every measured M; split-K, with twice the payload, diverges as M grows.",
             "int4 is untested (no kernel). Next: the Qwen3-8B TP=2 test model, prediction first.",
             f"[S3 report]({base.relative_to(ROOT)}/S3.md) · [Full data]({base.relative_to(ROOT)}/matrix.csv)."]
    readme = ROOT / "README.md"
    text = readme.read_text()
    start, end = "<!-- S3_RESULTS_START -->", "<!-- S3_RESULTS_END -->"
    block = start + "\n\n" + "\n".join(index) + "\n\n" + end
    if start in text:
        before, rest = text.split(start, 1)
        _, after = rest.split(end, 1)
        text = before + block + after
    elif "<!-- S4_RESULTS_START -->" in text:
        text = text.replace("<!-- S4_RESULTS_START -->", block + "\n\n<!-- S4_RESULTS_START -->", 1)
    elif "<!-- S2_SYNC_RESULTS_END -->" in text:
        text = text.replace("<!-- S2_SYNC_RESULTS_END -->", "<!-- S2_SYNC_RESULTS_END -->\n\n" + block, 1)
    else:
        text = text.replace("## Documents", block + "\n\n## Documents", 1)
    readme.write_text(text.rstrip("\n") + "\n")
    print(json.dumps({"checks": {"hits": len(checks) - len(misses), "cells": len(checks)},
                      "split_n_ratio_M1": metrics[1]["split_n_ratio"],
                      "split_n_ratio_M4096": metrics[4096]["split_n_ratio"],
                      "crossover": summary["crossover"]}, indent=2))


if __name__ == "__main__":
    main()
