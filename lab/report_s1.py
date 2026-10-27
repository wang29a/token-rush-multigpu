#!/usr/bin/env python3
"""Generate S1 tables and conclusions from immutable JSONL measurements."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import re

ROOT = Path(__file__).resolve().parents[1]


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def load(run):
    statuses = read_rows(run / "status.jsonl")
    assert len(statuses) == 10, f"incomplete run {run}"
    assert all(s["returncode"] == 0 and not s["timed_out"] for s in statuses), run
    meta = json.loads((run / "metadata.json").read_text())
    commit = meta["git_commit"]["stdout"].strip()
    recorded_code = subprocess.check_output(
        ["git", "-C", str(ROOT), "show", f"{commit}:lab/01_link.py"])
    assert hashlib.sha256(recorded_code).hexdigest() == meta["script_sha256"], run
    rows = []
    for variant in run.iterdir():
        if not variant.is_dir():
            continue
        if (variant / "rows.jsonl").exists():
            rows += read_rows(variant / "rows.jsonl")
        if (variant / "rank0.jsonl").exists():
            a, b = (read_rows(variant / f"rank{rank}.jsonl") for rank in (0, 1))
            assert len(a) == len(b)
            for x, y in zip(a, b):
                assert (x["kind"], x["mode"], x["src"], x["bytes"]) == (
                    y["kind"], y["mode"], y["src"], y["bytes"])
                assert x["correct"] and y["correct"]
                z = dict(x)
                z["rank"] = "max_per_sample"
                for key in ("gpu_us_samples", "wall_us_samples"):
                    z[key] = list(map(max, zip(x[key], y[key])))
                for key in ("gpu", "wall"):
                    z[f"{key}_us_median"] = statistics.median(z[f"{key}_us_samples"])
                z["effective_GBs"] = z["bytes"] / z["gpu_us_median"] / 1000
                rows.append(z)
    assert len(rows) == 12 * len(meta["parameters"]["sizes"]), run
    for row in rows:
        assert row["correct"]
        assert len(row["gpu_us_samples"]) == meta["parameters"]["samples"]
        assert all(x > 0 for x in row["gpu_us_samples"])
        assert abs(statistics.median(row["gpu_us_samples"])-row["gpu_us_median"]) < 1e-6
    return meta, sorted(rows, key=lambda x: (x["kind"], x["mode"], x["src"], x["bytes"]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("result_dir", type=Path)
    args = p.parse_args()
    base = args.result_dir.resolve()
    meta, rows = load(base / "run3")
    confirm_meta, confirmation = load(base / "run4")
    assert meta["script_sha256"] == confirm_meta["script_sha256"]
    def get(kind, mode, src, size, data=rows):
        return next(x for x in data if (x["kind"],x["mode"],x["src"],x["bytes"]) ==
                    (kind,mode,src,size))
    def table(size, data=rows):
        lines = ["| measured configuration | origin | GPU time (us) | wall time (us) | effective GB/s |",
                 "|---|---|---:|---:|---:|"]
        for kind in ("copy", "roundtrip", "broadcast"):
            for mode in ("eager", "graph"):
                for src in (0, 1):
                    r = get(kind,mode,src,size,data)
                    direction = f"{src}->{1-src}" + (f"->{src}" if kind=="roundtrip" else "")
                    lines.append(f"| {kind} / {mode} | {direction} | {r['gpu_us_median']:.3f} | "
                                 f"{r['wall_us_median']:.3f} | {r['effective_GBs']:.3f} |")
        return "\n".join(lines)
    csv_fields = ["run","kind","mode","src","bytes","gpu_us_median","gpu_us_min",
                  "gpu_us_max","wall_us_median","effective_GBs","correct"]
    with (base / "matrix.csv").open("w") as f:
        writer=csv.DictWriter(f,fieldnames=csv_fields)
        writer.writeheader()
        for run, data in (("run3",rows),("run4",confirmation)):
            for r in data:
                entry={k:r[k] for k in csv_fields if k in r}
                entry.update(run=run,gpu_us_min=min(r["gpu_us_samples"]),
                             gpu_us_max=max(r["gpu_us_samples"]))
                writer.writerow(entry)
    graph_copy = [get("copy","graph",s,10240)["gpu_us_median"] for s in (0,1)]
    graph_rtt = [get("roundtrip","graph",s,10240)["gpu_us_median"] for s in (0,1)]
    graph_half = [x/2 for x in graph_rtt]
    bandwidth = [get("copy","graph",s,268435456)["effective_GBs"] for s in (0,1)]
    nccl = [get("broadcast","graph",s,10240)["gpu_us_median"] for s in (0,1)]
    repeat = []
    for kind in ("copy","roundtrip"):
        for src in (0,1):
            old = get(kind,"graph",src,10240)["gpu_us_median"]
            new = get(kind,"graph",src,10240,confirmation)["gpu_us_median"]
            relative = (new / old - 1) * 100
            limit = 25 if kind=="copy" else 20
            repeat.append({"kind":kind,"src":src,"primary_us":old,"confirmation_us":new,
                           "change_percent":relative,"within_predicted_tolerance":abs(relative)<=limit})
    crossover = []
    for src in (0,1):
        series=[r for r in rows if r["kind"]=="copy" and r["mode"]=="graph" and r["src"]==src]
        # Use the <=16 KiB median as a robust small-payload floor, rather
        # than picking the fastest sample. The first >=2x floor brackets
        # the empirical transition; it is not a fitted physical constant.
        floor=statistics.median(r["gpu_us_median"] for r in series if r["bytes"]<=16384)
        previous=0
        for r in series:
            if r["gpu_us_median"]>=2*floor:
                crossover.append({"src":src,"small_floor_us":floor,
                                  "lower_bytes":previous,"upper_bytes":r["bytes"]})
                break
            previous=r["bytes"]
    summary={"label":"measured","primary_run":"run3","confirmation_run":"run4",
             "code_commit":meta["git_commit"]["stdout"].strip(),
             "script_sha256":meta["script_sha256"],"gpu":meta["gpus"],"p2p":meta["p2p"],
             "graph_copy_10KiB_us":graph_copy,"graph_rtt_10KiB_us":graph_rtt,
             "graph_half_rtt_proxy_10KiB_us":graph_half,
             "graph_copy_256MiB_decimal_GBs":bandwidth,"graph_broadcast_10KiB_us":nccl,
             "confirmation":repeat,"empirical_crossover":crossover,
             "gate":{"one_way_copy_le_15us":max(graph_copy)<=15,
                     "half_rtt_proxy_le_15us":max(graph_half)<=15,
                     "full_rtt_le_15us":max(graph_rtt)<=15,
                     "p2p_enabled_variant":"unavailable",
                     "unqualified_S1_pass":False,"M0":"NOT EVALUATED: no all-reduce or synthetic chain in S1"}}
    (base / "summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    span=lambda values: f"{min(values):.3f}–{max(values):.3f}"
    rel=base.relative_to(ROOT)
    body=f"""# S1 — The link (2026-10-27, vast-57952342)

Generated by `lab/report_s1.py {rel}`. Values below are measured component
results, not model or engine performance. Test-model results do not transfer
to the production model; no model weights were downloaded for S1.

**Machine and version:** 2 x RTX 5090; PCIe Gen5 x16 on both; NODE topology;
400 W caps. PyTorch {meta['torch']}, CUDA wheel {meta['cuda']}, NCCL
{'.'.join(map(str,meta['nccl']))}. Code commit `{summary['code_commit']}`;
script SHA-256 `{meta['script_sha256']}`. Full software lock and machine
inventory are adjacent. Runtime-loaded driver libraries are recorded in
`run3/metadata.json`; the image's preconfigured compatibility library was used
without changing the host driver. Date/time boundaries are in UTC in raw JSON;
the experiment date is also 2026-10-27 in Asia/Shanghai.

**Prediction, before the run:** 5–10 us latency proxy, 25–50 GB/s large-message
bandwidth, crossover about 256 KiB. See `prediction.json`. These are preserved
estimates, not retrospectively changed to match measurements.

**What was done:** Two directions, 64 B–256 MiB, single `copy_` and dependent
return `copy_`, eager and one captured cross-device graph; one-GPU-per-process
NCCL broadcast as reference. Seven batches per cell, ten warmups. Small graph
captures unroll 100 operations; large captures unroll four; all normalize by
the actual operation count. CUDA events are on one device's current stream,
including copy stream dependencies; CPU wall times include submission and
completion. Eager timings include Python launch gaps. For broadcast, take the
maximum of the two ranks in each sample, then its median. Byte checks cover
the entire payload, including a changed source value after capture. Timing
uses a repeated buffer; these are component microbenchmarks.

**P2P:** `can_device_access_peer` is false in both directions. NVIDIA's read
and write checks report CNS. The P2P-enabled optimization is unavailable, not
silently represented by a fallback measurement. NCCL INFO logs explicitly
select `SHM/direct/direct` and say P2P is disabled. The exact runtime DMA route
for `copy_` was not separately traced; do not call it a direct peer-memory
measurement merely because the measured bandwidth is high.

**Measured (primary full sweep, run3):** 10 KB = 10,240 bytes = 10 KiB.
Each GPU time is the median of seven per-operation batch averages.

{table(10240)}

At 256 MiB:

{table(268435456)}

**Bottleneck:** graph one-way copy {span(graph_copy)} us at 10 KiB versus
{span(bandwidth)} GB/s at 256 MiB. Small messages are latency dominated;
large-message bandwidth is in the predicted interval. Graph capture is usable,
but does not reliably reduce copy round-trip latency on this fallback path.
The graph round trip is {span(graph_rtt)} us; its HALF, {span(graph_half)} us,
is only a per-hop proxy. Broadcast is {span(nccl)} us in graph; broadcast is
not all-reduce. Communication share, tok/s, and aggregate model roofline are
not defined for this stage.

**Crossover (derived from measured sweep):** use the median <=16 KiB graph
copy time as the latency floor and bracket the first measured size reaching
twice that floor. This is an empirical operational definition, not a fitted
link constant.

| origin | small-message floor (us) | crossover bracket (bytes) |
|---|---:|---:|
"""
    for c in crossover:
        body+=f"| {c['src']} | {c['small_floor_us']:.3f} | ({c['lower_bytes']}, {c['upper_bytes']}] |\n"
    body+="\n**Confirmation (run4, key sizes, same code, after run3 exited):**\n\n"
    body+=table(10240,confirmation)
    body+="\n\n| graph variant | origin | change from run3 | pre-run tolerance | outcome |\n|---|---|---:|---:|---|\n"
    for r in repeat:
        body+=f"| {r['kind']} | {r['src']} | {r['change_percent']:+.1f}% | {25 if r['kind']=='copy' else 20}% | {'within' if r['within_predicted_tolerance'] else 'MISSED'} |\n"
    body+="""
**Prediction vs result:** the original 5–10 us range is not uniformly met by
every directional graph copy/half-RTT proxy. Bandwidth meets 25–50 GB/s. The
crossover is bracketed by a sparse size sweep; the table preserves that
uncertainty. Directional and repeated-run differences remain visible rather
than selecting a fastest result. `confirmation-prediction.json` preserves the
repeat's numeric tolerances; its outcomes are in the table and `summary.json`.

**Errors and counterexamples preserved:** run1's naive graph capture failed
with capture isolation, because the peer stream had not been enrolled. An
explicit fork/join fixed this in run2. NCCL graph broadcast then wrote all
rows but stalled during communicator destruction: the bound `graph.replay`
method retained a live graph. Release that method and reset the graph before
destroying the process group. run3 and run4 exit normally for every variant.
run1/run2 are diagnostic artifacts, overlap while teardown stalls, and are
excluded from headline numbers. Their teardown terminations are recorded,
not counted as successful exits.

**Gate and next step:** the documents conflict on the S1 metric:
`multigpu-lab.md` asks for one message and a <=15 us link latency; `HANDOFF.md`
asks for a round trip against the same number. One-way graph copy and the
half-RTT proxy meet <=15 us; the complete graph round trip does not. P2P-on
is unavailable. Therefore S1 data collection is COMPLETE WITH QUALIFICATIONS,
not an unqualified pass. This is a reason to descend to the communication
primitive before any model work, not to replace the RTT by half of it in prose.

M0 is NOT EVALUATED: its L is a 10 KiB **all-reduce inside a graph**, with a
<=12 us gate and an additional capture/chain requirement. Neither `copy_`,
RTT/2 nor broadcast can be substituted into the production projection as L.
The next bounded investigation is the NCCL all-reduce baseline (S2), checking
the host-staged path and graph execution before any peer-pointer one-shot
kernel. Do not attempt peer-memory loads when runtime peer access is false.
Do not download models or start S3–S10 before resolving the stage/gate result.
No automatic instance destruction or driver patching was performed.

**Artifacts and reproduction:** `matrix.csv` contains both runs' complete
tables, including min/max of batch averages. Each raw row retains all seven
samples, correctness status and iteration/unroll parameters. Every variant
has stdout/stderr and a status record; NCCL debug logs identify its transport.

```bash
uv venv .venv --python /venv/main/bin/python
uv pip install --python .venv/bin/python --no-cache \\
  --index-url https://download.pytorch.org/whl/cu128 -r results/2026-10-27-vast-57952342/requirements.txt
.venv/bin/python lab/01_link.py --output results/<new-date-box>/run3 --timeout 60
.venv/bin/python lab/01_link.py --output results/<new-date-box>/run4 --timeout 60 \\
  --sizes 64 10240 262144 1048576 268435456
.venv/bin/python lab/report_s1.py results/<new-date-box>
```

Reference semantics used while implementing: [PyTorch CUDA graphs and
streams](https://docs.pytorch.org/docs/2.11/notes/cuda.html), [CUDA graph
stream capture](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html),
[NCCL graph capture](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/cudagraph.html).
The script uses PyTorch operations and no architecture-specific kernel.
"""
    (base/"S1.md").write_text(body)
    log=ROOT/"lab/LOG.md"
    existing=log.read_text() if log.exists() else ""
    later="\n\n## S2 —"+existing.split("## S2 —",1)[1] if "## S2 —" in existing else ""
    text="# Multi-GPU experiment log\n\n"+body.replace("# S1 —", "## S1 —",1)+later
    log.write_text(re.sub(r"\n{3,}","\n\n",text))
    index = ["Measured on 2026-10-27, vast-57952342, 2 x RTX 5090; primary run3.",
             f"Code `{summary['code_commit'][:7]}`; seven samples per cell. Both GPUs",
             "have Gen5 x16 links; runtime peer access is false in both directions.",
             "", "| graph component | origin GPU 0 | origin GPU 1 |",
             "|---|---:|---:|",
             f"| 10 KiB one-way copy (us) | {graph_copy[0]:.3f} | {graph_copy[1]:.3f} |",
             f"| 10 KiB full round trip (us) | {graph_rtt[0]:.3f} | {graph_rtt[1]:.3f} |",
             f"| 10 KiB NCCL broadcast (us, max rank) | {nccl[0]:.3f} | {nccl[1]:.3f} |",
             f"| 256 MiB one-way copy (decimal GB/s) | {bandwidth[0]:.3f} | {bandwidth[1]:.3f} |",
             "", "One-way copy meets 15 us; full round trip does not. P2P-on is unavailable.",
             "The source documents use conflicting S1 latency definitions, so this is",
             "**complete with qualifications**, not an unqualified S1 pass. M0 requires",
             "in-graph all-reduce measurements; copy/broadcast cannot clear it.", "",
             f"[Full log](lab/LOG.md) · [Raw results and report]({rel}/S1.md) ·",
             f"[Generated size sweep]({rel}/matrix.csv). run4 confirmation is included."]
    readme=ROOT/"README.md"
    start,end="<!-- S1_RESULTS_START -->","<!-- S1_RESULTS_END -->"
    text=readme.read_text()
    if start in text and end in text:
        before,rest=text.split(start,1)
        _,after=rest.split(end,1)
        readme.write_text(before+start+"\n\n"+"\n".join(index)+"\n\n"+end+after)
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
