#!/usr/bin/env python3
"""Validate S2 raw runs and generate tables, predictions vs results, decision."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import re

ROOT=Path(__file__).resolve().parents[1]


def _final(text):
    """One trailing newline; no trailing blank line."""
    return text.rstrip("\n") + "\n"
RUNS=("default1","default2","ring-ll1","tree-ll1","ring-simple1")


def read(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x]


def load(path):
    meta=json.loads((path/"metadata.json").read_text())
    sha=meta["git_commit"]["stdout"].strip()
    for filename,key in (("02_collectives.py","script_sha256"),("01_link.py","metadata_helper_sha256")):
        code=subprocess.check_output(["git","-C",str(ROOT),"show",f"{sha}:lab/{filename}"])
        assert hashlib.sha256(code).hexdigest()==meta[key],(path,filename)
    statuses=read(path/"status.jsonl")
    assert len(statuses)==(11 if path.name=="default1" else 3),path
    assert all(s["returncode"]==0 and not s["timed_out"] for s in statuses),path
    rows=[]
    for status in statuses:
        sub=path/status["variant"]
        a,b=(read(sub/f"rank{rank}.jsonl") for rank in (0,1))
        assert len(a)==len(b)
        for x,y in zip(a,b):
            assert (x["collective"],x["mode"],x["payload_bytes"]) == (y["collective"],y["mode"],y["payload_bytes"])
            for r in (x,y):
                assert r["correct"] and r["finite"] and r["fresh_pattern_correct"]
                assert len(r["gpu_us_samples"])==meta["parameters"]["samples"]
                assert all(v>0 for v in r["gpu_us_samples"])
                assert abs(statistics.median(r["gpu_us_samples"])-r["gpu_us_median"])<1e-6
            z=dict(x)
            z["rank"]="max_per_sample"
            z["run"]=path.name
            for clock in ("gpu","wall"):
                z[f"{clock}_us_samples"]=list(map(max,zip(x[f"{clock}_us_samples"],y[f"{clock}_us_samples"])))
                z[f"{clock}_us_median"]=statistics.median(z[f"{clock}_us_samples"])
            z["gpu_us_per_replay_median"]=z["gpu_us_median"]*z["unroll"]
            z["effective_algorithmic_GBs"]=z["payload_bytes"]/z["gpu_us_median"]/1000
            rows.append(z)
    assert len(rows)==(42 if path.name=="default1" else 6),path
    return meta,rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("result_dir",type=Path)
    args=p.parse_args()
    base=args.result_dir.resolve()
    loaded={r:load(base/r) for r in RUNS}
    assert len({m["script_sha256"] for m,_ in loaded.values()})==1
    meta=loaded["default1"][0]
    allrows=[r for _,rows in loaded.values() for r in rows]
    def get(run,kind,mode,size):
        return next(r for r in loaded[run][1] if (r["collective"],r["mode"],r["payload_bytes"])==(kind,mode,size))
    def value(run,mode,size):
        return get(run,"all_reduce",mode,size)["gpu_us_median"]
    isolated=value("default1","graph1",10240)
    chain=value("default1","graph128",10240)
    chain80=value("default1","graph128",81920)
    repeat=value("default2","graph128",10240)
    gain=(1-isolated/value("default1","eager",10240))*100
    deviation=(chain/isolated-1)*100
    chain_ms=chain*128/1000
    projected_step=8.35/2+1.56+chain_ms
    summary={"label":"measured components; production numbers below are projected",
             "code_commit":meta["git_commit"]["stdout"].strip(),
             "script_sha256":meta["script_sha256"],
             "default_graph1_10KiB_us":isolated,"default_graph128_10KiB_us_per_op":chain,
             "default_graph128_10KiB_ms_per_128":chain_ms,
             "default_graph128_80KiB_us_per_op":chain80,
             "repeat_graph128_10KiB_us_per_op":repeat,
             "chain_vs_single_percent":deviation,"graph_vs_eager_gain_percent":gain,
             "profiles":{r:{"10KiB_chain_us":value(r,"graph128",10240),
                            "80KiB_chain_us":value(r,"graph128",81920)} for r in RUNS},
             "gate":{"isolated_10KiB_le_12us":isolated<=12,
                     "dependent_chain_10KiB_le_12us":chain<=12,
                     "any_bounded_profile_chain_le_12us":any(value(r,"graph128",10240)<=12 for r in RUNS),
                     "full_S2_pass":False,"M0_cleared":False,
                     "one_shot":"P2P-based variant unavailable; host-staged one-shot not implemented/measured",
                     "decision":"Stop model implementation on this NCCL path; no further parameter sweep. Baseline complete, full S2 and M0 not cleared."},
             "projection":{"label":"projected, NOT a measured engine", "date":"2026-10-27",
                           "basis":"Measured default1 communication chain plus original estimated TP compute terms; extrapolated from component measurements, not from a working engine. Upstream single-card reference: token-rush Phase 4, 2026-09-09/10.",
                           "compute_terms_ms":[8.35/2,1.56],"communication_ms":chain_ms,
                           "step_ms":projected_step,"tokens_per_second":1000/projected_step,
                           "speedup_vs_upstream_reference":9.91/projected_step}}
    (base/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    fields=["run","profile","collective","mode","payload_bytes","input_bytes","output_bytes",
            "unroll","gpu_us_median","gpu_us_min","gpu_us_max","wall_us_median",
            "gpu_us_per_replay_median","effective_algorithmic_GBs","correct"]
    with (base/"matrix.csv").open("w") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in allrows:
            row={k:r[k] for k in fields if k in r}
            row.update(gpu_us_min=min(r["gpu_us_samples"]),gpu_us_max=max(r["gpu_us_samples"]))
            w.writerow(row)
    body=f"""## S2 — NCCL baseline (2026-10-27, vast-57952342)

Generated by `lab/report_s2.py {base.relative_to(ROOT)}`. These are measured
collective components, not model performance. Test-model results do not
transfer to the production model; no model was downloaded or run.

**Question:** Does the P2P-disabled NCCL path fit the latency budget for 128
dependent small SUM all-reduces per token?

**Prediction (written before the run):** graph 10 KiB 12–25 us; eager 16–35 us;
graph 80 KiB 15–35 us. The 128-operation graph: 1.536–3.200 ms at 10 KiB and
1.920–4.480 ms at 80 KiB. Graph improves 10 KiB by 10–40%, and normalized
chain latency differs from isolated graph latency by <=25%. Large-message
algorithmic bandwidth 15–45 GB/s. These estimates and the separate bounded
check predictions are preserved in `prediction.json` and
`bounded-check-prediction.json`, written before their respective runs.

**Machine/version/code:** same 2 x RTX 5090, Gen5 x16, NODE, 400 W box as S1.
Runtime peer access false both directions. PyTorch {meta['torch']}, CUDA wheel
{meta['cuda']}, NCCL {'.'.join(map(str,meta['nccl']))}. Code commit
`{summary['code_commit']}`; SHA-256 `{meta['script_sha256']}`. Every run records
parameters, UUIDs, topology, loaded CUDA libraries, code/helper hashes and
NCCL environment. Raw dates are UTC; the date is 2026-10-27 in Asia/Shanghai.

**What was done:** SUM all-reduce, all-gather, SUM reduce-scatter and broadcast
from either rank, two GPUs, one GPU per process. BF16 payloads 10/80 KiB,
1/64 MiB. `graph1` captures one operation; `graph128` captures 128 dependent
in-place SUM operations on the same tensor, for 10/80 KiB. Nine timing
batches, 20 warmups; 100 replays per small single-operation batch, 10 per
large batch, five per 128-chain batch. GPU events include the NCCL internal
stream completion edge; wall timing includes enqueue and completion. The
reported statistic takes max(rank) for each sample, then its median.

**Correctness:** complete nonuniform rank-specific patterns checked before
and after timing, with a different fresh pattern after capture. Timed SUM
inputs are zero to avoid repeated doubling overflow; SUM is still executed.
For a single correctness replay of graph128, inputs are scaled by 2^-120;
the exact expected final factor is 384, so every intermediate and final
BF16 value stays finite. Non-finite or incorrect values fail the worker.

**Measured default baseline (default1; us per operation):**

| collective | mode | 10 KiB | 80 KiB | 1 MiB | 64 MiB |
|---|---|---:|---:|---:|---:|
"""
    for kind in ("all_reduce","all_gather","reduce_scatter","broadcast0","broadcast1"):
        for mode in ("eager","graph1"):
            cells=[f"{get('default1',kind,mode,s)['gpu_us_median']:.3f}" for s in (10240,81920,1048576,67108864)]
            body+=f"| {kind} | {mode} | "+" | ".join(cells)+" |\n"
    body+="\nPayload sizes mean input bytes/rank for all-reduce, broadcast and all-gather;\nreduce-scatter has 2x payload input bytes and payload output bytes/rank.\nAll-gather output is 2x payload. Effective algorithmic bandwidth uses this\nnominal payload and is not PCIe bus bandwidth. No model roofline is claimed.\n\n"
    body+="**Measured dependent chains and bounded diagnostic profiles:**\n\n| run/profile | graph1 10 KiB (us) | chain 10 KiB (us/op) | 128 x 10 KiB (ms) | chain 80 KiB (us/op) |\n|---|---:|---:|---:|---:|\n"
    for r in RUNS:
        c=value(r,"graph128",10240)
        body+=f"| {r} | {value(r,'graph1',10240):.3f} | {c:.3f} | {c*128/1000:.3f} | {value(r,'graph128',81920):.3f} |\n"
    body+=f"""
**Bottleneck:** the default 10 KiB isolated graph is {isolated:.3f} us, but
the dependent chain is {chain:.3f} us/op ({chain_ms:.3f} ms/128),
{deviation:.1f}% above the isolated result. The repeat is {repeat:.3f} us/op
({(repeat/chain-1)*100:+.2f}%). NCCL INFO explicitly reports SHM/direct/direct;
P2P is disabled. Ring/LL does not remove the serial-chain cost; Tree/LL and
Ring/Simple are slower. This bounds what these configuration levers buy on
this code path. The extra cost is NOT proven to be pure link latency: graph
event/stream scheduling and transport behavior have not been separately
isolated. A communication-only chain also omits compute/communication
interactions present in a model.

**Prediction vs result:** isolated 10 KiB {isolated:.3f} us is below the
12–25 us range; 80 KiB {value('default1','graph1',81920):.3f} us is below
15–35 us. The predicted 128-chain ranges hold. Graph's 10 KiB gain is
{gain:.1f}%, above the predicted 10–40%. The <=25% normalized-chain agreement
prediction is REFUTED by {deviation:.1f}%. 64 MiB all-reduce achieves
{get('default1','all_reduce','graph1',67108864)['effective_algorithmic_GBs']:.2f}
GB/s algorithmic bandwidth, within the predicted 15–45 GB/s.

The bounded check predicted default repeat 11.45–17.17 us, Ring/LL 8–15 us,
Tree/LL 8–15 us, Ring/Simple 15–30 us for the 10 KiB chain. All fit their
ranges EXCEPT Tree/LL, which misses above 15 us. No tested chain profile
meets <=12 us, so the preregistered candidate-confirmation rule is not
triggered and the sweep stops here. Every configuration is reported;
none of these diagnostic environment overrides is persisted globally.

**Projection, not measurement (2026-10-27):** retain the original plan's
compute assumptions, 8.35/2 + 1.56 ms, and add the measured default1
communication-only chain. The resulting step is {projected_step:.3f} ms,
{1000/projected_step:.1f} tok/s, approximately {9.91/projected_step:.2f}x the
upstream single-card reference. Extrapolated from component measurements,
not from a working engine. This is NOT a same-box single-card baseline or a
measured model speedup. Source of single-card reference: zyhector/token-rush,
Phase 4, 2026-09-09/10. The calculation does not justify a 1.5x engine claim.

**Gate and decision:** NCCL baseline data collection is COMPLETE. Isolated
10 KiB passes the primitive <=12 us threshold, but the default and every
bounded alternative miss that budget in the 128-operation captured chain.
Do not select the isolated result and declare M0 clear. Full M0 remains
UNCLEARED: no GEMV/collective/norm synthetic chain or working engine was run.
Full S2's one-shot >=2x criterion is also UNEVALUATED: direct peer-memory
access is unavailable and no host-staged one-shot was implemented/measured.
Only two cards are allocated; no four-card results are implied.

**Next step:** stop model implementation on this NCCL path and stop parameter
hunting. Any further Mode A work must first isolate graph event/stream cost
or test a separately specified host-staged collective, with a new prediction;
do not jump to S3–S10 on this result. If the M0 gate cannot be met, the agreed
fallback remains Mode B, not adding cards. No claim that all possible
implementations or hardware are disproved is supported by these measurements.

**Superseded in part (2026-10-27):** the recorded cause probe in `s2-sync/`
attributed the 128-operation chain cost to capture-time graph structure, not
transport. The same operations with a per-step consumer cost less and do not
grow with depth, while the wait-each capture carried a far denser dependency
graph than the other variants. The stop decision above is therefore withdrawn;
the S2 numbers themselves stand unchanged. Read `s2-sync/S2-sync.md` and the
following log section. M0 was not cleared by that probe either.

**Reproduction:** use the S1 requirements lock, then:

```bash
.venv/bin/python lab/02_collectives.py --output results/<date-box>/s2/default1
.venv/bin/python lab/02_collectives.py --output results/<date-box>/s2/default2 \\
  --collectives all_reduce --sizes 10240 81920
# Repeat the preceding command with --profile ring-ll / tree-ll / ring-simple,
# using fresh output directories ring-ll1 / tree-ll1 / ring-simple1.
.venv/bin/python lab/report_s2.py results/<date-box>/s2
```

`matrix.csv` contains every aggregate cell; raw rank JSONL retains all samples,
parameters and correctness flags. `status.jsonl` records clean exit of all
variants; NCCL INFO logs identify transport/configuration. The report checks
all code hashes against recorded commits and verifies row/sample counts.
References: [PyTorch collectives](https://docs.pytorch.org/docs/2.11/distributed.html),
[NCCL graph capture](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/cudagraph.html),
[NCCL diagnostic environment overrides](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html).
"""
    (base/"S2.md").write_text(body)
    log=ROOT/"lab/LOG.md"
    old=log.read_text()
    marker="## S2 —"
    later=""
    if "## S2-sync —" in old:later="\n\n## S2-sync —"+old.split("## S2-sync —",1)[1]
    text=old.split(marker,1)[0].rstrip()+"\n\n"+body+later
    log.write_text(_final(re.sub(r"\n{3,}","\n\n",text)))
    index=["S2 NCCL baseline measured on 2026-10-27, same 2 x RTX 5090 box.",
           f"Code `{summary['code_commit'][:7]}`. BF16 SUM, nine samples; max rank per sample.","",
           "| default all-reduce component | 10 KiB | 80 KiB |",
           "|---|---:|---:|",
           f"| eager (us/op) | {value('default1','eager',10240):.3f} | {value('default1','eager',81920):.3f} |",
           f"| one-operation graph (us/op) | {isolated:.3f} | {value('default1','graph1',81920):.3f} |",
           f"| 128-operation graph (us/op) | {chain:.3f} | {chain80:.3f} |",
           f"| 128-operation graph (ms/128) | {chain_ms:.3f} | {chain80*128/1000:.3f} |","",
           "The isolated primitive meets 12 us; all tested dependent-chain profiles miss",
           "that budget. **M0 remains uncleared; full S2 one-shot comparison is unevaluated.**",
           "The chain cost was later attributed to capture-time graph structure, not transport;",
           "see the S2-sync block below.",
           f"[S2 report]({base.relative_to(ROOT)}/S2.md) · [Full data]({base.relative_to(ROOT)}/matrix.csv)."]
    readme=ROOT/"README.md";text=readme.read_text()
    start,end="<!-- S2_RESULTS_START -->","<!-- S2_RESULTS_END -->"
    if start in text:
        before,rest=text.split(start,1);_,after=rest.split(end,1)
        text=before+start+"\n\n"+"\n".join(index)+"\n\n"+end+after
    else:
        text=text.replace("## Documents",start+"\n\n"+"\n".join(index)+"\n\n"+end+"\n\n## Documents",1)
    readme.write_text(_final(text))
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
