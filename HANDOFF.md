# HANDOFF — Start Here

> For the next session, or for the same author three months from now. This file
> and `README.md` are sufficient; no prior conversation is required.

## Current status

**S1, S2 and the S2-sync cause probe are recorded (2026-10-27).** Read
`lab/LOG.md`, `results/2026-10-27-vast-57952342/S1.md`, its `s2/S2.md` and its
`s2-sync/S2-sync.md`. S2-sync withdrew S2's "stop this NCCL path" decision: the
128-operation chain miss came from capture-time graph structure, not transport,
and a dependent chain with a per-step consumer stays under 12 us. M0 still
remains uncleared, and no model or synthetic GEMV/collective/norm chain was run.

## Settled decisions (not open for discussion)

1. **Scope:** single stream (bs=1) is the primary case. Multiple GPUs are used
   here for **latency**, not capacity. Start at two cards; TP ≤ 4 (the 27B model
   has 4 KV heads, which is the architectural ceiling).
2. **Primary rig:** 2 × 5090 (the worst interconnect, where the mechanism is
   clearest). **Comparison rig:** 2 × A100 80 GB SXM for three hours (same code;
   see how the conclusions change when L is 3–5× smaller).
3. **Test model:** Qwen3-8B (bf16 and int4). **Production model:**
   Qwen3.8-27B int4 at 4.25 bpw, used only at S10.
4. **MoE is S7+:** optional, two days, off the main line.
5. **Code is written on the rented machine** (`/workspace/lab/`). This repository
   holds the plan, the stage code, the logs and the result summaries.

## Next: measure the synthetic M0 chain, once

S2-sync code is `lab/02_sync_probe.py`; generate its tables with
`lab/report_s2_sync.py`. `run4`/`run5` are the complete forward and
reverse-order measurements; `run1` (debug-mode graph-lifetime leak) and
`run2`/`run3` (Python 3.12 `init_rank` failure) are withdrawn or failed
diagnostics and must not be promoted. Exact numbers are generated in the log,
never retyped.

- The S2 wait-each chain is a capture artifact, not a link limit: it was the
  only variant whose per-step cost grew with depth, and its DOT dump carries a
  far denser dependency graph than tail-wait, same-stream or the consumer
  variants. The stop decision is withdrawn; the S2 numbers themselves stand.
- A dependent 10 KiB chain with a per-step consumer costs about 10.2 us both via
  same-stream native NCCL and via ProcessGroup with per-step waits, so the
  result does not depend on the native communicator. It is flat in depth.
- That meets the 12 us component budget but does not clear M0. The consumer is
  one elementwise multiply; a real layer adds GEMV and RMSNorm compute whose
  overlap with the all-reduce is unmeasured.
- Decomposition, to be checked, not assumed: communication about 8.4 us/step,
  the local multiply about 0.8 us, and roughly 1.1 us/step of serialization that
  neither term contains. The projection in the log charges that residual
  explicitly.
- Full S2's one-shot >=2x criterion remains unevaluated: peer access is false
  and no host-staged one-shot was implemented or measured.

**Decision:** one bounded synthetic M0 chain is the next experiment - a
GEMV-shaped block, the 10 KiB all-reduce and an RMSNorm-shaped block per layer,
captured with the consumer-style dependency, numeric prediction first, at most
two complete runs. If it holds at or below 12 us per layer, Mode A earns model
implementation; if not, the fallback is Mode B, not more cards. No model work
and no 1.5x claim before that measurement. No automatic instance destruction or
driver patching.

S1's original gate ambiguity remains documented in `S1.md`: one-message vs
full roundtrip against the same 15 us number. `run3`/`run4` are its primary and
confirmation; `run1`/`run2` retain early capture/teardown mistakes. Copy/broadcast
are never substituted for M0's all-reduce L.

## Backup on this instance

This is an interruptible/spot instance; `/workspace` is **not** a persistent
volume. At checkpoints, commit and push code, pre-run predictions, raw
artifacts and the generated report. Verify local HEAD against the remote.
Keep weights, caches, virtual environments, credentials and rendezvous files
out of Git. If authentication blocks push, say so explicitly and produce a
Git bundle; a local commit/bundle alone is not an off-instance backup.

Direct SSH push now works with a repository-scoped deploy key configured in
local Git configuration; `origin` is the SSH URL. The private key stays outside
the checkout. The GitHub plugin can read but its Git-object write test returned
403, so do not assume its reported user push permission grants API write scope.

## Do not

- Start with the 27B model. Use the test model and iterate five times a day.
- Use `/tmp` as a working directory; it is cleared.
- Write the prediction after the run. That is explanation, not prediction.
- Cite the numbers marked *projected* in `docs/` as conclusions.
- Present this repository as official token-rush work. It is an unaffiliated
  extension (see `README.md`, Provenance).

## File map

| File | Purpose |
|---|---|
| `README.md` | What this is, the ladder at a glance, and the relationship to upstream |
| `docs/multigpu-lab.md` | Eleven stages, the bottleneck decision table, the log template |
| `docs/multigpu.md` | Project plan: the 1.5× accounting, L is everything, kill criteria |
| `HANDOFF.md` | This file |

## Existing log and artifacts

`lab/LOG.md` is generated from recorded JSONL rows. Do not retype measured
numbers. The original pre-run prediction is `prediction.json` in the result
directory. Fix and repeat predictions were also written before those runs;
failed predictions and failed variants remain in the artifacts.
