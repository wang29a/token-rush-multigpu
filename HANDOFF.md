# HANDOFF — Start Here

> For the next session, or for the same author three months from now. This file
> and `README.md` are sufficient; no prior conversation is required.

## Current status

**S1 data collection completed with qualifications on 2026-10-27.** The
repository now contains the link probe, generated report and raw measurements.
Read `lab/LOG.md` and `results/2026-10-27-vast-57952342/S1.md`. No all-reduce or
model measurements exist yet. M0 is not evaluated.

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

## Next: resolve the communication gate, before model work

S1 code is `lab/01_link.py`; generate measured tables with `lab/report_s1.py`.
`run3` is the full sweep and `run4` the bounded confirmation. Both complete
all variants, correctness checks and teardown normally. `run1`/`run2` retain
capture-isolation and graph/NCCL teardown mistakes; they are diagnostic only.
The code commit and source hash are in every run's metadata.

- Runtime peer access is unavailable in both directions; P2P-on was not tested.
- One-way graph copy and half-roundtrip proxy meet 15 us. Full roundtrip does
  not. Exact generated measurements, repeats and size sweep are in `lab/LOG.md`.
- The original documents conflict: the lab calls S1 one-message latency,
  while this handoff originally called it roundtrip latency. Do not silently
  replace roundtrip with half-roundtrip or claim an unqualified S1 pass.
- Copy graphs work after explicitly forking and joining the remote stream.
  NCCL graph executables must be reset, including releasing bound replay
  methods, before process-group destruction.

The next bounded investigation is **NCCL all-reduce inside a graph (the S2
baseline)**, to resolve the actual primitive's cost. Write a new numeric
prediction first. This is a descent into the communication cause, not clearance
for S3 or model implementation. S1 copy/broadcast times cannot be used as M0's
all-reduce L. M0 additionally requires the synthetic chain/capture check.

S1's one-message gate is 15 us; M0's actual all-reduce gate is 12 us.
If M0 fails, Mode B remains the fallback. No driver patching or automatic
instance destruction. Do not add cards to evade a failed gate.

## Backup on this instance

This is an interruptible/spot instance; `/workspace` is **not** a persistent
volume. At checkpoints, commit and push code, pre-run predictions, raw
artifacts and the generated report. Verify local HEAD against the remote.
Keep weights, caches, virtual environments, credentials and rendezvous files
out of Git. If authentication blocks push, say so explicitly and produce a
Git bundle; a local commit/bundle alone is not an off-instance backup.

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
