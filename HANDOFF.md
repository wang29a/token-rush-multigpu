# HANDOFF — Start Here

> For the next session, or for the same author three months from now. This file
> and `README.md` are sufficient; no prior conversation is required.

## Current status

**S1 and the S2 NCCL baseline are recorded (2026-10-27).** Read `lab/LOG.md`,
`results/2026-10-27-vast-57952342/S1.md`, and its `s2/S2.md`.
Full S2's one-shot comparison is unevaluated. M0 remains uncleared. No model
or GEMV/collective/norm synthetic chain was run.

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

## Next: stop this NCCL path before model work

S2 code is `lab/02_collectives.py`; generate tables with `lab/report_s2.py`.
`default1` is the full four-size/five-operation sweep; `default2` repeats the
small all-reduce case. `ring-ll1`, `tree-ll1`, `ring-simple1` are the entire
bounded cause/configuration check. All variants exit normally, including
correctness checks and graph/NCCL teardown. Do not expand the sweep after
seeing these results. Exact numbers are generated in the log, never retyped.

- The isolated all-reduce meets 12 us, while every tested 128-operation graph
  misses that budget per operation. The <=25% isolated-vs-chain prediction was
  refuted. Report both; do not use the isolated result to declare M0 cleared.
- NCCL logs identify SHM/direct/direct, with peer access false in both
  directions. The P2P-based one-shot cannot be tested as specified. A custom
  host-staged one-shot has not been implemented or measured.
- Extra serial-chain cost has not been separated into graph event/stream
  overhead versus transport. It is a negative result for this implementation
  path, not proof of an immutable physical limit of all two-5090 systems.
- No model or synthetic GEMV/collective/norm chain was run. Full S2's one-shot
  >=2x criterion and full M0 remain unevaluated/uncleared.

**Decision:** do not enter S3–S10 model implementation on the present evidence.
Any further Mode A work first needs a new bounded cause experiment (isolate
captured stream/event edges, or specify a host-staged collective), with a
numeric prediction written before running. If the M0 gate cannot be met, the
agreed fallback is Mode B, not more cards. No automatic instance destruction
or driver patching.

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
