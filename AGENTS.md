# AGENTS.md

Constraints for agents in this repository. Read `README.md` first; these rules
outlive a single session. Write in the plan's register: factual and dated, not
promotional.

## The claim

Two cards at bs=1 buy latency: 1.5× the tokens at 0.75× the tokens per dollar,
not 2×. `L`, the latency of the 10 KB all-reduce, decides whether it pays.
Accounting and falsifiers are in `docs/multigpu.md`.

## Stage

S1, S2 and the S2-sync cause probe were measured on 2026-10-27; see `lab/LOG.md`.
S1 remains qualified (P2P unavailable; one-way/RTT gate definitions differ).
S2's isolated all-reduce meets the 12 us component threshold and its
128-operation graph misses it, but the S2-sync probe attributes that miss to
capture-time graph structure - a dense dependency graph built from repeated
waits - not transport: a dependent chain with a per-step consumer costs about
10.2 us at 10 KiB and is flat in depth. S2's stop decision is therefore
withdrawn; its raw numbers stand.

An S3 split sweep on one 8704x5120 bf16 layer measured the crossover: split-N
is about 1.5x the single-card layer at M=1 and still wins at M=64, but loses at
M=512 and M=4096; split-K, with twice the payload, diverges lower. TP therefore
pays at bs=1 and stops paying at large batch, so S7's batch cells must not reuse
those timings as if they were bs=1. int4 is untested (no int4 kernel installed).

An S4 capture probe measured 64-layer chains at the production shard shape and
a toy shape. Capture, exact correctness and additivity hold: the shard chain
equals compute plus communication within about 2%, its 64-layer communication
share is about 11-12%, and graphs beat eager by 3x only where the chain is
launch-bound (the toy), not where it is weight-bound (about 1.02x). M0's kill
criteria are not triggered, but **M0 is not cleared**: the chain used bf16
weights rather than the production int4 kernel, and the 1.56 ms fixed term
remains an estimate.

S6 then ran Qwen3-8B itself: TP=1 80.3 tok/s, TP=2 117.6 tok/s (1.46x), one
graph per card, logits within 0.24/0.19 of a Hugging Face CPU forward. The speed
bar is met, but **S6 is not passed**: TP=2's greedy text leaves TP=1 at token 4
and loops, and fp32 collectives change nothing, so the cause is bf16
sharded-GEMM rounding rather than a structural error. M1 is blocked until the
text matches; the next step is an fp32 residual stream. No 1.5x claim is made.
Computed outcomes remain `projected`, unmeasured inputs `estimate`; `measured`
is reserved for recorded runs with raw artifacts. Measurements enter only
through `lab/LOG.md` and `results/<date>-<box>/`.

## Numbers

- A measured result records machine, run date, code commit, parameters and raw
  artifacts; without them it is not evidence (rivals improve). `measured` also
  requires a prediction-versus-result entry; `projected` is computed from stated
  inputs; `estimate` is an assumption. Never promote a label in prose.
- A projection names its basis and date beside the number: measured inputs,
  estimated inputs, calculation. "Extrapolated from component measurements, not
  from a working engine" is the form.
- Single-card figures come from `zyhector/token-rush`, Phase 4, 2026-09-09/10
  (vast 36542); do not re-derive them or quote development numbers as report
  numbers.
- Rival results are re-run on the same box and day; a number from another
  machine is not a baseline.
- Test-model (Qwen3-8B) results do not transfer to 27B and are not conclusions.
- Generate measured tables from the logs; never retype numbers.

## Files

| File | Owns |
|---|---|
| `README.md` | What this is, the ladder, upstream relationship |
| `CLAUDE.md` | Symlink to this file; both tools read one text |
| `docs/multigpu-lab.md` | Stages, predictions, pass criteria, decision table, log template |
| `docs/multigpu.md` | The 1.5x accounting, L, TP/PP/DP, kill criteria |
| `HANDOFF.md` | Current state, settled decisions, next action |
| `lab/` | Stage code; paths in `docs/multigpu-lab.md` §6 |
| `lab/LOG.md` | Prediction, numbers, bottleneck, decision per stage |
| `results/<date>-<box>/` | Raw logs and artifacts behind those entries |

Weights, caches and credentials stay out of git.

The development instance is interruptible and `/workspace` is not a persistent
volume. Commit and push code, predictions and results at stage checkpoints;
verify the remote commit. A local commit is not an off-instance backup. If
authentication fails, report the failure and retain a Git bundle for recovery.

## Method

One stage, one decision: record the baseline and each variant the stage names,
then decide once. Write the prediction, with a number, before the run; a run
without one supports no claim.

A missed criterion is recorded, not explained away: repeat, descend to the
cause, or stop in writing. Stage gates and the M0 gate are separate; passing S1
does not clear M0. If M0 fails, the fallback is Mode B (8-bit weights at equal
latency), not more cards. Commit the script with the log entry it produced.

The test engine is PyTorch and Triton, with no architecture-specific kernels;
one file must run on both the 5090 and the A100 rig. The sampling merge is
deterministic and recomputed on both cards; a divergent RNG corrupts both states
silently.

Reuse NCCL, ring attention and schedulers; build only what a stage needs. Failed
predictions stay in the log with the counterexample that refuted them; the error
is a result.

## Scope

In scope: bs=1 latency; two cards for the primary ladder; TP ≤ 4 (4 KV heads);
Qwen3-8B as the test model; 27B only at S10. Four-card runs only where a stage
names them. The 5090 rig is PCIe-only; the A100 SXM comparison rig is rented
as-is.

Latency is the only objective; batch and overlap experiments price the S7/S8
alternatives, not a throughput product. Out of scope: serving traffic,
concurrency, schedulers, multi-node, RDMA, models that do not fit two cards,
any claim of 2x.
