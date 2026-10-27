# HANDOFF — Start Here

> For the next session, or for the same author three months from now. This file
> and `README.md` are sufficient; no prior conversation is required.

## Current status

**S1, S2, the S2-sync cause probe and the S4 64-layer chain are recorded
(2026-10-27).** Read `lab/LOG.md`, `results/2026-10-27-vast-57952342/S1.md`, its
`s2/S2.md`, `s2-sync/S2-sync.md` and `s4/S4.md`. S2-sync withdrew S2's "stop this
NCCL path" decision: the 128-operation chain miss came from capture-time graph
structure, not transport. S4 then captured 64-layer chains in one graph per
card: exact correctness, additivity within about 2%, 64-layer communication
share about 11-12%, and M0's kill criteria not triggered - but M0 is not
cleared because the chain is bf16 rather than the production int4 kernel. No
model has been run.

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

## Next: the Qwen3-8B test model at TP=2

S4 code is `lab/03_block_chain.py`; generate its tables with `lab/report_s4.py`.
`run1`/`run2` are the complete forward and reverse-order measurements; all 16
variant runs exited cleanly. Exact numbers are generated in the log, never
retyped.

- Capture holds at 64 layers and every replay is bit-identical to a
  no-collective reference. Additivity holds: the shard chain equals compute
  plus communication within about 2%.
- The shard chain is weight-read bound, so eager and graphed costs are nearly
  equal (about 1.02x). Graphs are a 3x lever only on the launch-bound toy chain.
  Do not repeat the claim that graphs are a 3x lever for the production shape.
- The 64-layer communication share is about 11-12%, comfortably under S5's 25%
  criterion, and it is a conservative bound: the production layer has more
  compute than the one-GEMV-per-layer chain.
- M0's kill criteria (L > 12 us, capture failure) are not triggered, but M0 is
  **not** declared cleared: the probe used bf16 weights, not the int4 4.25 bpw
  kernel, and the 1.56 ms fixed term is still an estimate from the single-card
  engine. The projection at the measured L is 6.82 ms, 146.6 tok/s, 1.45x
  upstream - projected, not measured.

**Decision:** the next deliverable is one measured model number, not more
synthetic shapes. Build the S5 block stack and then Qwen3-8B at TP=2 on this
box, with a numeric prediction first: target at least 1.4x, consistent with the
1.45x projection; below 1.35x means the fixed tax is worse than modelled.
Fallback stays Mode B, not more cards. No 1.5x claim before that measurement.

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
