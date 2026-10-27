# HANDOFF — Start Here

> For the next session, or for the same author three months from now. This file
> and `README.md` are sufficient; no prior conversation is required.

## Current status

**S1, S2, the S2-sync cause probe, the S3 split sweep, the S4 64-layer chain and
the S6 Qwen3-8B decode are recorded (2026-10-27).** Read `lab/LOG.md`,
`results/2026-10-27-vast-57952342/S1.md`, its `s2/S2.md`, `s2-sync/S2-sync.md`,
`s3/S3.md`, `s4/S4.md` and `s6/S6.md`. S2-sync withdrew S2's "stop this
NCCL path" decision: the 128-operation chain miss came from capture-time graph
structure, not transport. S4 then captured 64-layer chains in one graph per
card: exact correctness, additivity within about 2%, 64-layer communication
share about 11-12%, and M0's kill criteria not triggered - but M0 is not
cleared because the chain is bf16 rather than the production int4 kernel. The
Qwen3-8B test model was run in S6; the production 27B model has not been run.

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

## Next: make TP=2 text-equivalent before M1

S6 code is `lab/05_tp2_decode.py` (`lab/report_s6.py`); the Hugging Face
reference is `lab/05b_hf_reference.py`. Weights were downloaded to
`/tmp/qwen3-8b` (16 GB, not in git, not persistent). An earlier TP1/TP2 pair ran
before the engine was committed; it is archived under `s6/_uncommitted/` and is
not used. Exact numbers are generated in the log, never retyped.

- Measured decode, 96 greedy tokens after 8 warmup: TP=1 80.3 tok/s
  (12.45 ms/token, 15.3 GiB), TP=2 117.6 tok/s (8.50 ms/token, 8.2 GiB/card),
  so **1.46x**. The speed criterion (>= 1.4x) is met.
- Correctness is not established. First-step logits are close to the independent
  Hugging Face CPU forward, with the same argmax, but the saved TP=2 output
  diverges from TP=1 and loops. FP32 collectives produce the same saved tokens.
  This does not establish sharded-GEMM rounding as the cause or rule out later
  cache, position, attention, or execution errors. Treat the attribution in the
  original generated S6 report as a hypothesis, not a verified conclusion.
- The reported divergence index is an index into a discontinuous saved array:
  `benchmark()` advances generation for eight warmup steps without saving
  those tokens, then concatenates the initial prediction and timed outputs.
  Do not describe array index 4 as the true first divergent decode step.
- The 1.46x here is a test-model number and does not transfer to 27B. It is also
  not the 1.45x component projection; the agreement is coincidence.
- 26.8% of the TP=2 step does not shard (8.50 ms vs half of 12.45 ms). At the
  measured 10 KiB collective cost, 72 collectives would be about 0.6 ms, so most
  of that remainder is half-width GEMM efficiency, not measured communication.

**Decision:** do not start M1 on this engine. On resume, first record a new
numeric prediction and repair the diagnostic recording: preserve every token
through warmup or reset state after warmup; evaluate a contiguous matching
prefix rather than total matching token count (`report_s6.py` currently uses
the latter). Compare TP=1, TP=2 and HF on the same forced token sequence,
step by step, including the KV/position update, to locate the first actual
logit divergence. Check finiteness from computed tensors rather than trusting
the hardcoded row flag. Only then choose a controlled precision change such
as an FP32 residual stream. The unchanged saved tokens after FP32 all-reduce
are evidence for that specific variant, not proof of the underlying cause.

## Stop checkpoint (2026-10-27)

This handoff includes the read-only review of the S6 artifacts. No engine or
benchmark was changed during stop preparation. The original raw logs and
reports remain intact; the qualifications above supersede their stronger
cause/divergence claims. DSH's latest recorded turn is complete, with no open
step or pending tool calls. Code, pre-run predictions and raw artifacts through
S6 are committed and pushed. Verify the current HEAD against remote `main`
before stopping, as specified below. On resume, start with S6 correctness,
not a new speed claim or another model stage. Weights, virtual environment
and the SSH private key are excluded from Git and may need recreation on a
replacement instance; the experiment artifacts are in the remote repository.

S1's original gate ambiguity remains documented in `S1.md`: one-message vs
full roundtrip against the same 15 us number. `run3`/`run4` are its primary and
confirmation; `run1`/`run2` retain early capture/teardown mistakes. Copy/broadcast
are never substituted for M0's all-reduce L.

## Backup on this instance

This is an interruptible/spot instance; `/workspace` is **not** a persistent
volume. At checkpoints, commit and push code, pre-run predictions, raw
artifacts and the generated report. Verify local HEAD against the remote.
Keep weights, caches, virtual environments, credentials and rendezvous files
out of Git. `docs/RESUME.md` has the exact rebuild recipe for a fresh instance,
what survives stop versus destroy, and where the chat session store lives. If authentication blocks push, say so explicitly and produce a
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
