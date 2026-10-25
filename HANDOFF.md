# HANDOFF — Start Here

> For the next session, or for the same author three months from now. This file
> and `README.md` are sufficient; no prior conversation is required.

## Current status

**The plan is complete. Zero lines of code, zero measurements.** The repository
contains documentation only.

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

## Next: S1 (one hour, four numbers)

Once on the machine, answer one question: **what does a 10 KB round trip between
the two cards cost, in microseconds?**

1. `nvidia-smi topo -m` (PIX / PXB / SYS)
2. `nvidia-smi -q -d POWER` (power cap)
3. `torch.cuda.can_device_access_peer(0, 1)`
4. A timed 10 KB `copy_` (a rough L)

**Prediction (write it down before running):** L ≈ 5–10 µs, bandwidth ≈
25–50 GB/s.
**Pass criterion:** L ≤ 15 µs. If the board is x8/x8, or L > 20 µs, destroy the
instance and rent elsewhere. Do not compromise on this. Fifteen microseconds is
the stage gate; the M0 project gate in `docs/multigpu.md` is stricter (12 µs), so
a result between 12 and 15 µs passes S1 and still rules out Mode A.

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

## The first log entry

Following the template in `docs/multigpu-lab.md` §5, create `lab/LOG.md` on the
machine. The first entry is S1. When the run is done, copy that section and the
raw logs back and commit them to this repository.
