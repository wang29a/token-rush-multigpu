# Multi-GPU experiment log

Test-model results do not transfer to the production model. S1 is a link probe, not an engine or all-reduce benchmark.

## S1 — The link (2026-10-27, vast-57952342)

**Question:** What does a dependent 10 KiB GPU-to-GPU round trip cost; when does bandwidth dominate?

**Prediction (written before the run):** 5–10 us latency proxy, 25–50 GB/s (decimal) large-message bandwidth, crossover approximately 256 KiB; estimates from the plan. Known before running: NODE topology and the driver P2P read check reports CNS. P2P-enabled results may be unavailable. Preserve a failed prediction.

**Prediction artifact:** `results/2026-10-27-vast-57952342/prediction.json`.

**What was done:** Pending. No timings recorded yet.

**Measured:** Pending.

**Bottleneck:** Pending.

**Prediction vs result:** Pending.

**Next step:** S1 <=15 us is separate from M0 <=12 us for an in-graph all-reduce plus capture validation.
