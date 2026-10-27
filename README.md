# Token Rush · Multi-GPU Experiments

**The single-card method, applied one level up: on two RTX 5090s, with a
2 × A100 SXM rig as the comparison for three hours.**

This is neither an engine repository nor a proposal for "a multi-GPU solution".
It is an **experimental ladder**. It begins with the latency of one point-to-point
message and climbs, one stage at a time, to TP=2 on Qwen3.8-27B. Each stage
changes a single variable and records the same five entries: baseline,
optimization, number, bottleneck, next step. The purpose is to develop judgement
about multi-GPU inference rather than to collect conclusions.

> **Status: link, collective, capture-structure, 64-layer chain and Qwen3-8B
> TP=2 decode measured, 2026-10-27.** P2P is unavailable on this box. The isolated
> all-reduce meets the component threshold; the earlier 128-operation chain miss
> is attributed to capture-time graph structure, not transport. A 64-layer
> production-shard chain captures in one graph per card, is bit-exact, and equals
> compute plus communication within about 2%, with an 11-12% communication
> share. An S3 split sweep puts the bs=1 crossover on the winning side: split-N
> is about 1.5x one layer at M=1, and loses above M=64. Qwen3-8B decode measures
> **1.46x** (80.3 -> 117.6 tok/s), but TP=2's greedy text diverges from TP=1 at
> token 4, so **S6 is not passed** and M1 is blocked. M0's kill criteria are not
> triggered, but M0 is not cleared (bf16 chain, int4 kernel and fixed term still
> projected).

<!-- S1_RESULTS_START -->

Measured on 2026-10-27, vast-57952342, 2 x RTX 5090; primary run3.
Code `15314bc`; seven samples per cell. Both GPUs
have Gen5 x16 links; runtime peer access is false in both directions.

| graph component | origin GPU 0 | origin GPU 1 |
|---|---:|---:|
| 10 KiB one-way copy (us) | 7.039 | 11.054 |
| 10 KiB full round trip (us) | 20.278 | 19.005 |
| 10 KiB NCCL broadcast (us, max rank) | 10.851 | 10.842 |
| 256 MiB one-way copy (decimal GB/s) | 44.252 | 44.257 |

One-way copy meets 15 us; full round trip does not. P2P-on is unavailable.
The source documents use conflicting S1 latency definitions, so this is
**complete with qualifications**, not an unqualified S1 pass. M0 requires
in-graph all-reduce measurements; copy/broadcast cannot clear it.

[Full log](lab/LOG.md) · [Raw results and report](results/2026-10-27-vast-57952342/S1.md) ·
[Generated size sweep](results/2026-10-27-vast-57952342/matrix.csv). run4 confirmation is included.

<!-- S1_RESULTS_END -->

<!-- S2_RESULTS_START -->

S2 NCCL baseline measured on 2026-10-27, same 2 x RTX 5090 box.
Code `69cfc23`. BF16 SUM, nine samples; max rank per sample.

| default all-reduce component | 10 KiB | 80 KiB |
|---|---:|---:|
| eager (us/op) | 17.865 | 19.310 |
| one-operation graph (us/op) | 9.391 | 12.887 |
| 128-operation graph (us/op) | 14.309 | 18.424 |
| 128-operation graph (ms/128) | 1.832 | 2.358 |

The isolated primitive meets 12 us; all tested dependent-chain profiles miss
that budget. **M0 remains uncleared; full S2 one-shot comparison is unevaluated.**
The chain cost was later attributed to capture-time graph structure, not transport;
see the S2-sync block below.
[S2 report](results/2026-10-27-vast-57952342/s2/S2.md) · [Full data](results/2026-10-27-vast-57952342/s2/matrix.csv).

<!-- S2_RESULTS_END -->

<!-- S2_SYNC_RESULTS_START -->

S2-sync cause probe measured on 2026-10-27, same 2 x RTX 5090 box.
Code `4041b1c`. BF16 SUM, nine samples; max rank per sample.

| 10 KiB, depth 128 | us/step |
|---|---:|
| wait-each capture (the S2 structure) | 14.305 |
| tail-wait, communication only | 8.422 |
| same-stream NCCL, communication only | 8.355 |
| same-stream plus a per-step consumer | 10.228 |
| ProcessGroup plus the same consumer | 10.255 |

The S2 chain cost was a capture-time dependency artifact: the wait-each graph
had 8384 edges against 383 for tail-wait, growing
with depth. A dependent chain with a real consumer stays near 10.2 us and is flat
in depth, so the 'stop this NCCL path' decision was withdrawn. **M0 remains
uncleared**; the S4 chain below uses bf16 weights, not the int4 production layer.
[S2-sync report](results/2026-10-27-vast-57952342/s2-sync/S2-sync.md) · [Full data](results/2026-10-27-vast-57952342/s2-sync/matrix.csv).

<!-- S2_SYNC_RESULTS_END -->

<!-- S3_RESULTS_START -->

S3 linear-layer splits measured on 2026-10-27, same 2 x RTX 5090 box.
Code `e968042`. bf16, N=8704, K=5120, seven samples; max rank per sample.

| M | single | split-N | split-K | split-N speedup |
|---:|---:|---:|---:|---:|
| 1 | 57.2 | 39.1 | 39.5 | 1.46x |
| 8 | 59.8 | 47.5 | 48.5 | 1.26x |
| 64 | 123.7 | 79.5 | 100.8 | 1.56x |
| 512 | 253.8 | 314.7 | 478.2 | 0.81x |
| 4096 | 1699.9 | 2280.1 | 3430.3 | 0.75x |

Split-N wins at every measured M; split-K, with twice the payload, diverges as M grows.
int4 is untested (no kernel). Next: the Qwen3-8B TP=2 test model, prediction first.
[S3 report](results/2026-10-27-vast-57952342/s3/S3.md) · [Full data](results/2026-10-27-vast-57952342/s3/matrix.csv).

<!-- S3_RESULTS_END -->

<!-- S4_RESULTS_START -->

S4 64-layer capture probe measured on 2026-10-27, same 2 x RTX 5090 box.
Code `473853d`. bf16 GEMV at hidden 5120 / half-N 8704, 10 KiB SUM, nine samples.

| shard shape, 64 layers | us/layer | pass (ms) |
|---|---:|---:|
| chain (GEMV + all-reduce + RMSNorm), graph | 73.813 | 4.724 |
| compute only | 63.888 | 4.089 |
| communication only | 8.277 | 0.530 |
| eager chain | 75.313 | 4.820 |

Chain = compute + communication within 2.3%; communication share
11.2% at 64 layers. Capture and exact correctness hold. The chain is
one bf16 GEMV per layer, not the production int4 layer, so **M0 is not cleared** and
the step projection (6.82 ms, 1.45x) remains projected. Next: the
Qwen3-8B TP=2 test model, prediction first.
[S4 report](results/2026-10-27-vast-57952342/s4/S4.md) · [Full data](results/2026-10-27-vast-57952342/s4/matrix.csv).

<!-- S4_RESULTS_END -->

<!-- S6_RESULTS_START -->

S6 Qwen3-8B decode measured on 2026-10-27, same 2 x RTX 5090 box.
Code `07f22d1`. bf16, 96 greedy tokens after 8 warmup, one graph per card.

| variant | tok/s | ms/token | peak GiB/card |
|---|---:|---:|---:|
| tp1 | 80.3 | 12.45 | 15.3 |
| tp2 | 117.6 | 8.50 | 8.2 |
| tp2_fp32reduce | 114.5 | 8.74 | 8.2 |

TP=2 is **1.46x** TP=1, but its greedy text leaves TP=1 at token 4 and
loops; fp32 collectives do not change it. **S6's speed bar is met, S6 is not passed.**
Next: fp32 residual stream, then one re-run of this comparison.
[S6 report](results/2026-10-27-vast-57952342/s6/S6.md) · [Full data](results/2026-10-27-vast-57952342/s6/matrix.csv).

<!-- S6_RESULTS_END -->

## Documents

| Document | Question it answers |
|---|---|
| [docs/multigpu-lab.md](docs/multigpu-lab.md) | **How to learn it.** Eleven stages, each with a prediction and a pass criterion written down in advance; the bottleneck → next-step decision table; the log template. |
| [docs/multigpu.md](docs/multigpu.md) | **Whether it is worth doing, how to do it, and how to report it.** The accounting that makes a second card 1.5× rather than 2×, why a 10 KB all-reduce is the whole problem, the TP/PP/DP trade-offs, and the kill criteria. |

## Provenance

The idea tested here originates in
[`zyhector/token-rush`](https://github.com/zyhector/token-rush). That project
established the thesis this repository extends: at batch size 1 the cost of a
second card is a serial chain of small collectives, so the card is worth 1.5×,
not 2×. It also established the measurement discipline used throughout this
repository — write the prediction before the measurement, report the fraction of
the roofline wall, account for bytes, and capture the step in a graph — and
produced every single-card figure quoted here.

The multi-GPU plan, the eleven-stage ladder, and multi-GPU experiments are
original to this repository. **S1 link and S2 NCCL results are measured; all engine numbers
remain projections.** The recorded runs, predictions and raw artifacts are in
`lab/LOG.md` and `results/`. This repository is an independent
personal experiment; it is not an official token-rush project and carries no
endorsement from it.

When citing the single-card results (100.9 tok/s, 81.0% of wall, 228 / 357 / 378
tok/s, 224 s for the 256k prefill), cite token-rush, Phase 4, 2026-09-09/10. When
citing the multi-GPU plan, cite this repository as an unaffiliated extension.

Paths such as `docs/environment.md`, `docs/progress.md`, `docs/traps.md` and
`docs/baselines.md` refer to the upstream repository, with the exception of the
two documents listed above.

## Ladder

| Stage | Baseline → optimization | Pass criterion |
|---|---|---|
| S1 | single copy → in-graph + P2P enabled | L(10 KB) ≤ 15 µs |
| S2 | NCCL all-reduce → hand-written one-shot | 2× faster at 10 KB |
| S3 | single-card linear layer → split N / split K | predict the crossover |
| S4 | eager 64-layer chain → two-card graph capture | ≥ 3× faster than eager |
| S5 | 1 layer → 64 layers, all-reduce fused into the norm | comm share ≤ 25% |
| S6 | single-card test model → TP=2 full model | ≥ 1.4× |
| S7 | TP → PP / DP / PP+TP | one sentence per cell |
| S7+ | dense → MoE / EP (optional) | explain why EP loses at bs=1 |
| S8 | serial all-reduce → micro-batch overlap | hide ≥ 30% of communication |
| S9 | TP prefill → ring / sequence parallel | ≥ 20% faster than TP |
| S10 | test model → 27B production model | within 10% of projection |
| S11 | a pile of tables → decision memo | defend the trade-offs |

## Outputs

- A **parallelism × batch map**: for every cell, tok/s, communication share, and
  one sentence of explanation.
- An experimental log carrying **prediction versus measurement**, with the
  prediction written before the run. A wrong prediction is a result.
- One sentence that can be delivered without notes:

  > On two 5090s, 27B at 4.25 bpw in a single stream at TP=2 is 1.5×, not 2×:
  > 128 all-reduces of 10 KB cost 0.77 ms against a 4.2 ms half-step.

## Conventions

- Every number carries machine, date and version. Rival reproduction recipes stay
  upstream; they are not duplicated here.
- Write the prediction first, then run. If the prediction is wrong, state where
  it was wrong.
- Nothing that cannot be measured enters the conclusions.
- Code runs on rented machines (stage code in this checkout's `lab/`). After
  each stage, commit and push `LOG.md`, the stage code and raw results; verify
  the remote commit. This development instance is interruptible and has no
  persistent workspace volume.

## License

As with upstream, no license is attached yet. Cite the upstream measurements
referenced in the text (zyhector/token-rush, Phase 4, 2026-09-09/10).

[![](https://img.shields.io/badge/powered_by-dsh-4D6BFE?style=flat-square&logo=deepseek&logoColor=white)](https://github.com/deepseek-ai/deepseek-harness)
