# Token Rush · Multi-GPU Experiments

**The single-card method, applied one level up: on two RTX 5090s, with a
2 × A100 SXM rig as the comparison for three hours.**

This is neither an engine repository nor a proposal for "a multi-GPU solution".
It is an **experimental ladder**. It begins with the latency of one point-to-point
message and climbs, one stage at a time, to TP=2 on Qwen3.8-27B. Each stage
changes a single variable and records the same five entries: baseline,
optimization, number, bottleneck, next step. The purpose is to develop judgement
about multi-GPU inference rather than to collect conclusions.

> **Status: planning. No code and no measurements.** The first table will appear
> here once S1 produces a number.

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

The multi-GPU plan, the eleven-stage ladder, and all multi-GPU numbers are
original to this repository. **No multi-GPU number here has been measured.** Each
is a projection and is labelled as one. This repository is an independent
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
- Code runs on rented machines (`/workspace/lab/`). After each stage, copy
  `LOG.md`, the stage code and the raw results back; destroying the instance destroys everything
  on it.

## License

As with upstream, no license is attached yet. Cite the upstream measurements
referenced in the text (zyhector/token-rush, Phase 4, 2026-09-09/10).
