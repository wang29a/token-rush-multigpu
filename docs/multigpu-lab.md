# Multi-GPU Laboratory Manual

> **This is a learning ladder, not the project plan.** The plan — whether the work
> is worth doing, how the 1.5× is accounted for, and how the result is reported —
> is in `docs/multigpu.md`. This manual answers one question only: **how to learn
> multi-GPU inference one stage at a time, with every stage stating its
> bottleneck and the reason for the next step.**

**Provenance.** The experimental loop, the single-card thesis, and every
single-card figure referenced here come from
[`zyhector/token-rush`](https://github.com/zyhector/token-rush). The eleven-stage
ladder and all multi-GPU projections are original to this repository, which is an
independent personal extension and not an official token-rush project.

## 0. The method in one sentence

You have already run the single-card loop to the point of habit: **pose one
question → write the prediction → make the smallest change → measure one number
→ explain the difference between prediction and measurement → decide the next
step.** Multi-GPU work uses the same loop. Two things change: **the rulers** and
**the bottleneck checklist**. The loop does not change; what changes is how the
numbers are read.

## 1. The loop is unchanged; two things change

### 1.1 Rulers: from one to four

A single card is judged by one number: % of wall. Multi-GPU work requires four at
once, because they trade against each other:

| Metric | Definition | Question it answers |
|---|---|---|
| **S(n)** | T(1) / T(n) | How much faster the added cards actually are |
| **E(n)** | S(n) / n | Efficiency of the added cards; 1.5× on two cards gives E = 0.75 |
| **comm share** | communication time / step time | Whether this split is being consumed by communication |
| **straggler** | max(rank time) / mean | Whether one card is holding the others back |

Add one metric that does not drift with the implementation: **% of aggregate
wall**, with n × 1701 GB/s as the denominator. It is the same ruler as the
single-card 81%, with a different denominator. On two cards it falls to about
62%. That is not a regression; it is the price of tensor parallelism, and it
should be computed in advance.

### 1.2 Bottleneck checklist: six single-card checks plus three multi-GPU checks

The six single-card checks are already familiar: bandwidth, latency, launch
overhead, occupancy, host, and caches. Multi-GPU work adds three, and all three
are new:

1. **Overlap check.** Is step time ≈ compute + communication (serial), or
   ≈ max(compute, communication) (overlapped)? The first case means that no
   communication on the critical path was hidden. Read the gap between the two
   ranks' timelines in nsys; do not guess.
2. **Communication-share check.** comm / step. Above 20% means the next step
   should address communication; below 5% means communication is not the problem
   and time spent on it is wasted.
3. **Straggler check.** max(rank) / mean(rank). Above 5% means the shard is
   uneven: rank 0 performs more sampling, KV lengths are asymmetric, or a shard
   boundary falls in the wrong place.

## 2. Test bench: two rigs, one test model

### 2.1 Cards: the 5090 as the primary rig, NVLink as the comparison

Only four quantities actually vary along the ladder: per-card bandwidth, link
latency L, link bandwidth, and batch size. A single hardware point yields a
single data point, and it invites the mistake of treating "1.5× on a 5090" as a
general law of tensor parallelism. Two rigs are therefore necessary, but only one
stays resident.

| Rig | Configuration | Cost | What it teaches | What it cannot teach |
|---|---|---|---|---|
| **Primary** | 2 × RTX 5090 (PCIe, no NVLink, P2P disabled by default) | ~$1.2–4/h | TP on the worst interconnect: communication cost is magnified and the mechanism is clearest; the same interconnect topology as the project | How conclusions change under a good interconnect |
| **Comparison** | 2 × A100 80 GB SXM (NVLink 600 GB/s), 3–4 hours | ~$1–1.5 per card-hour | The same code with L 3–5× smaller and bandwidth 1.2× higher; shows how break-even moves | Expensive; short experiments only |
| Optional | 4 × any card, for the n=4 column of S7 | — | Scaling curve | — |
| Optional | 8 × H20 (common in China: NVLink 900 GB/s, 96 GB, 4 TB/s, weak compute) | Cheap | Isolates bandwidth from compute: fast decode, slow prefill | Weak compute distorts prefill and EP conclusions |

**Minimum viable set:** 2 × 5090 resident, plus 2 × A100 SXM for three hours.
That is sufficient.

**Before renting, spend $3 measuring the vendor's board.** The same "2×5090"
instance may be x16/x16 or x8/x8, or the two cards may hang off different PCIe
switches. `nvidia-smi topo -m` and the latency measured in S1 decide whether the
machine is worth renting by the hour. This step is itself S1.

### 2.2 Test model: Qwen3-8B (bf16 and an int4 variant)

It fits in 16 GB on a single card, which gives a TP=1 baseline on the same
machine; it runs TP=2 on two cards and TP=4/PP=4 on four. Hidden size 4096 and
just over thirty layers are not far from the production model (5120, 64 layers).
The Hugging Face implementation is clean, and int4 weights exist.

When substituting another model, check each item:

- `num_key_value_heads` — **sets the ceiling on TP**. A model with 8 KV heads can
  reach TP=8; the production model has 4, so TP ≤ 4. That difference is itself a
  lesson to measure at S7.
- `hidden_size` — the all-reduce payload is hidden × 2 B; for the production
  model it is 10 KB.
- `num_hidden_layers` — collectives per token = 2 × layers.
- GDN or linear attention — **the test model should not have it**; learn plain
  Transformer TP first. The production model has it, which adds another sharding
  dimension (split by head, plus a depthwise convolution); face that at S10.
- Existing int4 weights (GPTQ / AWQ / GGUF) — needed at S3 to measure "the
  heavier the quantization, the less TP pays".

**Write no architecture-specific kernel in the test engine.** Use PyTorch and
Triton so that the same file runs on the 5090 and on the A100/H100; otherwise the
three hours on the comparison rig are wasted.

**Production model:** Qwen3.8-27B int4 at 4.25 bpw, used only at S10.

**Conclusions from the test model may not be cited.** Its hidden size is smaller
and it has fewer layers, so the communication share is inflated: good for
learning (the mechanism is visible), bad for reporting (the numbers do not
transfer). Put this on the first line of the log.

## 3. Eleven stages (plus an optional branch, S7+)

Each stage is "baseline → one optimization → one number → one decision". If the
pass criterion is missed, do not climb further; first explain why.

| Stage | Baseline | Optimization at this stage | What is measured | Pass criterion |
|---|---|---|---|---|
| S1 | a single `memcpyPeer` | in-graph + P2P enabled | L(10 KB), bandwidth, crossover | L(10 KB) ≤ 15 µs |
| S2 | NCCL all-reduce | hand-written one-shot (write + flag + spin) | µs at 10 KB / 1 MB / 64 MB | one-shot 2× faster at 10 KB |
| S3 | single-card linear layer | split N / split K + all-reduce | four splits over an M sweep | predict the crossover in advance |
| S4 | eager 64-layer chain | two per-device graphs / NCCL in graph / spin in graph | launch tax, ms/step | graphs ≥3× faster than eager |
| S5 | 1-layer block | 64 layers + all-reduce fused into residual + norm | comm share, straggler | share ≤ 25% at 64 layers |
| S6 | single-card test model | TP=2 full model | S(2), % aggregate wall, correctness | ≥ 1.4× |
| S7 | TP | PP / DP / PP+TP | map over bs ∈ {1, 8, 64} | one sentence per cell |
| S7+ | TP=2 (dense) | EP / all-to-all (MoE, optional) | byte ledger at bs=1 vs bs=32 | explain why EP does not pay at bs=1 |
| S8 | serial all-reduce | two micro-batches / chunked all-reduce | overlap ratio (sum vs max) | hide ≥ 30% of communication |
| S9 | TP prefill | ring / sequence parallel | 32k / 128k prefill | SP ≥ 20% faster than TP |
| S10 | test model | 27B production model | raw / spec / 200k | within 10% of projection |
| S11 | a pile of tables | decision memo | tok/s/$, tok/s/W | defend the trade-offs |

### S1 — The link (0.5 day)

- **Question:** What does one message from one card to the other cost, and at
  what size does bandwidth take over?
- **Build:** `lab/01_link.py`. `nvidia-smi topo -m` and
  `torch.cuda.can_device_access_peer`; bidirectional ping-pong from 64 B to
  256 MB; once eager and once inside a graph; NCCL broadcast as a reference.
- **Prediction first:** L(10 KB) ≈ 5–10 µs, bandwidth ≈ 25–50 GB/s, crossover
  ≈ 256 KB.
- **Likely observations:** x8/x8 boards halve the bandwidth; with P2P disabled,
  L doubles.
- **Lesson:** latency and bandwidth are two regimes. 10 KB always sits on the
  latency side; this is the first foundation of the ladder.

### S2 — Collectives (1 day)

- **Question:** What does the collective that TP needs actually cost at our
  message sizes?
- **Build:** all-reduce / all-gather / reduce-scatter / broadcast at 10 KB /
  1 MB / 64 MB, on 2 and 4 cards; NCCL against a hand-written one-shot (each card
  writes its shard to every peer, sets a flag, spins, and reduces).
- **Prediction first:** one-shot 2–3× faster at 10 KB; NCCL wins at 64 MB.
- **Lesson:** the operating ranges of ring, tree and one-shot; why vLLM and
  SGLang ship their own all-reduce kernels rather than calling NCCL directly.

### S3 — One linear layer, four splits (1 day)

- **Question:** When is splitting a layer in half faster than not splitting it?
- **Build:** one `[N, K]` layer, M ∈ {1, 8, 64, 512, 4096}; four variants:
  single card; split N (output dimension) + all-reduce; split K (input
  dimension) + all-reduce; single-card split-K as a control.
- **Derive first:** T₂ = T₁/2 + comm, so the split wins only if comm < T₁/2. At
  M=1, T₁ is bytes/bandwidth, and the inequality can be evaluated directly.
- **Prediction first:** at M=1, split-N wins because the bytes halve; **int4
  loses sooner than bf16** because there are fewer bytes to split — the heavier
  the quantization, the less TP pays. This is a real connection between the
  project and the quantization work.
- **Lesson:** the break-even equation. Why `lm_head` (huge N) is worth splitting
  and a small projection is not.

### S4 — Two-card graph capture (1 day)

- **Question:** Can the whole step still be one graph? If not, what is the launch
  tax?
- **Build:** a 64-layer toy chain (GEMV + all-reduce + norm); measure eager, two
  per-device graphs, NCCL inside the graph, and one-shot spin inside the graph.
- **Prediction first:** eager is 3–10× slower than graphed (5–10 µs per launch ×
  two hundred launches).
- **Lesson:** the 9.11 ms launch tax measured on one card returns unchanged on
  two. Graphs are the precondition for every later measurement: if the graph does
  not hold, every number that follows is host time.

### S5 — One block, then a stack (1 day)

- **Question:** How do communication share and the fixed cost change with depth
  and width?
- **Build:** a TP block shaped like the test model, run at 1 / 8 / 32 / 64
  layers; fuse the all-reduce into residual + RMSNorm (one more shard in the
  sum).
- **Prediction first:** the share per layer is roughly constant; the fixed cost
  amortises with depth.
- **Lesson:** depth is friendly to TP; width is not necessarily. Where the fixed
  tax comes from and why it does not shrink with n.

### S6 — The full test model (2 days)

- **Question:** Does everything still hold when assembled into a real model?
- **Build:** roughly 300 lines of TP engine: weight sharding (head dimension,
  KV-head dimension, vocabulary dimension), KV sharding, graphs, sampling merge,
  decode loop. Check correctness by comparing TP2 and TP1 logits; at the first
  divergence, check whether the reference token is in the top 5.
- **Gate:** ≥ 1.4×. If it falls short, return to S5's numbers and look for fixed
  cost.
- **Lesson:** where the shard boundaries are. The sampling merge must be
  **recomputed on both cards and deterministic**; otherwise one card accepts 3
  drafts and the other 4, and the two states diverge silently.

### S7 — The parallelism menu (2 days)

- **Question:** TP / PP / DP / PP+TP — which wins at which batch size?
- **Build:** the same test model under four strategies; bs ∈ {1, 8, 64}.
- **Output:** a **3×4 map** (rows = strategy, columns = batch, cells = tok/s and
  comm share). That table is the complete answer to "multi-GPU optimization
  strategy".
- **Lesson:** PP loses at bs=1 (it re-reads, row by row, the weights that
  speculation has already amortised) and wins at bs=64; DP is always 2× but does
  not change latency. Every cell must be explained in one sentence and one
  number.

### S7+ — The MoE row (optional, 2 days)

Conclusion first: **the project does not need a design change for MoE** (the
27B model is dense plus GDN), but **learning multi-GPU work requires this
section**, because the active frontier in 2026 is MoE, and it splits S7's map by
batch.

- **Question:** When the model becomes MoE, which conclusions from earlier
  stages still hold?
- **Build:** a 30B-class MoE with roughly 3B active parameters (the
  Qwen3-30B-A3B class); at int4 it is about 17 GB and fits on two cards. Measure
  three things: (a) bytes actually read per token at bs=1, and tok/s; (b)
  all-to-all share of step time at bs=32; (c) TP=2 against EP=2 on the same
  model.
- **Prediction first:** at bs=1 an MoE is far faster than a dense model of the
  same parameter count, because it reads the 3B active parameters rather than the
  30B; EP is negative at bs=1, since an all-to-all for one token is pure latency.
  Beyond batch 32 the reverse holds: all-to-all and expert load imbalance become
  the main cost.
- **Lesson:** MoE replaces the entire "bytes → wall" ledger. Sparse activation
  discounts the bandwidth cost at bs=1 in proportion to the active fraction, and
  once batch arrives the cost shifts from reading weights to moving tokens. The
  same model on the same cards has a different optimal parallelism in each
  regime; that is exactly what S7's map expresses, with one more dimension.

Three changes, in order of importance: **the parallel axis moves from TP to EP
(all-to-all)**; **memory stops being a question of parameters per card and
becomes a question of expert placement** (hot experts drag down the slowest card,
and the straggler check becomes real here for the first time); **the bs=1 ledger
inverts** (sparse means reading less, not "splittable").

### S8 — Overlap (1–2 days)

- **Question:** Of the 128 all-reduces, how many can be hidden?
- **Build:** split the M=8 verify rows into two micro-batches so that the
  all-reduce of the first overlaps the compute of the second; then try a chunked
  all-reduce.
- **Prediction first:** nothing can be hidden at M=1 (there is no independent
  work); 30–40% at M=8.
- **Lesson:** **the batch created by speculative decoding is what makes overlap
  possible.** This is a real connection between the single-card project and this
  experiment, and the most direct evidence for why TP at bs=1 is hard.

### S9 — Long context and sequence parallelism (1 day)

- **Question:** How much more does sequence parallelism give on prefill than TP?
- **Build:** split 32k / 128k prompts in two and use ring attention (reuse an
  existing implementation; do not write one).
- **Prediction first:** SP 1.7–1.9×, TP 1.5×, because what is exchanged is KV,
  not the whole activation.
- **Lesson:** prefill and decode want different parallelism. This also explains
  why the prefill row in `docs/multigpu.md` looks better than the decode rows.

### S10 — The production model (3–4 days)

- **Question:** Can the cost model learned on the test model predict the 27B
  numbers?
- **Build:** shard the checkpoint from the upstream repository and run it on two
  cards (or adapt the test engine to the 27B geometry).
- **Gate:** within 10% of `T(n) = 8.35/n + 1.56 + 0.128·L`.
- **Lesson:** whether the model extrapolates. A large error means the fixed-cost
  term was not understood.

### S11 — The memo (0.5 day)

- **Output:** two pages — the conditions under which each strategy applies, its
  cost (tok/s/$, tok/s/W), and how the conclusions change if L becomes 20 µs or
  the batch becomes 64.
- **Lesson:** compress the eleven stages into three sentences. That is what you
  can defend.

## 4. Bottleneck → next step (decision table)

After a stage finishes, find the check that fired in this table before deciding
what to do next. This is the one page of the manual worth memorising.

| Symptom (which check fired) | Next lever | Typical gain |
|---|---|---|
| eager much slower than graphed | graph capture | 3–10× |
| message doubles, time does not | latency-bound: one-shot / fusion / fewer collectives | 1.2–2× |
| step ≈ compute + comm, not max | micro-batches, chunked all-reduce | hide 30–50% of communication |
| comm share > 20% | fuse the all-reduce into the norm; carry more tokens per collective | 5–15% |
| straggler > 5% | re-shard; move sampling off the critical card | 2–5% |
| already at ≥ 80% of aggregate wall | stop splitting; quantize | — |
| S(2n) ≈ S(n) | the fixed tax has plateaued; stop adding cards | — |

## 5. Log template

Record every stage against this skeleton. **The prediction must be written before
the run**; that is the only difference between learning and benchmarking.

```markdown
## Sn — name (date, machine)
**Question:**
**Prediction (written before the run):** specific numbers and the reasoning
**What was done:** files, key parameters
**Measured:**
| config | ms | tok/s | GB/s | % wall | comm share | straggler |
**Bottleneck:** which check fired, and the number that proves it
**Prediction vs result:** how far off the prediction was, and why
**Next step:** what this result forces, and against what threshold
```

The "prediction vs result" line is the most valuable one. Without it, eleven
stages produce a book of numbers; with it, you learn where your intuition was
wrong.

## 6. What to write and what to reuse

**Write yourself.** This is the learning. The code is written on the rented machine
(`/workspace/lab/`); this repository holds this list, the stage code and the completed
logs.

```
lab/01_link.py          # point-to-point latency and bandwidth
lab/02_collectives.py   # four collectives x three sizes x 2/4 cards
lab/oneshot.py          # hand-written all-reduce kernel (or a Triton version)
lab/03_linear.py        # four ways to split one linear layer
lab/04_graph.py         # four ways to capture a two-card graph
lab/05_stack.py         # 64-layer toy chain
lab/tp_engine/          # the ~300-line TP engine
lab/07_menu.py          # TP/PP/DP/TP+PP
lab/07b_moe.py          # S7+: MoE byte ledger and all-to-all (optional)
lab/08_overlap.py       # micro-batch overlap
lab/LOG.md              # the template above
```

**Reuse directly.** The repository's `bench/` timing methodology,
`scripts/matrix_table.py`, and, after S10, the GEMV / Marlin kernels and the
rival reproduction recipes (`docs/baselines.md`).

**Do not write yourself.** NCCL, ring attention, schedulers. The subject here is
how to use and measure them, not how to rebuild them.

**After every stage, copy the `LOG.md` section, the stage code and the raw logs
from `results/<date>-<box>/` back.** When the instance is destroyed, all of it goes
with it (the rule in `docs/environment.md`), so "what to copy before retreating"
is part of the plan, not a closing chore.

## 7. Time and cost

- **Time:** about 45–70 hours for the eleven stages; S7+ adds two days. At
  15–20 hours per week, three to four weeks.
- **Machines:** 2 × 5090 at an estimated $2.5/h, 55 hours ≈ **$140**; 2 × A100
  SXM for three hours **$20–30**; the four-card hours **$30–50**. Total
  **$150–250**, not much more than the single-card project, because most
  questions can be answered within a day.
- **Downloads:** 17 GB for the 27B model, 3.9 GB for the draft, about 16 GB bf16
  plus 4.5 GB int4 for the test model; once per new machine. Anything outside
  `/workspace` is destroyed with the instance, so do not stockpile.

## 8. What you should be able to draw at the end

A 4×4 table (TP / PP / DP / TP+PP × bs 1/8/64), each cell carrying tok/s and
communication share; add an MoE row if S7+ was run. Plus one sentence: which cell
flips if L becomes 20 µs, or the batch becomes 64.

And this sentence, deliverable without notes:

> On two 5090s, 27B at 4.25 bpw in a single stream at TP=2 is 1.5×, not 2×: 128
> all-reduces of 10 KB cost 0.77 ms against a 4.2 ms half-step. At batch 64, PP
> overtakes it, because the bottleneck moves from latency to weight bandwidth.
> With MoE the bs=1 ledger inverts: only the 3B active parameters are read, and
> EP becomes a net loss.

When that sentence comes without notes, the multi-GPU material is learned.

## 9. Relationship to the project plan

M0 in `docs/multigpu.md` (interconnect measurement plus projection) is S1 + S2 +
S4 here. M1–M3 correspond to S6 + S10. M4 (rivals on the same box, on the same
day) is not in this manual; it belongs to the report. In other words: run this
manual first and the answer to M0 is already in hand; if L looks bad, the PP and
DP cells of S7 show what else the second card can do.
