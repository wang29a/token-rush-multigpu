# Multi-GPU — one stream, two cards

**Status: proposal, 2026-10-25. Nothing in this file is measured.** The
single-card engine is closed (phases 0–5, `docs/progress.md` step 35). This is
the plan for testing the same thesis one level up, and it is written to be
killed by its first week: the whole outcome is one number — the latency of a
10 KB all-reduce between two 5090s — measured before any engine work starts.
The learn-it-yourself ladder that produces that number is
`docs/multigpu-lab.zh-CN.md` (S1–S11; its S1 + S2 + S4 are this file's M0).

## What a second card can buy

Three different products, priced differently. The choice is the first decision
in the document, so it is the first table.

| | what it buys | measured / projected | new engineering | verdict |
|---|---|---|---|---|
| **A. Latency** | the same model, one stream, ~1.5x faster | raw 100.9 → **145–160**; spec 228 → **330–360** | tensor parallelism, a cross-card collective inside the graph, 4–6 weeks | **this plan** |
| **B. Quality** | the same latency, lossless weights | KL 0.0232 → **≤0.005**, speed ~parity (94 raw) | an int8 g128 kernel + an 8-bit GPTQ pack, 1–2 weeks | the fallback, and the one that closes the project's unmet row |
| **C. Capacity** | 2 independent engines, 456 tok/s aggregate | 2 x 228, unchanged latency | one day (two ports, one supervisor) | rejected here; named because it is the cheapest option |

C is the honest baseline: two cards running two `tokenrush.serve` processes
double the throughput for a day's work. **This plan is not that.** It is for
halving the latency of one stream, which is the project's stated objective and
the only use of a second card that needs engine work. The report must say out
loud what A costs relative to C: **A is 1.5x the tokens and 0.75x the tokens
per dollar, and it lowers the roofline fraction from 81% to ~62%.** Both are
properties of tensor parallelism, not of this implementation. "Speed is the
only objective" makes the purchase legitimate; it does not make it efficient.

B is the no-regret path: it shares the packing and the kernels with A, needs
no interconnect work at all, and if week one kills A the cards are still not
wasted.

## The claim

Every production multi-GPU deployment in the field is a throughput deployment:
split the model so that N users' batches fit, and pay for the all-reduce out of
the batch. At bs=1 there is no batch to pay with. The collective is one hidden
vector — 10 KB, which is nothing to a PCIe link and everything to a dependency
chain — and it sits in the middle of every layer, **128 times per token**
(64 layers x 2: the mixer's output projection and the MLP's down projection
are both row-parallel and both need a sum). Nothing else in the step can
overlap it, because at bs=1 there is no other work.

That is the single-card thesis one level up, in its purest form. On one card
the design-point mismatch was worth 19–40 points of the wall. On two cards the
same mismatch reappears as a *serial* cost that a batch would have hidden, so
the second card is worth **1.5x, not 2x**, and the gap between those two
numbers is the project.

The corollary matters as much: **tensor parallelism does not compound with
speculation.** The verify step is bandwidth-bound and shards; the draft is
latency-bound and does not. At 13.8 ms/step the draft is ~2.6 ms (19%), so the
speculative row gains no more than the raw row and may gain less.

## The physics of the second card

### The step, decomposed

The measured single-card step is 9.91 ms for 13.65 GB of weight bytes. Written
as the model this plan is built on:

```
T(n) ≈ 8.35/n + 1.56 + 0.128 · L        [ms, L in µs]
```

- `8.35` = 13.65 GB / 1701 GB/s / 0.96 — the weight stream at the GEMV
  efficiency the engine measures (96% inside one graph), split n ways.
- `1.56` = the part of the step that does not shard with width: the 48-layer
  graphed GDN chain (**1.44 ms measured**, `docs/environment.md`), attention
  over the live length, sampling, the graph's serial dependencies.
- `0.128` = 128 collectives x 10 KB of payload, i.e. L µs each per step.

At n=1 the model gives 9.92 ms against a measured 9.91 — 0.1%. That is the
only thing calibrating it, and the two terms it cannot see — whether the GDN
chain shards (it might, at half the heads) and whether the half-N GEMV loses a
point — are both inside `1.56 ± 0.3`. **The projection's sensitivity is dominated
by `L`, which is the point.**

| n | bytes/card | ideal | 1.56 | 128 · L | total | tok/s | vs 1 |
|---|---|---|---|---|---|---|---|
| 1 (measured) | 13.65 GB | 8.35 | 1.56 | — | 9.91 | 100.9 | 1.00 |
| 2, L=4 µs | 6.83 | 4.18 | 1.56 | 0.51 | 6.25 | 160 | 1.59 |
| **2, L=6 µs** | 6.83 | 4.18 | 1.56 | 0.77 | **6.51** | **154** | **1.52** |
| 2, L=10 µs | 6.83 | 4.18 | 1.56 | 1.28 | 7.02 | 142 | 1.41 |
| 2, L=15 µs | 6.83 | 4.18 | 1.56 | 1.92 | 7.66 | 131 | 1.29 |
| 4, L=12 µs | 3.41 | 2.09 | 1.56 | 1.54 | 5.19 | 193 | 1.91 |
| 8 (architecturally out) | 1.71 | 1.04 | 1.56 | 3.84 | 6.44 | 155 | 1.54 |

Read the table as: the second card is worth 1.3–1.6x and everything about that
number is L. The fourth card is worth another 1.2x if a four-way collective
costs 12 µs, and the eighth is worth nothing without a switch fabric — it is
back where two cards were. **Eight is also impossible for this model:** GQA
gives 4 KV heads, so TP ≤ 4 unless KV is replicated, which throws away exactly
the bandwidth that the split was for.

### The one number

`L` is the latency of a 10 KB all-reduce between two 5090s, inside a captured
graph. It is set by four things, in this order:

1. **Peer-to-peer.** GeForce parts ship with P2P disabled by the driver; the
   community's multi-5090 boxes document a P2P driver configuration
   (`rtx6kpro`), and the serving engines are adding `sm_120` PCIe all-reduce
   paths precisely because the default is bad. With P2P a 10 KB write lands in
   ~1–2 µs; without it the payload makes two PCIe round trips through
   host-pinned memory.
2. **The collective's kernel.** NCCL's small-message path, or our own one-shot
   (write + flag + spin + reduce). A device-side spin is what makes it
   capture-safe; see below.
3. **The graph.** Every launch saved is 5–10 µs of host time and every launch
   inside the graph is captured for free. 128 collectives per step is 128
   launches if they are eager — the 9.1 ms launch tax the project already
   measured would come straight back.
4. **The payload.** 10 KB at bs=1, 80 KB at M=8. Both are latency-dominated;
   the link's bandwidth never enters the decode number.

Projected: **4–6 µs with P2P, 8–15 µs without.** The 1.4x line sits at about
L = 10 µs; the kill line, where the second card stops paying for the project at
all, is 12–15 µs.

### The interconnect, stated plainly

- No NVLink on a 5090, and there will not be. PCIe Gen5 x16 is 63 GB/s
  theoretical, ~50 GB/s practical; most consumer boards bifurcate to x8/x8,
  halving it. Decode does not care; prefill does (below).
- P2P is a driver property, not a hardware one. It must be measured on the
  actual box (`torch.cuda.can_device_access_peer`, then a bandwidth matrix at
  1 KB–256 MB), and it must never become the baseline: a patched driver is not
  something a reproducible report can depend on.
- Power is not the constraint the internet thinks it is: the 500 W-capped card
  lost 1% (`docs/environment.md`), so 2 x 500 W is fine. The PSU, the slot
  spacing and the heat are the real desktop problems.

### The roofline, restated

The project's signature metric is the fraction of the 1701 GB/s wall. With two
cards the wall is 3402 GB/s and the ideal step is 4.01 ms. At 6.51 ms the
engine would hold **62% of the aggregate wall**, against 81% on one card.

That is not a regression to hide; it is the price of latency, and it is
computable in advance. The report should carry both numbers side by side:
*1.5x the tokens, 0.75x the roofline.* A reader who understands the second
number cannot be fooled by the first.

## Scope: which parallelism, and why the others lose

| | splits | per-step communication | at bs=1 | verdict |
|---|---|---|---|---|
| **TP** | heads / rows inside a layer | 128 x 10 KB, **latency** | halves the bytes; the fixed tax stays | **chosen** |
| PP | layers | 1 x 10 KB per token | serialises depth; micro-batching the verify rows re-reads the weights per row | rejected |
| DP | requests | none | two streams, no latency change | not this project (option C) |
| SP | tokens (prefill) | KV exchange | ~2x prefill, needs replicated weights | 4-card option, not 2 |
| draft on card 2 | model roles | 358 KB/step of features | frees memory, does not shorten the step | fallback for memory only |

PP deserves the arithmetic because it looks like the obvious answer and is the
worst one. Two stages, each T/2, m micro-batches: makespan `(m+1)·T/2`. The
verify step already amortises the weights across K+1 rows — 8 rows cost 1.13x
one row — so splitting those rows into micro-batches means paying T/2 *per
row*: 8 micro-batches cost 4.5x what one batched pass costs. **PP destroys
exactly the amortisation that speculation creates.** DP is the opposite: it is
free, and it is a different product.

The draft-on-card-2 variant is worth naming and not worth building for speed:
the draft for step t+1 is conditioned on the target's hiddens produced by the
verify of step t, so it is on the critical path regardless of which card it
runs on. It buys memory, and memory is not what two 32 GB cards lack here.

## What the engine already gives us

- **The step is one graph, and a multi-device graph is the same problem with
  event edges.** The address-stability trap is already paid for
  (`docs/traps.md`); every buffer a captured graph touches is already
  preallocated and owned by the engine.
- **State is explicit and preallocated** (`tokenrush/state.py`): the shard is
  "half the KV heads, half the recurrent heads", and the commit logic
  (`n_accepted`, `slot`, `pos_t`) is already device-side and symmetric, so it
  runs unchanged on both cards.
- **The reduction the collective needs already exists.** `add_rmsnorm` takes
  the residual contribution as `[S, T, hidden]` fp32 split-K partials and sums
  them in the same kernel (`tokenrush/fused.py`). A peer's partial is one more
  `S` slice: the all-reduce becomes a read of one extra 10 KB buffer plus an
  arrive/wait, fused into a kernel that already launches — no new launch, no
  new buffer, no change to the graph's structure.
- **Layers are pure functions** (`tokenrush/model.py`): the shard boundary is a
  function boundary, not an architectural one.
- **GDN shards per head, and its conv is depthwise** — no cross-head mixing, so
  one collective per layer, after the output projection. 48 V heads / 16 QK
  heads split as 24/8 per card, and the 3:1 V:QK grouping means the boundary
  falls on a QK-head boundary: 5120 conv channels and 3072 gate channels per
  card.
- **Attention splits 24 Q / 4 KV heads into 12/2** — the same ratio GQA was
  built for, and TP=4 (1 KV head each) is the architectural ceiling.
- **The verify batch (M=8) amortises the collective** over 2.5–5 accepted
  tokens; the collective's payload grows to 80 KB but stays latency-bound.
- **The Marlin M-row kernel's halves are clean**: 17408/2 = 8704 and
  124160/2 = 62080, both multiples of the 64-column tile, and the layout's
  permutation is per-tile so it needs no new packer.

## What breaks

1. **Multi-device capture, and NCCL inside it.** The #1 risk. CUDA can capture
   cross-device work through event edges and NCCL documents graph capture, but
   neither is proven in torch on `sm_120`, and the host-bounce transport that
   P2P-off forces may not capture at all. The device-side answer is a
   **one-shot all-reduce**: each card's kernel writes its partial to the peer's
   buffer (or host-pinned staging), sets a flag, spins on the peer's flag,
   reduces. Two per-device graphs, replayed back to back, rendezvous inside the
   kernels with no host involvement and no NCCL — the second replay must be
   enqueued before the first graph can block, which it is because a replay
   returns immediately. The reduce itself prefers to fold into the residual
   add + RMSNorm that follows it, which already sums split-K partials
   (`tokenrush/fused.py`); if the spin cannot live inside that kernel, a
   separate spin-wait kernel per layer costs graph nodes (~0.1 ms for 128 of
   them), not host launch time. This is the design to build; NCCL is the thing
   to try first because it is 20 lines.
2. **Hangs.** A spin-wait that misses its peer hangs the box, and the community
   reports exactly that (`P2P deadlocks`). A timeout flag and a watchdog that
   aborts the process are part of the kernel, not an afterthought.
3. **Kernel geometry.** Every Triton kernel here is written against one model's
   constants (48 heads, 16 QK, 24/4, conv 10240). The GDN step, the attention
   decode and the prefill kernels all need their grids re-derived at 24/8,
   12/2, 5120 — a mechanical but wide change, and the M-row GDN's "two
   programs per head" trick has to be re-picked at the smaller head count.
4. **Sampling across a split vocabulary.** 124160 logits per card, one global
   argmax/top-k. The merge must be **replicated and deterministic** — both
   cards exchange their local top-64 and apply the same tie-break — so that
   both commit the same `n` and the same token without a second round trip.
   This is where an elegant design turns into a correctness bug if it is
   skipped: one card accepting 3 drafts and the other 4 corrupts both states
   silently.
5. **Prefill, which is the *good* news.** The collective becomes bandwidth-bound
   at chunk size (4096 x 5120 x 2 B = 42 MB per layer), but 42 MB against 64
   layers of 4096-token GEMMs is ~4% of the chunk's time, so prefill should
   scale **1.8–1.9x** — better than decode, precisely because its messages are
   large. 256k prefill measured 224 s (1160 tok/s, step 9); the projection is
   **~120 s**. This is the counter-intuitive result to verify first: TP is
   efficient exactly where the payload is big.
6. **Correctness changes shape.** TP changes summation order, so the 48/48
   greedy gate against HF does not survive and must not be forced to. The new
   gate is: TP2 vs TP1 logits within bf16 noise with the reference token in the
   top-5 at the first divergence (the standard already used); spec greedy ==
   raw greedy *exactly* on both cards; needle at 131k and 262k; and the
   serving snapshots doubled and re-checked.
7. **Memory per card** (256k, fp8 KV, both drafts): 6.83 weights + 4.2 KV +
   2.54 embed + 2.0 draft + ~1 activation = **~16.6 GB of 31.4**. At 8.25 bpw
   (Mode B) weights double to 13.9 and the total is **~23.6 GB** — tight but
   inside, and the compression of the current 29 GB peak to two 16 GB cards is
   itself a win for the serving path.

## The plan

Each phase is independently deliverable and produces one number. M0 is the
gate; M1–M4 only exist if it passes.

**M0 — the interconnect, and the decision (1 week).**
`scripts/tp/check_interconnect.py`: `nvidia-smi topo -m`; `can_device_access_peer`;
a copy-bandwidth matrix 1 KB–256 MB both directions; a hand-written one-shot
all-reduce and NCCL's, at 10 KB and 80 KB, eager and inside a graph, P2P on and
off; and a two-device capture probe. Then `scripts/tp/tp_projection.py`: a
64-layer synthetic chain at the real shard shapes (half-N int4 GEMV + one-shot
collective + fused add/rmsnorm) replayed from graphs, to check that the
component-model projection reproduces. **Deliverable: a table of L and a
projected raw number. Kill: L > 12 µs, or the two-device capture probe fails
and the one-shot kernel cannot be made to rendezvous — then stop and do Mode B
instead. Cost: one day of engineering, four days of waiting for a box.**

**M1 — raw decode on two cards (1–2 weeks).** Shard the checkpoint offline
(`scripts/tp/shard.py`, a rank-0/rank-1 directory in the same packing, embed
replicated, lm_head vocab-split); a `Shard` that owns one card's weights, state
and graph; the collective fused into `add_rmsnorm`; both graphs replayed per
step. Milestone: coherent greedy text, ≥8k tokens, TP2-vs-TP1 logit agreement
at every layer boundary, and **one raw number**. Projection 145–160; below 135
means the fixed tax is worse than modelled.

**M2 — the speculative step (1–2 weeks).** M-row Marlin at half-N; the draft's
five feature layers either sharded (plus its own collectives) or whole on rank
1; the split-vocabulary accept/commit; the DFlash ring cache per card.
Milestone: spec greedy == raw greedy exact on both, and one number.
Projection 330–360 prose. **This phase is allowed to be negative for the
thesis and is worth running anyway**: if the draft does not shard, its 2.6 ms
caps the whole speculative row at ~1.4x, and that is the most interesting
negative result in the plan.

**M3 — context and prefill (1 week).** 256k KV split across the two cards,
needle at 131k/262k, the 200k decode row, and the prefill row with chunk 4096
and chunk 1024. Projection: 200k raw 105–120; 256k prefill ~120 s.

**M4 — the rivals, same box, same day (1 week).** vLLM TP=2, SGLang TP=2,
llama.cpp `--split-mode row`, at bs=1, on the same prompts, with the bytes
counted from their own headers (`docs/baselines.md` method). This is the
headline measurement and it cannot be quoted from anyone else's numbers.
Expectation, not yet measured here: **1.2–1.5x**. The claim is only real if we
clear it.

**M5 — the quality mode (1–2 weeks, optional).** int8 g128 packing + GEMV +
GPTQ at 8 bits, run TP=2 at 8.25 bpw: ~94 raw, KL ≤0.005, the quality row
closed. No interconnect dependency.

## Targets

| | 1 card (measured) | 2 cards (projected, L = 4–9 µs) | ratio |
|---|---|---|---|
| raw decode, short context | 100.9 (81.0% of wall) | **145–160** (58–64% of the aggregate wall) | 1.45–1.6x |
| speculative, prose / code / math | 228 / 357 / 378 | **330–360 / 510–570 / 540–600** | 1.45–1.6x |
| raw decode at 200k | 71.6 (85.5%) | **105–120** | 1.5–1.7x |
| 256k prefill | 224 s (1160 tok/s) | **~120 s** | 1.8–1.9x |
| all-reduce, 128 collectives | — | **≤0.8 ms** (12% of the step) | — |
| L, 10 KB, in-graph | — | **≤8 µs** | — |
| vs vLLM TP=2 at bs=1 | — | **+40% or more** | — |
| KL at equal latency (Mode B) | 0.0232 at 4.25 bpw | **≤0.005** at 8.25 bpw | — |

The two rows that decide the report are the roofline (62% is the honest
number, and it is a *fall*) and the rival row (the engine tax under TP, which
is the only thing that is ours).

## What would falsify this

- **L above 12 µs without P2P.** 128 x 15 µs = 1.9 ms on a 6 ms step; the
  second card is then worth 1.3x and Mode B is the better project.
- **The collective cannot live in the graph.** If each of the 128 collectives
  costs a host launch, the 9.1 ms launch tax returns and the plan is dead on
  arrival — hence the device-side one-shot kernel as the primary design, not
  the fallback.
- **The GDN chain does not shard and the half-N GEMV loses 5 points.** Both
  are inside the model's `1.56`; together they take another 10% off.
- **The draft does not shard.** Already priced in: the speculative row then
  gains *less* than raw, and the headline stays on raw.
- **A rival already gets 1.5x from a second card at bs=1.** Then the engine
  margin is small and the honest report is "TP=2 is a 1.5x latency purchase;
  here is the accounting", which is still a result but a smaller one. Measure
  it before writing anything else.
- **x8/x8 slots.** Decode barely notices; prefill loses its 1.8x and lands near
  1.5x. Check `nvidia-smi topo -m` and the link width in the first hour.

## Prior art

Pointers, not evidence — found by search on 2026-10-25 and **not read in full
here**; their numbers are not reproduced and must be re-measured on our box.

- `local-inference-lab/rtx6kpro` — a 6 x 5090 box, PCIe all-reduce notes and
  the P2P driver configuration: <https://github.com/local-inference-lab/rtx6kpro>
- `noonghunna/club-3090` — 2 x 3090 dual-stream community project with a custom
  all-reduce: <https://github.com/noonghunna/club-3090>
- SGLang PR #32479 — `[SM120]` optional PCIe all-reduce, one-shot + fused
  RMSNorm + FP8 DMA ring: <https://github.com/sgl-project/sglang/pull/32479>
- vLLM PR #50771 — fast driver P2P check on compute capability 12.x:
  <https://github.com/vllm-project/vllm/pull/50771>
- `informatico-madrid/blackwell-linux-infra-optimizer` — 5090 deployment notes
  that name P2P deadlocks as the failure mode:
  <https://github.com/informatico-madrid/blackwell-linux-infra-optimizer>
- NCCL, *Using NCCL with CUDA Graphs*:
  <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/cudagraph.html>

The SGLang and vLLM pull requests are the ones that date the plan: the big
engines are moving onto `sm_120` PCIe collectives now, so the "nobody tunes
this" window is closing and the report must carry a date.

## Open decisions

1. **Two cards, not four.** The model says 4 is worth another 1.2x at L=12 µs
   and 8 is worth nothing; the architecture (4 KV heads) caps TP at 4 anyway.
   Four is a different box, a different PSU and a different bill; revisit only
   if M0 measures L(4) ≤ 12 µs.
2. **Host-staged collective as the baseline, P2P as the upside.** A patched
   driver cannot be a dependency of a reproducible report.
3. **Headline: Mode A or Mode B?** A is the sequel to the thesis; B closes the
   one row the project conceded. They are different products and the data will
   choose — but the choice should be made *after* M0, not before.
4. **The box.** The author's 2 x 5090 for development; one rented 2-GPU instance
   for the report, one sitting, every rival the same day — the Phase 4 protocol,
   unchanged.
5. **What is out of scope on purpose:** NVLINK, multi-node, RDMA, models that do
   not fit two cards, serving traffic, and any claim of 2x.
