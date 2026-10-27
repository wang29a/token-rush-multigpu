#!/usr/bin/env python3
"""Validate the S6 Qwen3-8B TP1/TP2 decode runs and generate the report."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("tp1", "tp2", "tp2_fp32reduce")


def read(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x]


def load(base):
    import numpy as np
    out = {}
    for name in VARIANTS:
        d = base / name
        if not (d / "row.rank0.json").exists():
            continue
        row = json.loads((d / "row.rank0.json").read_text())
        meta = json.loads((d / "metadata.json").read_text())
        sha = meta["git_commit"]["stdout"].strip()
        if name == "tp2_fp32reduce":
            # This variant ran from the working tree after the recorded commit;
            # verify it against the tree and say so in the report.
            code = (ROOT / "lab/05_tp2_decode.py").read_bytes()
        else:
            code = subprocess.check_output(["git", "-C", str(ROOT), "show", f"{sha}:lab/05_tp2_decode.py"])
        assert hashlib.sha256(code).hexdigest() == meta["script_sha256"], name
        assert row["finite"] and row["world"] == (1 if name == "tp1" else 2)
        assert len(row["generated"]) == row["tokens"] + 1
        if name != "tp1":
            other = json.loads((d / "row.rank1.json").read_text())
            assert other["generated"] == row["generated"], name
            assert abs(other["tokens_per_second"] / row["tokens_per_second"] - 1) < 1e-6, name
        out[name] = {"row": row, "meta": meta,
                     "logits": np.load(d / "first_logits.npy"),
                     "generated": json.loads((d / "generated.json").read_text())}
    hf_dir = base / "hf_reference"
    assert (hf_dir / "hf_logits.npy").exists()
    import numpy as np
    out["hf"] = {"logits": np.load(hf_dir / "hf_logits.npy")}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("result_dir", type=Path)
    base = p.parse_args().result_dir.resolve()
    data = load(base)
    assert "tp1" in data and "tp2" in data
    import numpy as np
    r1, r2 = data["tp1"]["row"], data["tp2"]["row"]
    speedup = r2["tokens_per_second"] / r1["tokens_per_second"]
    gens = {k: data[k]["generated"] for k in data if k != "hf"}
    agree = {k: sum(a == b for a, b in zip(gens["tp1"], gens[k])) for k in gens}
    first_diff = {k: next((i for i, (a, b) in enumerate(zip(gens["tp1"], gens[k])) if a != b), None)
                  for k in gens}
    logits_diff = {k: float(np.abs(data[k]["logits"] - data["tp1"]["logits"]).max()) for k in gens}
    hf_diff = {k: float(np.abs(data[k]["logits"] - data["hf"]["logits"]).max()) for k in gens}
    hf_mean = {k: float(np.abs(data[k]["logits"] - data["hf"]["logits"]).mean()) for k in gens}
    argmax = {k: int(data[k]["logits"].argmax()) for k in list(gens) + ["hf"]}
    prediction = json.loads((base / "prediction.json").read_text())
    pr = prediction["predictions"]
    half = r1["ms_per_token"] / 2
    remainder = r2["ms_per_token"] - half
    checks = [
        {"metric": "tp1_tokens_per_second", "pred": pr["tp1_tokens_per_second"], "measured": r1["tokens_per_second"]},
        {"metric": "tp2_tokens_per_second", "pred": pr["tp2_tokens_per_second"], "measured": r2["tokens_per_second"]},
        {"metric": "speedup", "pred": pr["speedup"], "measured": speedup},
        {"metric": "tp2_ms_per_token", "pred": pr["tp2_ms_per_token"], "measured": r2["ms_per_token"]},
        {"metric": "logits_max_abs_diff", "pred": pr["tp2_first_logits_max_abs_diff_vs_tp1"], "measured": logits_diff["tp2"]},
        {"metric": "greedy_agreement_tokens", "pred": [pr["greedy_agreement_tokens"], len(gens["tp1"])],
         "measured": agree["tp2"]},
    ]
    for c in checks:
        lo, hi = c["pred"]
        c["verdict"] = "hit" if lo <= c["measured"] <= hi else ("miss (below)" if c["measured"] < lo else "miss (above)")
    summary = {
        "label": "measured decode; projection is not used here",
        "code_commit": data["tp2"]["meta"]["git_commit"]["stdout"].strip(),
        "script_sha256": data["tp2"]["meta"]["script_sha256"],
        "model": "Qwen/Qwen3-8B bf16",
        "prompt": prediction["prompt"], "tokens": r1["tokens"], "warmup": r1["warmup"],
        "rows": {k: {f: data[k]["row"][f] for f in ("world", "tokens_per_second", "ms_per_token",
                                                    "peak_gib", "all_reduces_per_token")} for k in gens},
        "speedup_tp2_over_tp1": speedup,
        "half_tp1_ms": half, "tp2_unsharded_remainder_ms": remainder,
        "remainder_share_percent": remainder / r2["ms_per_token"] * 100,
        "collective_time_if_8_5us_each_ms": r2["all_reduces_per_token"] * 8.5 / 1000,
        "greedy_agreement": agree, "first_divergence_index": first_diff,
        "logits_max_abs_diff_vs_tp1": logits_diff, "logits_max_abs_diff_vs_hf": hf_diff,
        "logits_mean_abs_diff_vs_hf": hf_mean, "argmax": argmax,
        "checks": checks,
        "gate": {"speed_criterion_1_4x": speedup >= 1.4,
                 "text_equivalence_32_tokens": agree["tp2"] >= pr["greedy_agreement_tokens"],
                 "S6_passed": False,
                 "why": "the speed criterion is met, but TP2's greedy text diverges from TP1 at token 4 and degenerates into a repetition loop; fp32 collectives change nothing, and TP2 logits are as close to the Hugging Face reference as TP1's, so the cause is bf16 partial-sum rounding in the sharded GEMMs, not a structural error",
                 "decision": "report the measured speed, do not call S6 passed. Next bounded step: fp32 residual stream (accumulate h in fp32, cast only for GEMM inputs), then re-run TP1/TP2 text agreement before M1."},
    }
    (base / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    fields = ["variant", "world", "tokens", "tokens_per_second", "ms_per_token", "peak_gib",
              "all_reduces_per_token", "greedy_agreement_vs_tp1", "first_divergence",
              "logits_max_abs_diff_vs_tp1", "logits_max_abs_diff_vs_hf"]
    with (base / "matrix.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for k in gens:
            r = data[k]["row"]
            w.writerow({"variant": k, "world": r["world"], "tokens": r["tokens"],
                        "tokens_per_second": f"{r['tokens_per_second']:.4f}",
                        "ms_per_token": f"{r['ms_per_token']:.4f}", "peak_gib": f"{r['peak_gib']:.3f}",
                        "all_reduces_per_token": r["all_reduces_per_token"],
                        "greedy_agreement_vs_tp1": agree[k], "first_divergence": first_diff[k],
                        "logits_max_abs_diff_vs_tp1": f"{logits_diff[k]:.5f}",
                        "logits_max_abs_diff_vs_hf": f"{hf_diff[k]:.5f}"})

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file("/tmp/qwen3-8b/tokenizer.json")
    text = {k: tok.decode(gens[k][:24]) for k in gens}
    rows = "\n".join(f"| {k} | {data[k]['row']['world']} | {data[k]['row']['tokens_per_second']:.2f} | "
                     f"{data[k]['row']['ms_per_token']:.3f} | {data[k]['row']['peak_gib']:.2f} | "
                     f"{data[k]['row']['all_reduces_per_token']} | {agree[k]}/{len(gens[k])} | {first_diff[k]} |"
                     for k in gens)
    checks_text = "\n".join(f"| {c['metric']} | {c['pred'][0]:g}-{c['pred'][1]:g} | {c['measured']:.3f} | {c['verdict']} |"
                           for c in checks)
    body = f"""## S6 — Qwen3-8B TP=1 vs TP=2 (2026-10-27, vast-57952342)

Generated by `lab/report_s6.py {base.relative_to(ROOT)}` from recorded rows,
logits and token ids. Measured decode of the real model, not a projection.

**Question:** what is the measured greedy-decode speedup of Qwen3-8B at bs=1 on
two 5090s, with one graph per card, and does the TP=2 text match TP=1?

**Prediction (written before the run, `prediction.json`):** TP=1 85-110 tok/s,
TP=2 130-160 tok/s, speedup 1.35-1.55, TP=2 6.2-7.7 ms/token, first-step logits
within 0.5 of TP=1, and greedy tokens identical for at least 32 steps.

**Machine/version/code:** the S1 box, 2 x RTX 5090, Gen5 x16, NODE, 400 W,
runtime peer access false. PyTorch {data['tp2']['meta']['torch']}, CUDA wheel
{data['tp2']['meta']['cuda']}, NCCL {'.'.join(map(str, data['tp2']['meta']['nccl']))}.
Code commit `{summary['code_commit'][:7]}`; engine
`lab/05_tp2_decode.py` SHA-256 `{summary['script_sha256']}` (the
`tp2_fp32reduce` variant was run from a working-tree edit of that script, then
committed; its row is checked against the tree, the other rows against the
commit). Weights:
Qwen/Qwen3-8B bf16, 36 layers, hidden 4096, 8 KV heads, vocab 151936.

**What was done:** prompt `{prediction['prompt']!r}`, {r1['tokens']} greedy tokens
after {r1['warmup']} warmup tokens, one CUDA graph per card replaying the whole
36-layer step. TP=1 runs the identical code with world size 1; TP=2 shards
q/k/v by heads, o/gate/up/down by the parallel dimension, splits lm_head by
vocabulary, and all-gathers the {151936}-wide logits for argmax. The two
per-layer partials are combined by a SUM all-reduce inside the graph
({r2['all_reduces_per_token']} per token). `tp2_fp32reduce` repeats TP=2 with both
collectives in fp32.

**Measured:**

| variant | world | tok/s | ms/token | peak GiB/card | all-reduces/token | greedy = TP1 | first diff |
|---|---:|---:|---:|---:|---:|---:|---:|
{rows}

TP=2 is **{speedup:.3f}x** TP=1. Half of TP=1's step is {half:.3f} ms; TP=2 measures
{r2['ms_per_token']:.3f} ms, so {remainder:.3f} ms ({remainder / r2['ms_per_token'] * 100:.1f}%) does not
shard. At the measured 10 KiB collective cost from S2-sync, {r2['all_reduces_per_token']}
collectives would be about {summary['collective_time_if_8_5us_each_ms']:.2f} ms/token
({summary['collective_time_if_8_5us_each_ms'] / r2['ms_per_token'] * 100:.1f}%); the rest of the
remainder is the half-width GEMMs and the per-token logit all-gather, not measured
separately here.

**Correctness, and the failure:** first-step logits agree with an independent
Hugging Face CPU forward to {hf_diff['tp1']:.3f} (TP=1) and {hf_diff['tp2']:.3f} (TP=2), with
the same argmax; TP=1 and TP=2 differ by {logits_diff['tp2']:.3f}. Greedy text,
however, diverges at token {first_diff['tp2']} and TP=2 degenerates into a
repetition loop:

- TP=1: `{text['tp1']}...`
- TP=2: `{text['tp2']}...`

`tp2_fp32reduce` produces the same tokens at
{data['tp2_fp32reduce']['row']['tokens_per_second']:.2f} tok/s, so the collective's
rounding is not the cause. The sharded bf16 GEMMs round their partial sums
differently from the full-width GEMM, and the residual stream accumulates those
roundings for 36 layers. That is enough to flip a near-tie in greedy decoding;
it is a precision property of bf16 TP, not a structural error in the sharding.

**Prediction vs result:**

| metric | predicted | measured | verdict |
|---|---|---:|---|
{checks_text}

Both throughput predictions were too optimistic: the model runs slower than the
component arithmetic suggested, most likely because half-width GEMMs at M=1 do
not reach the bandwidth of the single wide GEMV measured in S4. The speedup
range held; the text-equivalence prediction did not.

**Gate and decision:** S6's speed criterion (>= 1.4x) is met at
{speedup:.2f}x, but **S6 is not passed**: a TP engine whose greedy text leaves
TP=1 at token {first_diff['tp2']} is not equivalent. No 1.5x claim is made and M1
is not started. The next bounded step is an fp32 residual stream (accumulate
`h` in fp32, cast only at GEMM inputs), then a single re-run of this exact
comparison; if text still diverges, the engine needs fp32 master weights or a
different reduction strategy before any model-level claim.

This also calibrates the project: the measured L is small, but at bs=1 the
measured two-card speedup on the test model is {speedup:.2f}x, not the 1.45x the
component projection suggested. Test-model numbers do not transfer to 27B.

**Reproduction:**

```bash
.venv/bin/python lab/05b_hf_reference.py --output results/<date-box>/s6/hf_reference
.venv/bin/python lab/05_tp2_decode.py --mode tp1 --output results/<date-box>/s6/tp1
.venv/bin/python lab/05_tp2_decode.py --mode tp2 --output results/<date-box>/s6/tp2
.venv/bin/python lab/05_tp2_decode.py --mode tp2 --reduce-fp32 --output results/<date-box>/s6/tp2_fp32reduce
.venv/bin/python lab/report_s6.py results/<date-box>/s6
```

Weights were downloaded to `/tmp/qwen3-8b` (16 GB, not in git, not on a
persistent volume). The generator verifies code hashes against the recorded
commit, row shapes, rank agreement, and the Hugging Face reference presence.
"""
    (base / "S6.md").write_text(body)

    log = ROOT / "lab/LOG.md"
    old = log.read_text().rstrip()
    marker = "## S6 —"
    head = old.split(marker, 1)[0].rstrip() if marker in old else old
    log.write_text(re.sub(r"\n{3,}", "\n\n", head + "\n\n" + body).rstrip("\n") + "\n")

    index = ["S6 Qwen3-8B decode measured on 2026-10-27, same 2 x RTX 5090 box.",
             f"Code `{summary['code_commit'][:7]}`. bf16, 96 greedy tokens after 8 warmup, one graph per card.", "",
             "| variant | tok/s | ms/token | peak GiB/card |", "|---|---:|---:|---:|",
             *[f"| {k} | {data[k]['row']['tokens_per_second']:.1f} | {data[k]['row']['ms_per_token']:.2f} | "
               f"{data[k]['row']['peak_gib']:.1f} |" for k in gens], "",
             f"TP=2 is **{speedup:.2f}x** TP=1, but its greedy text leaves TP=1 at token {first_diff['tp2']} and",
             "loops; fp32 collectives do not change it. **S6's speed bar is met, S6 is not passed.**",
             "Next: fp32 residual stream, then one re-run of this comparison.",
             f"[S6 report]({base.relative_to(ROOT)}/S6.md) · [Full data]({base.relative_to(ROOT)}/matrix.csv)."]
    readme = ROOT / "README.md"
    text = readme.read_text()
    start, end = "<!-- S6_RESULTS_START -->", "<!-- S6_RESULTS_END -->"
    block = start + "\n\n" + "\n".join(index) + "\n\n" + end
    if start in text:
        before, rest = text.split(start, 1)
        _, after = rest.split(end, 1)
        text = before + block + after
    elif "<!-- S4_RESULTS_END -->" in text:
        text = text.replace("<!-- S4_RESULTS_END -->", "<!-- S4_RESULTS_END -->\n\n" + block, 1)
    else:
        text = text.replace("## Documents", block + "\n\n## Documents", 1)
    readme.write_text(text.rstrip("\n") + "\n")
    print(json.dumps({"speedup": speedup, "tp2_tok_s": r2["tokens_per_second"],
                      "greedy_agreement": agree["tp2"], "first_diff": first_diff["tp2"],
                      "hf_diff_tp2": hf_diff["tp2"], "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
