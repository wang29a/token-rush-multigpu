#!/usr/bin/env python3
"""S6: Qwen3-8B greedy decode at bs=1, TP=1 and TP=2, one graph per card.

Not a serving engine: one prompt, greedy tokens, decode latency only. The TP=1
path is the same code with world size 1, so the comparison is same-box and
same-day by construction.

Weight sharding on load:
  q_proj        rows by Q head        (16 heads/rank; 4 Q heads per KV head)
  k_proj/v_proj rows by KV head       (4 KV heads/rank)
  o_proj        columns (row-parallel) + SUM all-reduce
  gate/up_proj  rows (column-parallel)
  down_proj     columns (row-parallel) + SUM all-reduce
  lm_head       rows by vocabulary    (75968/rank); logits all-gathered for argmax
  embed_tokens  replicated: the lookup reads one row, not the table

The KV cache is a static [max_len, kv_heads, 128] buffer per layer, and
attention runs over the full static length with an additive mask, so the
captured graph has fixed shapes. Token and position are device tensors updated
between replays; nothing in the timed loop reads back to the host.
"""
import argparse
import ctypes
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
MODEL = Path("/tmp/qwen3-8b")
HIDDEN, INTER, LAYERS = 4096, 12288, 36
HEADS, KV_HEADS, HEAD_DIM = 32, 8, 128
VOCAB, THETA, EPS = 151936, 1_000_000.0, 1e-6


class NativeNccl:
    class UniqueId(ctypes.Structure):
        _fields_ = [("internal", ctypes.c_char * 128)]

    def __init__(self, uid, rank):
        libraries = {line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                     if '/libnccl.so' in line}
        assert len(libraries) == 1, libraries
        self.lib = ctypes.CDLL(next(iter(libraries)))
        self.lib.ncclGetErrorString.argtypes = [ctypes.c_int]
        self.lib.ncclGetErrorString.restype = ctypes.c_char_p
        self.lib.ncclCommInitRank.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int,
                                              self.UniqueId, ctypes.c_int]
        self.lib.ncclAllReduce.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                                           ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
        self.lib.ncclCommDestroy.argtypes = [ctypes.c_void_p]
        for name in ("ncclCommInitRank", "ncclAllReduce", "ncclCommDestroy"):
            getattr(self.lib, name).restype = ctypes.c_int
        self.comm = ctypes.c_void_p()
        assert len(uid) == 128
        self.check(self.lib.ncclCommInitRank(ctypes.byref(self.comm), 2,
                                             self.UniqueId.from_buffer_copy(uid), rank))

    def check(self, code):
        if code:
            raise RuntimeError(self.lib.ncclGetErrorString(code).decode())

    def all_reduce(self, tensor, stream):
        self.check(self.lib.ncclAllReduce(tensor.data_ptr(), tensor.data_ptr(), tensor.numel(),
                                         9, 0, self.comm, stream.cuda_stream))

    def close(self):
        self.check(self.lib.ncclCommDestroy(self.comm))


class Shard:
    """One rank's slice of the HF checkpoint, read straight from safetensors."""

    def __init__(self, path, rank, world, device):
        import torch
        self.torch = torch
        self.device = device
        self.path = Path(path)
        self.index = json.loads((self.path / "model.safetensors.index.json").read_text())["weight_map"]
        self.rank, self.world = rank, world
        self.hq = HEADS // world
        self.hkv = KV_HEADS // world
        self.q_rows = (rank * self.hq * HEAD_DIM, (rank + 1) * self.hq * HEAD_DIM)
        self.kv_rows = (rank * self.hkv * HEAD_DIM, (rank + 1) * self.hkv * HEAD_DIM)
        self.o_cols = (rank * HIDDEN // world, (rank + 1) * HIDDEN // world)
        self.inter_rows = (rank * INTER // world, (rank + 1) * INTER // world)
        self.vocab_rows = (rank * (VOCAB // world), (rank + 1) * (VOCAB // world))

    def _full(self, name):
        from safetensors import safe_open
        with safe_open(str(self.path / self.index[name]), framework="pt", device="cpu") as f:
            return f.get_tensor(name)

    def get(self, name, rows=None, cols=None):
        t = self._full(name)
        if rows is not None:
            t = t[rows[0]:rows[1]]
        if cols is not None:
            t = t[:, cols[0]:cols[1]]
        return t.to(self.device, dtype=self.torch.bfloat16).contiguous()

    def load(self):
        w = {"hq": self.hq, "hkv": self.hkv, "vocab_rows": self.vocab_rows}
        w["embed"] = self.get("model.embed_tokens.weight")
        w["norm"] = self.get("model.norm.weight")
        w["lm_head"] = self.get("lm_head.weight", rows=self.vocab_rows)
        for i in range(LAYERS):
            p = f"model.layers.{i}."
            w[("ln1", i)] = self.get(p + "input_layernorm.weight")
            w[("ln2", i)] = self.get(p + "post_attention_layernorm.weight")
            w[("q", i)] = self.get(p + "self_attn.q_proj.weight", rows=self.q_rows)
            w[("k", i)] = self.get(p + "self_attn.k_proj.weight", rows=self.kv_rows)
            w[("v", i)] = self.get(p + "self_attn.v_proj.weight", rows=self.kv_rows)
            w[("o", i)] = self.get(p + "self_attn.o_proj.weight", cols=self.o_cols)
            w[("gate", i)] = self.get(p + "mlp.gate_proj.weight", rows=self.inter_rows)
            w[("up", i)] = self.get(p + "mlp.up_proj.weight", rows=self.inter_rows)
            w[("down", i)] = self.get(p + "mlp.down_proj.weight", cols=self.inter_rows)
        return w


def build(rank, world, device, max_len, native, stream):
    import torch
    w = Shard(MODEL, rank, world, device).load()
    idx = torch.arange(max_len, device=device, dtype=torch.float32)
    inv = 1.0 / (THETA ** (torch.arange(0, HEAD_DIM, 2, device=device, dtype=torch.float32) / HEAD_DIM))
    freqs = torch.outer(idx, inv)
    emb = torch.cat([freqs, freqs], dim=1).contiguous()
    cos = emb.cos().to(torch.bfloat16).contiguous()
    sin = emb.sin().to(torch.bfloat16).contiguous()
    ar = torch.arange(max_len, device=device)
    mask_all = torch.full((max_len, max_len), -30000.0, device=device, dtype=torch.bfloat16)
    mask_all[ar[:, None] >= ar[None, :]] = 0.0
    k_cache = [torch.zeros(max_len, w["hkv"], HEAD_DIM, device=device, dtype=torch.bfloat16)
               for _ in range(LAYERS)]
    v_cache = [torch.zeros_like(k_cache[0]) for _ in range(LAYERS)]
    token = torch.zeros(1, device=device, dtype=torch.int64)
    pos = torch.zeros(1, device=device, dtype=torch.int64)
    mask_row = torch.zeros(1, 1, 1, max_len, device=device, dtype=torch.bfloat16)
    hq, hkv = w["hq"], w["hkv"]
    scale = HEAD_DIM ** -0.5

    def rmsnorm(x, weight):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + EPS) * weight

    def rope(t, c, s):
        x1, x2 = t[..., :HEAD_DIM // 2], t[..., HEAD_DIM // 2:]
        return t * c + torch.cat([-x2, x1], dim=-1) * s

    def step():
        h = w["embed"].index_select(0, token)
        c = cos.index_select(0, pos).view(1, 1, HEAD_DIM)
        s = sin.index_select(0, pos).view(1, 1, HEAD_DIM)
        for i in range(LAYERS):
            n1 = rmsnorm(h, w[("ln1", i)])
            q = rope((n1 @ w[("q", i)].T).view(1, hq, HEAD_DIM), c, s)
            k = rope((n1 @ w[("k", i)].T).view(1, hkv, HEAD_DIM), c, s)
            v = (n1 @ w[("v", i)].T).view(1, hkv, HEAD_DIM)
            k_cache[i].index_copy_(0, pos, k)
            v_cache[i].index_copy_(0, pos, v)
            q4 = q.view(1, hkv, hq // hkv, HEAD_DIM)
            scores = torch.einsum("bghd,lgd->bghl", q4, k_cache[i]) * scale + mask_row
            probs = torch.softmax(scores, dim=-1)
            attn = torch.einsum("bghl,lgd->bghd", probs, v_cache[i]).reshape(1, hq * HEAD_DIM)
            o = attn @ w[("o", i)].T
            if native is not None:
                native.all_reduce(o, stream)
            h = h + o
            n2 = rmsnorm(h, w[("ln2", i)])
            g = torch.nn.functional.silu(n2 @ w[("gate", i)].T) * (n2 @ w[("up", i)].T)
            d = g @ w[("down", i)].T
            if native is not None:
                native.all_reduce(d, stream)
            h = h + d
        h = rmsnorm(h, w["norm"])
        return h @ w["lm_head"].T

    return {"step": step, "k_cache": k_cache, "v_cache": v_cache, "token": token,
            "pos": pos, "mask_all": mask_all, "mask_row": mask_row, "logits": None,
            "world": world}


def write_meta(args, prompt_ids, out):
    spec = importlib.util.spec_from_file_location("s1", ROOT / "lab/01_link.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    meta = helper.metadata(args)
    meta.update(stage="S6 Qwen3-8B decode", mode=args.mode,
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                metadata_helper_sha256=hashlib.sha256((ROOT / "lab/01_link.py").read_bytes()).hexdigest(),
                model="Qwen/Qwen3-8B", model_path=str(MODEL), prompt=args.prompt,
                prompt_ids=prompt_ids, started_utc=datetime.now(timezone.utc).isoformat())
    (out / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")


def benchmark(m, args, prompt_ids, native, rank, out, dist=None):
    import torch
    step = m["step"]
    token, pos, mask_row, mask_all = m["token"], m["pos"], m["mask_row"], m["mask_all"]
    device = token.device

    def run_token(tok_id, position):
        token.fill_(tok_id)
        pos.fill_(position)
        mask_row.copy_(mask_all.index_select(0, pos))
        return step()

    with torch.no_grad():
        logits = run_token(prompt_ids[0], 0)
        torch.cuda.synchronize()
        assert bool(torch.isfinite(logits).all().item()), "non-finite logits"
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        run_token(prompt_ids[-1], len(prompt_ids))
        torch.cuda.synchronize()
        with torch.cuda.graph(graph, stream=torch.cuda.current_stream()):
            m["logits"] = step()
        graph.instantiate()
    for c in m["k_cache"] + m["v_cache"]:
        c.zero_()
    torch.cuda.synchronize()
    logits_buf = m["logits"]
    world = m["world"]
    if world == 2:
        gather_buf = torch.empty(world, logits_buf.shape[1], device=device, dtype=logits_buf.dtype)

        def sample():
            dist.all_gather_into_tensor(gather_buf, logits_buf)
            torch.argmax(gather_buf.view(1, -1), dim=-1, keepdim=True, out=tok_buf)
    else:
        def sample():
            torch.argmax(logits_buf, dim=-1, keepdim=True, out=tok_buf)

    with torch.no_grad():
        for p, t in enumerate(prompt_ids):
            token.fill_(t)
            pos.fill_(p)
            mask_row.copy_(mask_all.index_select(0, pos))
            graph.replay()
        torch.cuda.synchronize()
        if world == 2:
            dist.all_gather_into_tensor(gather_buf, logits_buf)
        first = (gather_buf.view(1, -1) if world == 2 else logits_buf).float().cpu()
    first_logits = first.numpy()
    next_token = int(first_logits.argmax(axis=-1)[0])
    tok_buf = torch.tensor([next_token], device=device, dtype=torch.int64)
    start_pos = len(prompt_ids)
    gen_buf = torch.empty(args.tokens, device=device, dtype=torch.int64)
    names = {0: "rank0", 1: "rank1"}

    def decode_one(position):
        token.copy_(tok_buf)
        pos.fill_(position)
        mask_row.copy_(mask_all.index_select(0, pos))
        graph.replay()
        sample()

    with torch.no_grad():
        for k in range(args.warmup):
            decode_one(start_pos + k)
        torch.cuda.synchronize()
        if dist is not None:
            dist.barrier()
            torch.cuda.synchronize()
        begin = time.perf_counter()
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        start_ev.record()
        for k in range(args.tokens):
            decode_one(start_pos + args.warmup + k)
            gen_buf[k].copy_(tok_buf[0])
        end_ev.record()
        end_ev.synchronize()
        elapsed = time.perf_counter() - begin
        gpu_ms = start_ev.elapsed_time(end_ev)
    generated = [int(next_token)] + [int(x) for x in gen_buf.tolist()]
    peak = torch.cuda.max_memory_allocated(device) / 2 ** 30
    row = {"mode": args.mode, "rank": rank, "layers": LAYERS, "hidden": HIDDEN, "vocab": VOCAB,
           "max_len": args.max_len, "tokens": args.tokens, "warmup": args.warmup,
           "elapsed_s": elapsed, "gpu_ms": gpu_ms, "ms_per_token": elapsed * 1000 / args.tokens,
           "tokens_per_second": args.tokens / elapsed, "world": world,
           "all_reduces_per_token": (2 * LAYERS if world == 2 else 0),
           "peak_gib": peak, "finite": True, "generated": generated}
    (out / f"row.{names[rank]}.json").write_text(json.dumps(row, indent=2) + "\n")
    if rank == 0:
        import numpy as np
        np.save(out / "first_logits.npy", first_logits)
        (out / "generated.json").write_text(json.dumps(generated) + "\n")
        print(json.dumps({k: row[k] for k in ("mode", "world", "tokens_per_second",
                                               "ms_per_token", "peak_gib")}), flush=True)
    torch.cuda.synchronize()
    if dist is not None:
        dist.barrier()
    graph.reset()
    del graph


def worker(rank, args, uid, prompt_ids):
    import torch
    import torch.distributed as dist
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    out = Path(args.output)
    dist.init_process_group("nccl", init_method=(out / "rendezvous").as_uri(), rank=rank,
                            world_size=2, timeout=timedelta(seconds=300),
                            device_id=torch.device(f"cuda:{rank}"))
    torch.cuda.set_stream(torch.cuda.Stream())
    native = NativeNccl(uid, rank)
    m = build(rank, 2, f"cuda:{rank}", args.max_len, native, torch.cuda.current_stream())
    benchmark(m, args, prompt_ids, native, rank, out, dist)
    native.close()
    dist.destroy_process_group()


def tp1(args, prompt_ids):
    import torch
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    write_meta(args, prompt_ids, out)
    torch.cuda.set_stream(torch.cuda.Stream())
    m = build(0, 1, "cuda:0", args.max_len, None, torch.cuda.current_stream())
    benchmark(m, args, prompt_ids, None, 0, out, None)
    (out / "finished.json").write_text(json.dumps({"finished_utc": datetime.now(timezone.utc).isoformat()}) + "\n")


def tp2(args, prompt_ids):
    import torch
    import torch.multiprocessing as mp
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    write_meta(args, prompt_ids, out)
    uid = torch.cuda.nccl.unique_id()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    mp.spawn(worker, args=(args, uid, prompt_ids), nprocs=2, join=True)
    (out / "finished.json").write_text(json.dumps({"finished_utc": datetime.now(timezone.utc).isoformat()}) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("tp1", "tp2"), required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--tokens", type=int, default=96)
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--prompt", default="The capital of France is")
    args = p.parse_args()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(MODEL / "tokenizer.json"))
    prompt_ids = tok.encode(args.prompt).ids
    assert 1 <= len(prompt_ids) < args.max_len - args.tokens - args.warmup
    if args.mode == "tp2":
        tp2(args, prompt_ids)
    else:
        tp1(args, prompt_ids)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
