#!/usr/bin/env python3
"""S3: one [N,K] layer, four splits, M sweep.

single: one card, full layer, no collective.
split_n: two cards, half of N each, then a SUM all-reduce of the [M, N/2]
         output. The combine of a split-N layer is a concatenation, so the
         collective is timed to price the synchronization a following
         column-parallel layer would need; its arithmetic is not used.
split_k: two cards, half of K each (contiguous copies), exact partial sums
         combined by a SUM all-reduce.
single_split_k: one card computes both K halves locally: the cost of splitting
         without a collective.

bf16 only. int4 is not tested: no int4 GEMV kernel is installed here.
"""
import argparse
import ctypes
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
N, K = 8704, 5120
MS = (1, 8, 64, 512, 4096)
VARIANTS = ("single", "split_n", "split_k", "single_split_k")
REPS = {1: 200, 8: 100, 64: 50, 512: 5, 4096: 1}


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


def append(path, row):
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


def worker(rank, args, uid):
    import torch
    import torch.distributed as dist
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    out = Path(args.output)
    dist.init_process_group("nccl", init_method=(out / "rendezvous").as_uri(), rank=rank,
                            world_size=2, timeout=timedelta(seconds=90),
                            device_id=torch.device(f"cuda:{rank}"))
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    m = args.m
    variant = args.variant
    half_n, half_k = N // 2, K // 2
    # Rotate over eight matrices so the 89 MB operand is not served from L2:
    # a real layer stack streams different weights every layer. One shared
    # matrix would fit in L2 and overstate single-card bandwidth.
    wcount = 8
    gen = torch.Generator(device=f"cuda:{rank}").manual_seed(20261027)
    weights = []
    for _ in range(wcount):
        w = torch.empty(N, K, device=f"cuda:{rank}", dtype=torch.bfloat16)
        w.normal_(mean=0.0, std=0.02, generator=gen)
        weights.append(w)
    x = torch.empty(m, K, device=f"cuda:{rank}", dtype=torch.bfloat16)
    x.normal_(mean=0.0, std=0.5, generator=gen)
    native = NativeNccl(uid, rank)
    active = variant != "single" or rank == 0
    payload = 0

    if variant == "split_n":
        operands = [w[rank * half_n:(rank + 1) * half_n].contiguous() for w in weights]
        payload = m * half_n * 2
        step = lambda i: x @ operands[i % wcount].T
        combine = lambda y: native.all_reduce(y, stream)
        ref_slice = lambda ref: ref[:, rank * half_n:(rank + 1) * half_n]
    elif variant == "split_k":
        operands = [w[:, rank * half_k:(rank + 1) * half_k].contiguous() for w in weights]
        xk = x[:, rank * half_k:(rank + 1) * half_k].contiguous()
        payload = m * N * 2
        step = lambda i: xk @ operands[i % wcount].T
        combine = lambda y: native.all_reduce(y, stream)
        ref_slice = lambda ref: ref
    elif variant == "single_split_k":
        operands = [[w[:, i * half_k:(i + 1) * half_k].contiguous() for i in range(2)] for w in weights]
        xk = [x[:, i * half_k:(i + 1) * half_k].contiguous() for i in range(2)]
        step = lambda i: xk[0] @ operands[i % wcount][0].T + xk[1] @ operands[i % wcount][1].T
        combine = lambda y: None
        ref_slice = lambda ref: ref
    else:
        operands = weights
        step = lambda i: x @ operands[i % wcount].T
        combine = lambda y: None
        ref_slice = lambda ref: ref

    reference = x @ weights[0].T
    if active:
        # split-N: the combine is a concatenation, so the card's own half is
        # compared before any collective touches it. split-K variants are only
        # correct after their combine, so they are compared after it.
        y = step(0)
        if variant in ("split_k", "single_split_k"):
            combine(y)
            torch.cuda.synchronize()
            ref = reference
        else:
            ref = ref_slice(reference)
        max_diff = (y - ref).abs().max().item()
        ok = bool(torch.allclose(y, ref, rtol=1e-2, atol=1e-2))
        exact = bool(torch.equal(y, ref))
        assert ok, ("correctness", variant, m, max_diff)
        assert bool(torch.isfinite(y).all().item())
        del y
    else:
        max_diff, exact, ok = 0.0, True, True

    reps = REPS[m]
    for r in range(5):
        if active:
            y = step(r)
            combine(y)
    torch.cuda.synchronize()
    dist.barrier()
    torch.cuda.synchronize()
    samples = []
    for _ in range(7):
        dist.barrier()
        torch.cuda.synchronize()
        if active:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for r in range(reps):
                y = step(r)
                combine(y)
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) * 1000 / reps)
        else:
            samples.append(float("nan"))
    if active:
        assert all(v > 0 for v in samples)
    row = {"m": m, "n": N, "k": K, "variant": variant, "rank": rank,
           "active": active, "dtype": "bfloat16", "reps": reps, "samples": 7,
           "payload_bytes": payload, "us_samples": samples,
           "us_median": statistics.median(samples) if active else None,
           "correct": ok, "exact_equal": exact, "max_abs_diff": max_diff,
           "weight_bytes_per_card": (sum(o.numel() for o in operands) * 2
                                     if active and variant != "single_split_k"
                                     else (sum(o.numel() for pair in operands for o in pair) * 2 if active else 0)),
           "weight_matrices": wcount,
           "collective": combined_name(variant)}
    append(out / f"rank{rank}.jsonl", row)
    if rank == 0:
        print(json.dumps({k: row[k] for k in ("m", "variant", "us_median", "payload_bytes", "exact_equal")}), flush=True)
    del weights, x, reference
    torch.cuda.synchronize()
    dist.barrier()
    native.close()
    dist.destroy_process_group()


def combined_name(variant):
    return {"single": "none", "split_n": "sum_all_reduce", "split_k": "sum_all_reduce",
            "single_split_k": "local_add"}[variant]


def run(args):
    import torch
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location("s1", ROOT / "lab/01_link.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    meta = helper.metadata(args)
    meta.update(stage="S3 linear splits", script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                metadata_helper_sha256=hashlib.sha256((ROOT / "lab/01_link.py").read_bytes()).hexdigest(),
                layer={"n": N, "k": K, "dtype": "bfloat16"}, ms=list(MS),
                started_utc=datetime.now(timezone.utc).isoformat())
    (out / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    jobs = [(m, variant) for m in MS for variant in VARIANTS]
    if args.reverse:
        jobs.reverse()
    for m, variant in jobs:
        dest = out / f"m{m}-{variant}"
        dest.mkdir()
        cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", "--m", str(m),
               "--variant", variant, "--output", str(dest)]
        env = os.environ.copy()
        env.update(NCCL_DEBUG="WARN", OMP_NUM_THREADS="1")
        print("START", m, variant, flush=True)
        begun = time.monotonic()
        with (dest / "stdout.txt").open("w") as stdout, (dest / "stderr.txt").open("w") as stderr:
            child = subprocess.Popen(cmd, stdout=stdout, stderr=stderr, env=env, start_new_session=True)
            timeout = False
            try:
                child.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                timeout = True
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
        status = {"m": m, "variant": variant, "returncode": child.returncode, "timed_out": timeout,
                  "seconds": time.monotonic() - begun, "argv": cmd}
        append(out / "status.jsonl", status)
        print(json.dumps(status), flush=True)
    (out / "finished.json").write_text(json.dumps({"finished_utc": datetime.now(timezone.utc).isoformat()}) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--timeout", type=int, default=240)
    p.add_argument("--reverse", action="store_true")
    p.add_argument("--worker", action="store_true")
    p.add_argument("--m", type=int, choices=MS)
    p.add_argument("--variant", choices=VARIANTS)
    args = p.parse_args()
    if args.worker:
        import torch
        import torch.multiprocessing as mp
        uid = torch.cuda.nccl.unique_id()
        mp.spawn(worker, args=(args, uid), nprocs=2, join=True)
    else:
        run(args)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
