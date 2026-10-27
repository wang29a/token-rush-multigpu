#!/usr/bin/env python3
"""S4/M0 capture probe: 64-layer chain of GEMV + 10 KiB all-reduce + RMSNorm.

Two shapes: `toy` (512 hidden, 512 half-N) for S4's launch-tax question and
`shard` (5120 hidden, 8704 half-N - the production shard shape) for the
component-model reproduction check. bf16 weights, no model, no attention, no
custom kernel: this probe measures structure and additivity, not the
production int4 engine.

Variants: graph_chain, graph_compute, graph_comm, eager_chain.
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
SHAPES = {"toy": (64, 512, 512), "shard": (64, 5120, 8704)}
VARIANTS = ("graph_chain", "graph_compute", "graph_comm", "eager_chain")
EPS = 1e-6


class NativeNccl:
    """Public NCCL C API on the caller's CUDA stream (see lab/02_sync_probe.py)."""

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
        # ncclBfloat16=9, ncclSum=0, from the installed NCCL 2.28 header.
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
                            world_size=2, timeout=timedelta(seconds=60),
                            device_id=torch.device(f"cuda:{rank}"))
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    layers, hidden, n = SHAPES[args.shape]
    gen = torch.Generator(device=f"cuda:{rank}").manual_seed(20261027)
    weights = torch.empty(layers, n, hidden, device=f"cuda:{rank}", dtype=torch.bfloat16)
    weights.normal_(mean=0.0, std=0.02, generator=gen)
    pattern = (torch.arange(hidden, device=f"cuda:{rank}", dtype=torch.int32) % 7 - 3).to(torch.bfloat16)
    native = NativeNccl(uid, rank)
    comm = args.variant in ("graph_chain", "graph_comm", "eager_chain")
    compute = args.variant in ("graph_chain", "graph_compute", "eager_chain")

    def one_layer(x, layer):
        if compute:
            y = torch.mv(weights[layer], x)
            z = y[:hidden] + x
        else:
            z = x
        if comm:
            native.all_reduce(z, stream)
        if compute:
            return z * torch.rsqrt(z.pow(2).mean() + EPS)
        return z

    def chain(x, out_buf):
        for layer in range(layers):
            x = one_layer(x, layer)
        if out_buf is not None:
            out_buf.copy_(x)
        return x

    def reference(x):
        """Exact expected result: both ranks hold identical data, so SUM = 2x."""
        for layer in range(layers):
            if compute:
                z = 2 * (torch.mv(weights[layer], x)[:hidden] + x)
                x = z * torch.rsqrt(z.pow(2).mean() + EPS)
            else:
                x = 2 * x
        return x

    def trial_input():
        if compute:
            return pattern * 0.01
        return pattern * (2.0 ** -layers)

    x0 = trial_input()
    expected = reference(x0)
    assert bool(torch.isfinite(expected).all().item()), "reference non-finite"
    result = torch.empty_like(x0)
    if args.variant == "eager_chain":
        x = x0.clone()
        begun = time.perf_counter_ns()
        chain(x, result)
        torch.cuda.synchronize()
        rows_wall = (time.perf_counter_ns() - begun) / 1000
    else:
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        x = x0.clone()
        with torch.cuda.graph(graph, stream=stream):
            chain(x, result)
        graph.instantiate()
        graph.replay()
        torch.cuda.synchronize()
        max_diff = (result - expected).abs().max().item()
        assert torch.allclose(result, expected, rtol=0.02, atol=0.02), ("graph correctness", args.shape, args.variant, max_diff)
        exact = bool(torch.equal(result, expected))
    if args.variant == "eager_chain":
        exact = bool(torch.equal(result, expected))
        max_diff = (result - expected).abs().max().item()
        assert torch.allclose(result, expected, rtol=0.02, atol=0.02), ("eager correctness", args.shape, args.variant, max_diff)

    gpu = []
    wall = []
    if args.variant == "eager_chain":
        for _ in range(args.warmup):
            chain(x0.clone(), None)
        torch.cuda.synchronize()
        dist.barrier()
        for _ in range(args.samples):
            dist.barrier()
            torch.cuda.synchronize()
            begun = time.perf_counter_ns()
            chain(x0.clone(), None)
            torch.cuda.synchronize()
            wall.append((time.perf_counter_ns() - begun) / 1000)
        gpu = wall
        repeats = 1
    else:
        for _ in range(args.warmup):
            graph.replay()
        torch.cuda.synchronize()
        repeats = 5
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for _ in range(args.samples):
            dist.barrier()
            torch.cuda.synchronize()
            begun = time.perf_counter_ns()
            start.record()
            for _ in range(repeats):
                graph.replay()
            end.record()
            end.synchronize()
            gpu.append(start.elapsed_time(end) * 1000 / repeats)
            wall.append((time.perf_counter_ns() - begun) / 1000 / repeats)
        graph.reset()
        del graph

    launches = layers * ((2 if compute else 1) + (1 if comm else 0)) + 1
    row = {"variant": args.variant, "shape": args.shape, "rank": rank, "layers": layers,
           "hidden": hidden, "half_n": n, "dtype": "bfloat16",
           "collective_bytes": hidden * 2 if comm else 0,
           "samples": args.samples, "replays_per_sample": repeats, "warmup": args.warmup,
           "approx_launches_per_pass": launches,
           "gpu_us_per_pass_samples": gpu, "wall_us_per_pass_samples": wall,
           "gpu_us_per_pass_median": statistics.median(gpu),
           "wall_us_per_pass_median": statistics.median(wall),
           "gpu_us_per_layer_median": statistics.median(gpu) / layers,
           "wall_us_per_layer_median": statistics.median(wall) / layers,
           "gpu_ms_per_pass_median": statistics.median(gpu) / 1000,
           "timing": "wall" if args.variant == "eager_chain" else "cuda_events",
           "exact_equal_to_reference": exact, "max_abs_diff": max_diff,
           "weights_bytes_per_card": layers * n * hidden * 2,
           "correct": True, "finite": True}
    append(out / f"rank{rank}.jsonl", row)
    if rank == 0:
        print(json.dumps(row), flush=True)
    del weights, expected, result, x0
    torch.cuda.synchronize()
    dist.barrier()
    native.close()
    dist.destroy_process_group()


def run(args):
    import torch
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location("s1", ROOT / "lab/01_link.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    meta = helper.metadata(args)
    meta.update(stage="S4/M0 64-layer chain probe",
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                metadata_helper_sha256=hashlib.sha256((ROOT / "lab/01_link.py").read_bytes()).hexdigest(),
                shapes={k: {"layers": v[0], "hidden": v[1], "half_n": v[2]} for k, v in SHAPES.items()},
                started_utc=datetime.now(timezone.utc).isoformat())
    (out / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    jobs = [(shape, variant) for shape in SHAPES for variant in VARIANTS]
    if args.reverse:
        jobs.reverse()
    for shape, variant in jobs:
        dest = out / f"{shape}-{variant}"
        dest.mkdir()
        cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", "--shape", shape,
               "--variant", variant, "--output", str(dest), "--samples", str(args.samples),
               "--warmup", str(args.warmup)]
        env = os.environ.copy()
        env.update(NCCL_DEBUG="INFO", NCCL_DEBUG_FILE=str(dest / "nccl.%h.%p.txt"), OMP_NUM_THREADS="1")
        print("START", shape, variant, flush=True)
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
        status = {"shape": shape, "variant": variant, "returncode": child.returncode,
                  "timed_out": timeout, "seconds": time.monotonic() - begun, "argv": cmd}
        append(out / "status.jsonl", status)
        print(json.dumps(status), flush=True)
    (out / "finished.json").write_text(json.dumps({"finished_utc": datetime.now(timezone.utc).isoformat()}) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--samples", type=int, default=9)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--timeout", type=int, default=180)
    p.add_argument("--reverse", action="store_true")
    p.add_argument("--worker", action="store_true")
    p.add_argument("--shape", choices=tuple(SHAPES))
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
