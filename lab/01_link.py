#!/usr/bin/env python3
"""S1: dependent copy latency, size sweep, graph support, NCCL broadcast.

Run from the repository with its CUDA-enabled Python:
  .venv/bin/python lab/01_link.py --output results/<date>-<box>/run1

10 KB means 10,240 bytes. Copy latency is NOT all-reduce latency. A round
trip is two ordered copies of the same payload, never two independent copies.
Each variant runs in a fresh subprocess with a wall-clock timeout. No driver
or global NCCL configuration is modified. P2P is queried, never assumed.
"""

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
import traceback

SIZES = [64, 256, 1024, 4096, 8192, 10240, 16384, 65536, 262144,
         1048576, 4194304, 16777216, 67108864, 268435456]
ROOT = Path(__file__).resolve().parents[1]


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2) + "\n")


def append(path, obj):
    with path.open("a") as f:
        f.write(json.dumps(obj) + "\n")


def command(args):
    p = subprocess.run(args, capture_output=True, text=True, timeout=30)
    return {"argv": args, "returncode": p.returncode,
            "stdout": p.stdout, "stderr": p.stderr}


def sync(torch):
    for device in range(2):
        torch.cuda.synchronize(device)


def timing(torch, op, device, iterations, samples):
    """Both events on one GPU; op must join back onto its current stream."""
    gpu_us, wall_us = [], []
    with torch.cuda.device(device):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for _ in range(samples):
            sync(torch)
            wall_start = time.perf_counter_ns()
            start.record()
            for _ in range(iterations):
                op()
            end.record()
            end.synchronize()
            wall_us.append((time.perf_counter_ns() - wall_start) / 1000 / iterations)
            gpu_us.append(start.elapsed_time(end) * 1000 / iterations)
    return {"gpu_us_samples": gpu_us, "wall_us_samples": wall_us,
            "gpu_us_median": statistics.median(gpu_us),
            "wall_us_median": statistics.median(wall_us)}


def copy_worker(args):
    import torch
    torch.set_num_threads(1)
    torch.cuda.set_device(args.src)
    # Pin a non-default current stream on each GPU. copy_ inserts the
    # source->destination stream dependencies; returning the payload creates
    # an explicit dependency on the forward copy, including the other GPU.
    streams = [torch.cuda.Stream(device=i) for i in range(2)]
    for stream in streams:
        torch.cuda.set_stream(stream)
    torch.cuda.set_device(args.src)
    out = Path(args.output)
    for size in args.sizes:
        a = torch.full((size,), 17, dtype=torch.uint8, device=f"cuda:{args.src}")
        b = torch.zeros(size, dtype=torch.uint8, device=f"cuda:{1-args.src}")
        def op():
            b.copy_(a, non_blocking=True)
            if args.kind == "roundtrip":
                a.copy_(b, non_blocking=True)
        for _ in range(args.warmup):
            op()
        sync(torch)
        # Always check an eager transfer before attempting capture.
        assert bool(torch.all(b == 17).item()), "eager payload mismatch"
        unroll = 100 if size <= 1048576 else 4
        graph = None
        if args.mode == "graph":
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.device(args.src):
                with torch.cuda.graph(graph, stream=streams[args.src]):
                    # Fork the peer stream from the capture origin BEFORE
                    # copy_ inserts its bidirectional event dependencies.
                    # Otherwise its first event refers to uncaptured work.
                    streams[1-args.src].wait_stream(streams[args.src])
                    for _ in range(unroll):
                        op()
                    streams[args.src].wait_stream(streams[1-args.src])
            for _ in range(args.warmup):
                graph.replay()
            sync(torch)
            measured_op = graph.replay
            iterations = 10 if size <= 1048576 else 3
        else:
            unroll = 1
            measured_op = op
            iterations = 100 if size <= 1048576 else 8
        row = timing(torch, measured_op, args.src, iterations, args.samples)
        for key in ("gpu_us_samples", "wall_us_samples"):
            row[key] = [x / unroll for x in row[key]]
        for key in ("gpu_us_median", "wall_us_median"):
            row[key] /= unroll
        # A fresh value after capture proves that graph replay moves data,
        # rather than validating stale bytes left by the warm-up.
        a.fill_(91)
        b.zero_()
        sync(torch)
        measured_op()
        sync(torch)
        assert bool(torch.all(b == 91).item()), "fresh destination payload mismatch"
        assert bool(torch.all(a == 91).item()), "roundtrip source payload mismatch"
        hops = 2 if args.kind == "roundtrip" else 1
        row.update({"kind": args.kind, "mode": args.mode, "src": args.src,
                    "dst": 1-args.src, "bytes": size, "hops": hops,
                    "iterations_per_sample": iterations, "graph_unroll": unroll,
                    "samples": args.samples, "correct": True,
                    "us_per_hop_proxy": row["gpu_us_median"] / hops,
                    "effective_GBs": size * hops / row["gpu_us_median"] / 1000})
        append(out / "rows.jsonl", row)
        print(json.dumps(row), flush=True)
        del graph, a, b


def nccl_worker(rank, args):
    import torch
    import torch.distributed as dist
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    out = Path(args.output)
    dist.init_process_group("nccl", init_method=(out / "rendezvous").as_uri(),
                            rank=rank, world_size=2, timeout=timedelta(seconds=45),
                            device_id=torch.device(f"cuda:{rank}"))
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    for src in (0, 1):
        for size in args.sizes:
            t = torch.full((size,), 17 if rank == src else 0,
                           dtype=torch.uint8, device=f"cuda:{rank}")
            def op():
                # Work.wait orders the current CUDA stream after NCCL's
                # internal stream; GPU events must include the communication.
                dist.broadcast(t, src=src, async_op=True).wait()
            for _ in range(args.warmup):
                op()
            torch.cuda.synchronize()
            assert bool(torch.all(t == 17).item()), "broadcast warmup mismatch"
            unroll = 100 if size <= 1048576 else 4
            graph = None
            if args.mode == "graph":
                graph = torch.cuda.CUDAGraph()
                dist.barrier()
                torch.cuda.synchronize()
                with torch.cuda.graph(graph, stream=stream):
                    for _ in range(unroll):
                        op()
                for _ in range(args.warmup):
                    graph.replay()
                torch.cuda.synchronize()
                measured_op = graph.replay
                iterations = 10 if size <= 1048576 else 3
            else:
                unroll = 1
                measured_op = op
                iterations = 100 if size <= 1048576 else 8
            gpu_us, wall_us = [], []
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            for _ in range(args.samples):
                dist.barrier()
                torch.cuda.synchronize()
                wall_start = time.perf_counter_ns()
                start.record()
                for _ in range(iterations):
                    measured_op()
                end.record()
                end.synchronize()
                gpu_us.append(start.elapsed_time(end) * 1000 / iterations / unroll)
                wall_us.append((time.perf_counter_ns()-wall_start) / 1000 / iterations / unroll)
            t.fill_(91 if rank == src else 0)
            dist.barrier()
            torch.cuda.synchronize()
            measured_op()
            torch.cuda.synchronize()
            assert bool(torch.all(t == 91).item()), "fresh broadcast payload mismatch"
            row = {"kind": "broadcast", "mode": args.mode, "rank": rank,
                   "src": src, "bytes": size, "hops": 1, "correct": True,
                   "iterations_per_sample": iterations, "graph_unroll": unroll,
                   "samples": args.samples, "gpu_us_samples": gpu_us,
                   "wall_us_samples": wall_us,
                   "gpu_us_median": statistics.median(gpu_us),
                   "wall_us_median": statistics.median(wall_us)}
            row["effective_GBs"] = size / row["gpu_us_median"] / 1000
            append(out / f"rank{rank}.jsonl", row)
            if rank == 0:
                print(json.dumps(row), flush=True)
            del graph, t
    dist.destroy_process_group()


def metadata(args):
    import torch
    if torch.cuda.device_count() < 2:
        raise RuntimeError("S1 requires at least two CUDA GPUs")
    info = {"started_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version, "torch": torch.__version__,
            "cuda": torch.version.cuda, "nccl": torch.cuda.nccl.version(),
            "architectures": torch.cuda.get_arch_list(),
            "parameters": vars(args),
            "git_commit": command(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
            "git_status": command(["git", "-C", str(ROOT), "status", "--porcelain"]),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "gpus": [str(torch.cuda.get_device_properties(i)) for i in range(2)],
            "p2p": {f"{i}->{j}": torch.cuda.can_device_access_peer(i, j)
                    for i, j in ((0, 1), (1, 0))},
            "environment": {k: v for k, v in os.environ.items()
                            if k.startswith("NCCL_") or k in
                            ("CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH", "OMP_NUM_THREADS")},
            "commands": [command(c) for c in
                         (["nvidia-smi", "topo", "-m"],
                          ["nvidia-smi", "topo", "-p2p", "r"],
                          ["nvidia-smi", "topo", "-p2p", "w"],
                          ["nvidia-smi", "--query-gpu=index,name,uuid,driver_version,pci.bus_id,pcie.link.gen.current,pcie.link.width.current,power.limit,clocks.current.sm,clocks.current.memory,temperature.gpu,utilization.gpu", "--format=csv"])],
            "p2p_enabled_variant": "available" if all(
                torch.cuda.can_device_access_peer(i,j) for i,j in ((0,1),(1,0)))
                else "UNAVAILABLE: runtime peer access is false; no driver patch attempted"}
    for i in range(2):
        x = torch.ones(4096, device=f"cuda:{i}")
        assert x.sum().item() == 4096, "CUDA arithmetic smoke check failed"
    maps = Path("/proc/self/maps").read_text().splitlines()
    info["loaded_cuda_libraries"] = sorted(set(line.split()[-1] for line in maps
                                             if "/" in line and any(s in line for s in
                                             ("libcuda.", "libcudart.", "libnccl."))))
    return info


def run(args):
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    write(out / "metadata.json", metadata(args))
    variants = [(kind, mode, src) for kind in ("copy", "roundtrip")
                for mode in ("eager", "graph") for src in (0, 1)]
    variants += [("broadcast", mode, 0) for mode in ("eager", "graph")]
    for kind, mode, src in variants:
        name = f"{kind}-{mode}-{src}"
        dest = out / name
        dest.mkdir()
        cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", kind,
               "--mode", mode, "--src", str(src), "--output", str(dest),
               "--samples", str(args.samples), "--warmup", str(args.warmup),
               "--sizes"] + list(map(str, args.sizes))
        env = os.environ.copy()
        env.update({"NCCL_DEBUG": "INFO", "NCCL_DEBUG_FILE": str(dest / "nccl.%h.%p.txt"),
                    "OMP_NUM_THREADS": "1"})
        print(f"START {name}", flush=True)
        start = time.monotonic()
        with (dest / "stdout.txt").open("w") as stdout, (dest / "stderr.txt").open("w") as stderr:
            p = subprocess.Popen(cmd, stdout=stdout, stderr=stderr, env=env,
                                 start_new_session=True)
            timed_out = False
            try:
                p.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(p.pid, signal.SIGTERM)
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
                    p.wait()
        status = {"variant": name, "returncode": p.returncode, "timed_out": timed_out,
                  "seconds": time.monotonic()-start, "argv": cmd}
        append(out / "status.jsonl", status)
        print(json.dumps(status), flush=True)
    write(out / "finished.json", {"finished_utc": datetime.now(timezone.utc).isoformat(),
                                  "note": "Inspect all variant statuses; failures are results."})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--samples", type=int, default=7)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--timeout", type=int, default=180)
    p.add_argument("--sizes", nargs="+", type=int, default=SIZES)
    p.add_argument("--worker", choices=("copy", "roundtrip", "broadcast"))
    p.add_argument("--mode", choices=("eager", "graph"), default="eager")
    p.add_argument("--src", type=int, choices=(0, 1), default=0)
    args = p.parse_args()
    if args.samples < 1 or args.warmup < 1 or any(s < 1 for s in args.sizes):
        p.error("samples, warmup and sizes must be positive")
    if args.worker == "broadcast":
        import torch.multiprocessing as mp
        mp.spawn(nccl_worker, args=(args,), nprocs=2, join=True)
    elif args.worker:
        args.kind = args.worker
        copy_worker(args)
    else:
        run(args)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
