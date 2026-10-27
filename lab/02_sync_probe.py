#!/usr/bin/env python3
"""Bounded S2 cause experiment: ordering, stream placement, real consumers.

wait_tail is ONLY a communication-only control, not a valid replacement for
per-layer consumer synchronization. same_stream places native NCCL and local
PyTorch compute on the same CUDA stream via torch.cuda.nccl, reusing NCCL.
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

ROOT=Path(__file__).resolve().parents[1]
VARIANTS=("wait_each","wait_tail","same_stream","wait_each_consumer",
          "same_stream_consumer","local_consumer")


class NativeNccl:
    """Thin public NCCL C-API binding; tensors/streams remain PyTorch-owned.

    torch.cuda.nccl.init_rank in this wheel fails under Python 3.12 with
    PY_SSIZE_T_CLEAN. Reuse the already loaded NCCL library directly instead.
    Signatures, UID size and bf16 enum are checked against installed nccl.h.
    """
    class UniqueId(ctypes.Structure):
        _fields_=[("internal",ctypes.c_char*128)]

    def __init__(self,uid,rank):
        libraries={line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                   if '/libnccl.so' in line}
        assert len(libraries)==1,libraries
        self.lib=ctypes.CDLL(next(iter(libraries)))
        self.lib.ncclGetErrorString.argtypes=[ctypes.c_int]
        self.lib.ncclGetErrorString.restype=ctypes.c_char_p
        self.lib.ncclCommInitRank.argtypes=[ctypes.POINTER(ctypes.c_void_p),ctypes.c_int,self.UniqueId,ctypes.c_int]
        self.lib.ncclAllReduce.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t,
                                        ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p]
        self.lib.ncclCommDestroy.argtypes=[ctypes.c_void_p]
        for name in ('ncclCommInitRank','ncclAllReduce','ncclCommDestroy'):
            getattr(self.lib,name).restype=ctypes.c_int
        self.comm=ctypes.c_void_p()
        assert len(uid)==128
        self.check(self.lib.ncclCommInitRank(ctypes.byref(self.comm),2,self.UniqueId.from_buffer_copy(uid),rank))

    def check(self,code):
        if code:raise RuntimeError(self.lib.ncclGetErrorString(code).decode())

    def all_reduce(self,tensor,stream):
        # ncclBfloat16=9; ncclSum=0, from the installed NCCL 2.28 header.
        self.check(self.lib.ncclAllReduce(tensor.data_ptr(),tensor.data_ptr(),tensor.numel(),9,0,
                                         self.comm,stream.cuda_stream))

    def close(self):
        self.check(self.lib.ncclCommDestroy(self.comm))


def append(path,row):
    with path.open("a") as f:f.write(json.dumps(row)+"\n")


def worker(rank,args,uid):
    import torch
    import torch.distributed as dist
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    out=Path(args.output)
    dist.init_process_group("nccl",init_method=(out/"rendezvous").as_uri(),
                            rank=rank,world_size=2,timeout=timedelta(seconds=40),
                            device_id=torch.device(f"cuda:{rank}"))
    stream=torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    native_comm=None
    if args.variant.startswith("same_stream"):
        native_comm=NativeNccl(uid,rank)
    consumer=args.variant.endswith("consumer")
    local=args.variant=="local_consumer"
    for size in args.sizes:
        n=size//2
        pattern=(torch.arange(n,device=f"cuda:{rank}",dtype=torch.int32)%7-3).to(torch.bfloat16)
        tensor=torch.empty(n,device=f"cuda:{rank}",dtype=torch.bfloat16)
        for depth in args.depths:
            def chain():
                handles=[]
                for _ in range(depth):
                    if not local:
                        if native_comm is not None:
                            native_comm.all_reduce(tensor,stream)
                        else:
                            work=dist.all_reduce(tensor,op=dist.ReduceOp.SUM,async_op=True)
                            handles.append(work)
                            if args.variant!="wait_tail":work.wait()
                    if consumer:tensor.mul_(0.5)
                if args.variant=="wait_tail":handles[-1].wait()
            def prepare(fresh):
                pat=-pattern+5 if fresh else pattern
                if local:
                    factor=(rank+1)*2.0**min(depth,120)
                elif consumer:
                    factor=rank+1
                else:
                    factor=(rank+1)*2.0**-min(depth,120)
                tensor.copy_(pat*factor)
                dist.barrier()
                torch.cuda.synchronize()
                return pat
            def verify(pat):
                torch.cuda.synchronize()
                if local:factor=(rank+1)*2.0**(min(depth,120)-depth)
                elif consumer:factor=1.5
                else:factor=3*2.0**(depth-1-min(depth,120))
                assert bool(torch.isfinite(tensor).all().item()),"non-finite output"
                assert torch.equal(tensor,pat*factor),f"fresh correctness failed: {args.variant}/{size}/{depth}"
            pat=prepare(False)
            chain()
            verify(pat)
            tensor.zero_()
            for _ in range(args.warmup):chain()
            torch.cuda.synchronize()
            dist.barrier()
            torch.cuda.synchronize()
            # Keep the raw graph explicitly owned until reset. In this
            # PyTorch build global debug mode with keep_graph=False retains
            # a raw graph while marking has_graph_=false, leaking NCCL graph
            # user objects and stalling communicator destruction.
            graph=torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(graph,stream=stream):chain()
            graph.instantiate()
            if size==10240 and depth==128 and rank==0:
                graph.debug_dump(str(out/"graph-10KiB-128.dot"))
            for _ in range(args.warmup):graph.replay()
            torch.cuda.synchronize()
            repeats=100 if depth==1 else max(5,128//depth)
            start=torch.cuda.Event(enable_timing=True)
            end=torch.cuda.Event(enable_timing=True)
            gpu,wall=[],[]
            for _ in range(args.samples):
                dist.barrier()
                torch.cuda.synchronize()
                begun=time.perf_counter_ns()
                start.record()
                for _ in range(repeats):graph.replay()
                end.record()
                end.synchronize()
                gpu.append(start.elapsed_time(end)*1000/repeats/depth)
                wall.append((time.perf_counter_ns()-begun)/1000/repeats/depth)
            pat=prepare(True)
            graph.replay()
            verify(pat)
            row={"variant":args.variant,"rank":rank,"payload_bytes":size,"depth":depth,
                 "dtype":"bfloat16","samples":args.samples,"replays_per_sample":repeats,
                 "gpu_us_per_step_samples":gpu,"wall_us_per_step_samples":wall,
                 "gpu_us_per_step_median":statistics.median(gpu),
                 "wall_us_per_step_median":statistics.median(wall),
                 "gpu_ms_per_chain_median":statistics.median(gpu)*depth/1000,
                 "correct":True,"fresh_pattern_correct":True,"finite":True,
                 "has_intermediate_consumer":consumer,"native_same_stream":native_comm is not None}
            append(out/f"rank{rank}.jsonl",row)
            if rank==0:print(json.dumps(row),flush=True)
            graph.reset()
            del graph,pat
        del tensor,pattern
    dist.barrier()
    torch.cuda.synchronize()
    if native_comm is not None:native_comm.close()
    del native_comm
    dist.destroy_process_group()


def run(args):
    import torch
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    spec=importlib.util.spec_from_file_location("s1",ROOT/"lab/01_link.py")
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    meta=helper.metadata(args)
    meta.update(stage="S2-sync cause probe",script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                metadata_helper_sha256=hashlib.sha256((ROOT/"lab/01_link.py").read_bytes()).hexdigest(),
                native_wrapper_sha256=hashlib.sha256(Path(torch.cuda.nccl.__file__).read_bytes()).hexdigest(),
                started_utc=datetime.now(timezone.utc).isoformat())
    (out/"metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
    variants=list(VARIANTS)
    if args.reverse:variants.reverse()
    for variant in variants:
        dest=out/variant;dest.mkdir()
        cmd=[sys.executable,str(Path(__file__).resolve()),"--worker","--variant",variant,
             "--output",str(dest),"--samples",str(args.samples),"--warmup",str(args.warmup),
             "--sizes"]+list(map(str,args.sizes))+["--depths"]+list(map(str,args.depths))
        env=os.environ.copy();env.update(NCCL_DEBUG="INFO",NCCL_DEBUG_FILE=str(dest/"nccl.%h.%p.txt"),OMP_NUM_THREADS="1")
        print("START",variant,flush=True);begun=time.monotonic()
        with (dest/"stdout.txt").open("w") as stdout,(dest/"stderr.txt").open("w") as stderr:
            child=subprocess.Popen(cmd,stdout=stdout,stderr=stderr,env=env,start_new_session=True)
            timeout=False
            try:child.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                timeout=True;os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        status={"variant":variant,"returncode":child.returncode,"timed_out":timeout,
                "seconds":time.monotonic()-begun,"argv":cmd}
        append(out/"status.jsonl",status);print(json.dumps(status),flush=True)
    (out/"finished.json").write_text(json.dumps({"finished_utc":datetime.now(timezone.utc).isoformat()})+"\n")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",required=True)
    p.add_argument("--samples",type=int,default=9)
    p.add_argument("--warmup",type=int,default=20)
    p.add_argument("--timeout",type=int,default=75)
    p.add_argument("--sizes",nargs="+",type=int,default=[10240,81920])
    p.add_argument("--depths",nargs="+",type=int,default=[1,8,32,128])
    p.add_argument("--reverse",action="store_true")
    p.add_argument("--worker",action="store_true")
    p.add_argument("--variant",choices=VARIANTS)
    args=p.parse_args()
    if any(d<1 or d>128 for d in args.depths) or any(s<2 or s%2 for s in args.sizes):
        p.error("even positive sizes and depths 1..128 required")
    if args.worker:
        import torch
        import torch.multiprocessing as mp
        uid=torch.cuda.nccl.unique_id() if args.variant.startswith("same_stream") else None
        mp.spawn(worker,args=(args,uid),nprocs=2,join=True)
    else:run(args)


if __name__=="__main__":
    try:main()
    except Exception:traceback.print_exc();sys.exit(1)
