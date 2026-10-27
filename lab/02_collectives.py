#!/usr/bin/env python3
"""S2 NCCL baseline on two GPUs; one GPU/process, bounded worker lifetimes.

Compare SUM all-reduce, all-gather, SUM reduce-scatter and broadcast, eager
versus graph. graph128 is a dependent 128-SUM chain on the SAME buffer, only
at 10/80 KiB. Timed SUM buffers are zero to avoid exponential overflow. Full
nonzero, rank-distinct BF16 patterns validate actual results outside timing.
The graph128 correctness probe starts at 2^-120, so its exact final value
stays finite. No peer pointers or architecture-specific kernels are used.
"""
import argparse
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
PROFILES = {"default": {}, "ring-ll": {"NCCL_ALGO":"Ring", "NCCL_PROTO":"LL"},
            "tree-ll": {"NCCL_ALGO":"Tree", "NCCL_PROTO":"LL"},
            "ring-simple": {"NCCL_ALGO":"Ring", "NCCL_PROTO":"Simple"}}


def append(path, row):
    with path.open("a") as f:
        f.write(json.dumps(row)+"\n")


def worker(rank, args):
    import torch
    import torch.distributed as dist
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    out=Path(args.output)
    dist.init_process_group("nccl", init_method=(out/"rendezvous").as_uri(),
                            rank=rank, world_size=2, timeout=timedelta(seconds=40),
                            device_id=torch.device(f"cuda:{rank}"))
    stream=torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    unroll=128 if args.mode=="graph128" else 1
    sizes=[s for s in args.sizes if args.mode!="graph128" or s<=81920]
    for size in sizes:
        n=size//2
        pattern=(torch.arange(n,device=f"cuda:{rank}",dtype=torch.int32)%7-3).to(torch.bfloat16)
        inp=torch.empty(n*(2 if args.kind=="reduce_scatter" else 1),
                        device=f"cuda:{rank}",dtype=torch.bfloat16)
        if args.kind in ("all_reduce","broadcast0","broadcast1"):
            output=inp
        else:
            output=torch.empty(n*(2 if args.kind=="all_gather" else 1),
                               device=f"cuda:{rank}",dtype=torch.bfloat16)
        def op():
            if args.kind=="all_reduce":
                work=dist.all_reduce(inp,op=dist.ReduceOp.SUM,async_op=True)
            elif args.kind=="all_gather":
                work=dist.all_gather_into_tensor(output,inp,async_op=True)
            elif args.kind=="reduce_scatter":
                work=dist.reduce_scatter_tensor(output,inp,op=dist.ReduceOp.SUM,async_op=True)
            else:
                work=dist.broadcast(inp,src=int(args.kind[-1]),async_op=True)
            # Orders current stream after NCCL's internal stream. Events on
            # the current stream then include the collective, not just enqueue.
            work.wait()
        def prepare(steps, fresh):
            pat=-pattern+5 if fresh else pattern
            scale=2.0**-120 if steps==128 else 1.0
            if args.kind=="reduce_scatter":
                for shard in range(2):
                    inp[shard*n:(shard+1)*n].copy_((pat+8*shard)*(rank+1))
            else:
                inp.copy_(pat*((rank+1)*scale))
            if output is not inp:
                output.fill_(-99)
            dist.barrier()
            torch.cuda.synchronize()
            return pat
        def verify(pat, steps):
            torch.cuda.synchronize()
            assert bool(torch.isfinite(output).all().item()), "non-finite collective output"
            if args.kind=="all_reduce":
                factor=3*(2.0**(steps-1-120) if steps==128 else 1)
                assert torch.equal(output,pat*factor), "SUM all-reduce mismatch"
            elif args.kind=="all_gather":
                for src in range(2):
                    assert torch.equal(output[src*n:(src+1)*n],pat*(src+1)), "all-gather mismatch"
            elif args.kind=="reduce_scatter":
                assert torch.equal(output,(pat+8*rank)*3), "SUM reduce-scatter mismatch"
            else:
                assert torch.equal(output,pat*(int(args.kind[-1])+1)), "broadcast mismatch"
        pat=prepare(1,False)
        op()
        verify(pat,1)
        inp.zero_()
        if output is not inp:
            output.zero_()
        for _ in range(args.warmup):
            op()
        torch.cuda.synchronize()
        graph=None
        if args.mode!="eager":
            graph=torch.cuda.CUDAGraph()
            dist.barrier()
            torch.cuda.synchronize()
            with torch.cuda.graph(graph,stream=stream):
                for _ in range(unroll):
                    op()
            execute=graph.replay
            for _ in range(args.warmup):
                execute()
            torch.cuda.synchronize()
        else:
            execute=op
        repetitions=(100 if size<=1048576 else 10) if unroll==1 else 5
        start=torch.cuda.Event(enable_timing=True)
        end=torch.cuda.Event(enable_timing=True)
        gpu_us,wall_us=[],[]
        for _ in range(args.samples):
            dist.barrier()
            torch.cuda.synchronize()
            wall_start=time.perf_counter_ns()
            start.record()
            for _ in range(repetitions):
                execute()
            end.record()
            end.synchronize()
            gpu_us.append(start.elapsed_time(end)*1000/repetitions/unroll)
            wall_us.append((time.perf_counter_ns()-wall_start)/1000/repetitions/unroll)
        pat=prepare(unroll,True)
        execute()
        verify(pat,unroll)
        row={"collective":args.kind,"mode":args.mode,"profile":args.profile,
             "rank":rank,"payload_bytes":size,"input_bytes":inp.numel()*inp.element_size(),
             "output_bytes":output.numel()*output.element_size(),"dtype":"bfloat16",
             "reduction":"SUM" if args.kind in ("all_reduce","reduce_scatter") else None,
             "unroll":unroll,"repetitions":repetitions,"samples":args.samples,
             "gpu_us_samples":gpu_us,"wall_us_samples":wall_us,
             "gpu_us_median":statistics.median(gpu_us),
             "wall_us_median":statistics.median(wall_us),
             "gpu_us_per_replay_median":statistics.median(gpu_us)*unroll,
             "correct":True,"finite":True,"fresh_pattern_correct":True,
             "timed_inputs":"zeros; nonzero rank-specific verification outside timing"}
        append(out/f"rank{rank}.jsonl",row)
        if rank==0:
            print(json.dumps(row),flush=True)
        del execute
        if graph is not None:
            graph.reset()
        del graph,inp,output,pattern,pat
    dist.destroy_process_group()


def run(args):
    out=Path(args.output).resolve()
    out.mkdir(parents=True,exist_ok=False)
    spec=importlib.util.spec_from_file_location("s1_link",ROOT/"lab/01_link.py")
    link=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(link)
    meta=link.metadata(args)
    meta["stage"]="S2 NCCL baseline (not hand-written one-shot)"
    meta["script_sha256"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    meta["metadata_helper_sha256"]=hashlib.sha256((ROOT/"lab/01_link.py").read_bytes()).hexdigest()
    meta["profile_environment"]=PROFILES[args.profile]
    meta["started_utc"]=datetime.now(timezone.utc).isoformat()
    (out/"metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
    variants=[(kind,mode) for kind in args.collectives for mode in ("eager","graph1")]
    if "all_reduce" in args.collectives:
        variants.append(("all_reduce","graph128"))
    for kind,mode in variants:
        name=f"{kind}-{mode}"
        dest=out/name
        dest.mkdir()
        cmd=[sys.executable,str(Path(__file__).resolve()),"--worker","--kind",kind,
             "--mode",mode,"--profile",args.profile,"--output",str(dest),
             "--samples",str(args.samples),"--warmup",str(args.warmup),"--sizes"]+list(map(str,args.sizes))
        env=os.environ.copy()
        env.update(PROFILES[args.profile])
        env.update(NCCL_DEBUG="INFO",NCCL_DEBUG_FILE=str(dest/"nccl.%h.%p.txt"),OMP_NUM_THREADS="1")
        print("START",name,args.profile,flush=True)
        begun=time.monotonic()
        with (dest/"stdout.txt").open("w") as stdout,(dest/"stderr.txt").open("w") as stderr:
            child=subprocess.Popen(cmd,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
            timeout=False
            try:
                child.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                timeout=True
                os.killpg(child.pid,signal.SIGTERM)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL)
                    child.wait()
        status={"variant":name,"profile":args.profile,"returncode":child.returncode,
                "timed_out":timeout,"seconds":time.monotonic()-begun,"argv":cmd}
        append(out/"status.jsonl",status)
        print(json.dumps(status),flush=True)
    (out/"finished.json").write_text(json.dumps({"finished_utc":datetime.now(timezone.utc).isoformat()})+"\n")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",required=True)
    p.add_argument("--sizes",nargs="+",type=int,default=[10240,81920,1048576,67108864])
    p.add_argument("--collectives",nargs="+",choices=("all_reduce","all_gather","reduce_scatter","broadcast0","broadcast1"),
                   default=["all_reduce","all_gather","reduce_scatter","broadcast0","broadcast1"])
    p.add_argument("--samples",type=int,default=9)
    p.add_argument("--warmup",type=int,default=20)
    p.add_argument("--timeout",type=int,default=75)
    p.add_argument("--profile",choices=tuple(PROFILES),default="default")
    p.add_argument("--worker",action="store_true")
    p.add_argument("--kind",choices=("all_reduce","all_gather","reduce_scatter","broadcast0","broadcast1"))
    p.add_argument("--mode",choices=("eager","graph1","graph128"))
    args=p.parse_args()
    if args.samples<1 or args.warmup<1 or any(s<2 or s%2 for s in args.sizes):
        p.error("positive samples/warmup and even payload sizes >=2 bytes required")
    if args.worker:
        import torch.multiprocessing as mp
        mp.spawn(worker,args=(args,),nprocs=2,join=True)
    else:
        run(args)


if __name__=="__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
