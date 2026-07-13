# Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Test script for fused matmul + reduce-scatter kernels.

Run with pytest:
    pytest tests/comm/test_matmul_reduce_scatter.py -vv -s

Run standalone:
    python test_matmul_reduce_scatter.py --correctness
    python test_matmul_reduce_scatter.py --benchmark
    python test_matmul_reduce_scatter.py --profile

Other options:
    --dtype: Data type for input and weight tensors (default: bfloat16)
"""

import argparse
import pytest
import random
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from flashinfer.comm import MatmulReduceScatterWorkspace, matmul_reduce_scatter
from flashinfer.utils import get_compute_capability

HID = 8192
OUT_HID = 2048


# Reference matmul then reduce-scatter implementation (the baseline this op
# replaces; used for benchmark timing).
def ref_matmul_reduce_scatter(
    inp: torch.Tensor,
    w: torch.Tensor,
    group: dist.ProcessGroup,
):
    world_size = dist.get_world_size(group)
    y = inp @ w  # (M, N) local partial
    out = torch.empty(
        (inp.shape[0] // world_size, w.shape[1]), device=inp.device, dtype=inp.dtype
    )
    dist.reduce_scatter_tensor(out, y, group=group)
    return out


# fp32 ground truth: reduce-scatter of the fp32 matmul. Used for correctness.
def fp32_truth(inp: torch.Tensor, w: torch.Tensor, group: dist.ProcessGroup):
    world_size = dist.get_world_size(group)
    y = (inp.float() @ w.float()).contiguous()
    out = torch.empty(
        (inp.shape[0] // world_size, w.shape[1]),
        device=inp.device,
        dtype=torch.float32,
    )
    dist.reduce_scatter_tensor(out, y, group=group)
    return out


# Reduce-scatter sums partials of large opposing magnitude, so individual output
# elements can suffer catastrophic cancellation -- an elementwise atol/rtol check
# is meaningless near zero. We instead validate the relative error in norm against
# an fp32 ground truth. For reference, the bf16 NCCL baseline itself lands around
# 0.3% relative-norm error; the fused kernel accumulates in fp32 and is typically
# tighter (~0.24%).
REL_NORM_TOL = 2e-2


def assert_rs_close(got: torch.Tensor, inp, w, group, *, name: str = ""):
    truth = fp32_truth(inp, w, group)
    assert torch.isfinite(got).all(), f"non-finite output{(' ' + name) if name else ''}"
    rel = (got.float() - truth).norm() / truth.norm()
    assert rel < REL_NORM_TOL, (
        f"relative-norm error {rel.item():.4f} exceeds {REL_NORM_TOL} "
        f"{('(' + name + ')') if name else ''}"
    )
    return rel


def setup(rank: int, world_size: int, port: int):
    """Initialize distributed process group and return common state."""
    print(f"Rank {rank} of {world_size} is initializing")
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=world_size,
        device_id=device,
    )
    group = dist.group.WORLD
    torch.manual_seed(rank + 52)
    return device, group


def unit_test(rank: int, world_size: int, port: int, dtype: torch.dtype):
    device, group = setup(rank, world_size, port)
    w = torch.randn((HID, OUT_HID), device=device, dtype=dtype)
    inp = torch.randn(16 * 1024, HID, device=device, dtype=dtype)
    workspace = MatmulReduceScatterWorkspace(group, 16 * 1024, OUT_HID, dtype=dtype)
    for strategy in ("tail", "push", "auto"):
        out = matmul_reduce_scatter(
            inp, w, group, workspace, verbose=True, strategy=strategy
        )
        rel = assert_rs_close(out, inp, w, group, name=strategy)
        if rank == 0:
            print(
                f"unit_test passed ({strategy}): relative-norm error = {rel.item():.5f}"
            )
    workspace.destroy()
    dist.destroy_process_group()


@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="Tests require at least 2 CUDA devices",
)
@pytest.mark.skipif(
    get_compute_capability(torch.device("cuda:0"))[0] < 9,
    reason="Tests runs only on SM90+ devices",
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_matmul_reduce_scatter(dtype: torch.dtype):
    import os
    import sys

    # mp.spawn starts fresh interpreters that need to re-import this module;
    # ensure the repo root is on sys.path so 'tests.comm' is findable.
    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    port = random.randint(30000, 60000)
    world_size = torch.cuda.device_count()
    mp.spawn(unit_test, args=(world_size, port, dtype), nprocs=world_size, join=True)


def run_profile(rank: int, world_size: int, port: int, dtype: torch.dtype):
    device, group = setup(rank, world_size, port)
    w = torch.randn((HID, OUT_HID), device=device, dtype=dtype)
    inp = torch.randn(16 * 1024, HID, device=device, dtype=dtype)
    workspace = MatmulReduceScatterWorkspace(group, 16 * 1024, OUT_HID, dtype=dtype)
    # Warmup
    matmul_reduce_scatter(inp, w, group, workspace, verbose=True)
    ref_matmul_reduce_scatter(inp, w, group)
    # Synchronize timer
    dist.barrier(group)

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
    ) as prof:
        for _ in range(3):
            matmul_reduce_scatter(inp, w, group, workspace)
            ref_matmul_reduce_scatter(inp, w, group)
            prof.step()
    torch.cuda.synchronize()
    workspace.destroy()
    gpu_arch = torch.cuda.get_device_properties(device).name.replace(" ", "_")
    prof.export_chrome_trace(f"{gpu_arch}_rs_rank{rank}.json")
    dist.destroy_process_group()


def run_correctness(rank: int, world_size: int, port: int, dtype: torch.dtype):
    device, group = setup(rank, world_size, port)
    w = torch.randn((HID, OUT_HID), device=device, dtype=dtype)
    bs_list = [2**15, 2**14, 2**13, 2**12]
    # Non-power-of-two sizes (kept divisible by world_size, which reduce-scatter
    # requires; the odd per-rank chunk also exercises partial reduce tiles).
    bs_list += [world_size * 520]
    # One workspace sized for the largest bs, reused for every smaller one.
    workspace = MatmulReduceScatterWorkspace(group, max(bs_list), OUT_HID, dtype=dtype)
    for bs in bs_list:
        for strategy in ("tail", "push"):
            for _ in range(10):
                inp = torch.randn(bs, HID, device=device, dtype=dtype)
                dist.barrier(group)
                out = matmul_reduce_scatter(inp, w, group, workspace, strategy=strategy)
                assert_rs_close(out, inp, w, group, name=strategy)
        print(f"Rank {rank} of {world_size}: Correctness check passed (bs={bs})")
    workspace.destroy()
    dist.destroy_process_group()


def run_benchmark(rank: int, world_size: int, port: int, dtype: torch.dtype):
    device, group = setup(rank, world_size, port)
    w = torch.randn((HID, OUT_HID), device=device, dtype=dtype)

    # Warmup: compile kernels + allocate the fused op's symmetric buffers, and
    # ramp GPU clocks with a sustained matmul burst so the first (largest) timed
    # size is not measured at cold boost clocks (which otherwise shows a one-off
    # spike for that size only).
    inp = torch.randn(16 * 1024, HID, device=device, dtype=dtype)
    workspace = MatmulReduceScatterWorkspace(group, 2**16, OUT_HID, dtype=dtype)
    matmul_reduce_scatter(inp, w, group, workspace)
    clock_warm = torch.randn(16 * 1024, HID, device=device, dtype=dtype)
    for _ in range(50):
        _ = clock_warm @ w
    torch.cuda.synchronize(device)
    dist.barrier(group)

    bench_warmup = 10
    bench_iters = 50

    def _bench_one_mean_ms(fn):
        dist.barrier(group)
        torch.cuda.synchronize(device)
        for _ in range(bench_warmup):
            fn()
        dist.barrier(group)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        total_ms = 0.0
        for _ in range(bench_iters):
            start.record()
            fn()
            end.record()
            end.synchronize()
            total_ms += float(start.elapsed_time(end))
        mean_ms = total_ms / bench_iters
        local = torch.tensor([mean_ms], device=device, dtype=torch.float32)
        gathered = [torch.empty_like(local) for _ in range(world_size)]
        dist.all_gather(gathered, local, group=group)
        if rank != 0:
            return None
        means = torch.stack(gathered, dim=0).cpu().flatten()  # (world,)
        avg_mean = means.mean().item()
        max_mean = means.max().item()
        return avg_mean, max_mean

    bs_list = [2**16, 2**15, 2**14, 2**13, 2**12, 2**11, 2**10]
    if rank == 0:
        print(f"[benchmark] iters={bench_iters}, warmup={bench_warmup}")
        bs_w = 8
        col_w = 13
        inner_gap = " "
        group_w = col_w * 2 + len(inner_gap)
        speedup_w = 9
        vbar = " | "
        header_top = (
            f"{'':>{bs_w}}{vbar}"
            f"{'pull':^{group_w}}{vbar}"
            f"{'ref':^{group_w}}{vbar}"
            f"{'speedup':^{speedup_w}}"
        )
        header_bottom = (
            f"{'# tokens':>{bs_w}}{vbar}"
            f"{'avg_mean':>{col_w}}{inner_gap}{'max_mean':>{col_w}}{vbar}"
            f"{'avg_mean':>{col_w}}{inner_gap}{'max_mean':>{col_w}}{vbar}"
            f"{'avg':>{speedup_w}}"
        )
        sep = "-" * len(header_bottom)
        print(sep)
        print(header_top)
        print(header_bottom)
        print(sep)

    for bs in bs_list:
        inp = torch.randn(bs, HID, device=device, dtype=dtype)
        dist.barrier(group)
        pull = _bench_one_mean_ms(
            lambda: matmul_reduce_scatter(inp, w, group, workspace)
        )
        ref = _bench_one_mean_ms(lambda: ref_matmul_reduce_scatter(inp, w, group))
        if rank == 0:
            pull_avg, pull_max = pull
            ref_avg, ref_max = ref
            speedup = ref_avg / pull_avg
            print(
                f"{bs:8d}{vbar}"
                f"{pull_avg:13.3f} {pull_max:13.3f}{vbar}"
                f"{ref_avg:13.3f} {ref_max:13.3f}{vbar}"
                f"{speedup:>{speedup_w}.2f}x"
            )

    if rank == 0:
        print(sep)
    workspace.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--correctness",
        action="store_true",
        help="Check correctness across a range of batch sizes",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Run with torch profiler and export Chrome trace",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Benchmark GPU time of pull vs reference implementations",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Data type for input and weight tensors (default: bfloat16)",
    )
    args = parser.parse_args()

    # IP port number for multi-process rendezvous
    port = random.randint(30000, 60000)
    world_size = torch.cuda.device_count()
    dtype = getattr(torch, args.dtype)
    spawn_kwargs = dict(args=(world_size, port, dtype), nprocs=world_size, join=True)

    if args.profile:
        mp.spawn(run_profile, **spawn_kwargs)
    elif args.correctness:
        mp.spawn(run_correctness, **spawn_kwargs)
    elif args.benchmark:
        mp.spawn(run_benchmark, **spawn_kwargs)
    else:
        mp.spawn(unit_test, **spawn_kwargs)
