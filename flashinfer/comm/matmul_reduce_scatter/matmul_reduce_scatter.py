# Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Matmul + reduce-scatter fused using a compute-signal/pull-reduce algorithm.

For testing, see test_matmul_reduce_scatter.py.

Problem:
    Each rank computes the partial Y = input @ weight from its local input (M, K)
    and weight (K, N); rank p keeps output (M // world_size, N), the sum over
    ranks of Y[p * chunk : (p + 1) * chunk], chunk = M // world_size. Equivalent
    to torch.distributed.reduce_scatter_tensor(output, input @ weight, group).

Algorithm (compute-signal/pull-reduce):
    The GEMM writes Y into a symmetric buffer, then signals; a kernel spin-waits
    per source and sums the rank's token slice in fp32. Two strategies:

    1. tail (small M): one full-size GEMM, per-peer doorbells, one kernel that
       pulls the slice out of every peer's Y over NVLink.
    2. push (large M): world_size row-block GEMMs in rotated order, own block
       last; the copy engine pushes each finished block to its owner and
       signals, hidden under the later block GEMMs.

    Buffers and signals cycle across 3 slots by call index with monotonic
    sequence numbers, so steady state needs no cross-rank barrier.

Workspace:
    Symmetric buffers live in a caller-owned MatmulReduceScatterWorkspace:
    create once with the maximum token count, reuse for any M <= max_M,
    destroy() when done.

Routing:
    - Triton implementation (matmul_reduce_scatter_triton) on all architectures;
      a cuTile fast path for SM >= 100 may be added later.

Example (run with torchrun or mp.spawn across all GPU ranks)::

    import torch
    import torch.distributed as dist
    from flashinfer.comm import matmul_reduce_scatter, MatmulReduceScatterWorkspace

    # --- per-rank setup ---
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    group = dist.group.WORLD

    # --- inputs ---
    M, K, N = 16384, 8192, 2048  # M must be divisible by world_size
    dtype = torch.bfloat16
    inp = torch.randn(M, K, device=device, dtype=dtype)
    w   = torch.randn(K, N, device=device, dtype=dtype)

    # --- workspace (collective; allocate once, reuse for any M <= max_M) ---
    workspace = MatmulReduceScatterWorkspace(group, max_M=M, N=N, dtype=dtype)

    # --- fused matmul + reduce-scatter ---
    # out shape: (M // world_size, N)
    out = matmul_reduce_scatter(inp, w, group, workspace)

    workspace.destroy()
"""

from typing import Literal

import torch
import torch.distributed as dist

from flashinfer.utils import register_custom_op
from .matmul_reduce_scatter_triton import (
    MatmulReduceScatterWorkspace as MatmulReduceScatterWorkspace,
)
from .matmul_reduce_scatter_triton import matmul_reduce_scatter_triton


@register_custom_op(
    "flashinfer::matmul_reduce_scatter",
    mutates_args=[],
)
def matmul_reduce_scatter(
    inp: torch.Tensor,
    w: torch.Tensor,
    group: dist.ProcessGroup,
    workspace: MatmulReduceScatterWorkspace,
    *,
    verbose: bool = False,
    strategy: Literal["auto", "tail", "push"] = "auto",
):
    """Compute-signal/pull-reduce matmul + reduce-scatter; dispatches to Triton.
    ``strategy="auto"`` picks tail/push by token count (crossover 8192)."""
    return matmul_reduce_scatter_triton(
        inp, w, group, workspace, verbose=verbose, strategy=strategy
    )
