# Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
cuTile implementation of compute-signal/pull-reduce matmul + reduce-scatter. Requires SM >= 100 (Blackwell+).

See matmul_reduce_scatter.py for the public routing entry point and full algorithm description.
"""

from typing import Literal

import torch
import torch.distributed as dist
import cuda.tile as ct

from .matmul_reduce_scatter_triton import (
    _NUM_SLOTS,
    _push_min_tokens,
    MatmulReduceScatterWorkspace,
)


# cuTile kernel: spin-wait per source, then reduce this rank's tile in fp32.
# seq is a runtime (non-Constant) scalar so a monotonic value does not
# specialize and recompile mid-serving (Triton's do_not_specialize analogue).
@ct.kernel
def wait_reduce_cutile_kernel(
    srcs,  # list of (CHUNK_M, N) partial views; signal-arrival order, own last
    signal_pad,  # local signal pad; slot i gates srcs[i]
    out,
    seq,
    tile_m: ct.Constant[int],
    tile_n: ct.Constant[int],
    num_src: ct.Constant[int],
):
    num_tiles_n = ct.cdiv(out.shape[1], tile_n)
    bid = ct.bid(0)
    m_idx = bid // num_tiles_n
    n_idx = bid % num_tiles_n

    acc = ct.full((tile_m, tile_n), 0, dtype=ct.float32)
    for i in range(num_src):
        # Spin-wait for the producer's data-ready signal; the own partial
        # (last) is stream-ordered, no wait. atomic_add(+0) does nothing by
        # itself: it just keeps cuda.tile (as of 1.5.0) from optimizing the
        # spin loop away. acquire.sys pairs with the producer's
        # cuStreamWriteValue32 release.
        if i < num_src - 1:
            sig = ct.atomic_add(
                signal_pad,
                (i,),
                0,
                memory_order=ct.MemoryOrder.ACQUIRE,
                memory_scope=ct.MemoryScope.SYS,
            ).astype(ct.int64)
            while sig < seq:
                sig = ct.atomic_add(
                    signal_pad,
                    (i,),
                    0,
                    memory_order=ct.MemoryOrder.ACQUIRE,
                    memory_scope=ct.MemoryScope.SYS,
                ).astype(ct.int64)
        t = ct.load(
            srcs[i],
            index=(m_idx, n_idx),
            shape=(tile_m, tile_n),
            padding_mode=ct.PaddingMode.ZERO,
        )
        acc = acc + t.astype(ct.float32)
    ct.store(out, index=(m_idx, n_idx), tile=acc.astype(out.dtype))


# Launcher for compute-signal/pull-reduce matmul + reduce-scatter (cuTile / SM >= 100)
def matmul_reduce_scatter_cutile(
    inp: torch.Tensor,
    w: torch.Tensor,
    group: dist.ProcessGroup,
    workspace: MatmulReduceScatterWorkspace,
    *,
    verbose: bool = False,
    strategy: Literal["auto", "tail", "push"] = "auto",
):
    M, K = inp.shape
    N = w.shape[1]
    assert w.shape[0] == K, "reduction dimension mismatch"

    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    assert M % world_size == 0, (
        "inp.shape[0] (token count) must be divisible by world_size"
    )
    chunk_rows = M // world_size
    device = inp.device
    row0 = rank * chunk_rows

    if workspace._destroyed:
        raise RuntimeError("workspace has been destroyed")
    if workspace.group.group_name != group.group_name:
        raise ValueError("workspace was created for a different process group")
    if workspace.max_M < M or N != workspace.N or inp.dtype != workspace.dtype:
        raise ValueError(
            f"workspace (max_M={workspace.max_M}, N={workspace.N}, "
            f"dtype={workspace.dtype}) cannot serve inp {tuple(inp.shape)} "
            f"@ w {tuple(w.shape)} with dtype {inp.dtype}"
        )

    if strategy == "auto":
        push_min = _push_min_tokens(world_size)
        strategy = "push" if push_min <= M else "tail"
    elif strategy not in ("tail", "push"):
        raise ValueError(f"strategy must be 'auto', 'tail' or 'push', got {strategy}")

    workspace.seq += 1
    seq = workspace.seq
    slot_idx = seq % _NUM_SLOTS
    slot = workspace.slots[slot_idx]
    y = slot["y"]
    peer_signal = slot["peer_signal"]
    peer_arrival = slot["peer_arrival"]
    pull_srcs, push_srcs, peer_recv = workspace._sources(M)[slot_idx]

    out = torch.empty(chunk_rows, N, device=device, dtype=inp.dtype)

    def gemm_block(r0, r1):
        torch.matmul(inp[r0:r1], w, out=y[r0:r1])

    tile_m, tile_n = 16, 256  # reduce tile; bandwidth-bound, larger adds nothing
    # One tile per block; a non-persistent grid pipelines better than a
    # persistent loop under cuTile.
    grid = (ct.cdiv(chunk_rows, tile_m) * ct.cdiv(N, tile_n),)

    main_stream = torch.cuda.current_stream()

    def wait_reduce(srcs):
        ct.launch(
            main_stream,
            grid,
            wait_reduce_cutile_kernel,
            (
                list(srcs),
                slot["signal_pad"],
                out,
                seq,
                tile_m,
                tile_n,
                world_size,
            ),
        )
        return out

    if verbose and rank == 0:
        print(
            f"matmul_reduce_scatter_cutile: M={M}, N={N}, K={K}, "
            f"world_size={world_size}, chunk_rows={chunk_rows}, "
            f"strategy={strategy}, seq={seq}, grid={grid}"
        )

    if strategy == "tail":
        gemm_block(0, M)
        # cuStreamWriteValue32's memory barrier publishes the GEMM's writes first.
        for p in range(world_size):
            if p != rank:
                torch.ops.symm_mem.stream_write_value32_(
                    peer_signal[p], peer_arrival[p], seq
                )
        return wait_reduce(pull_srcs)

    # Push: GEMMs back-to-back first, then copy-engine pushes gated per block.
    comm_stream = workspace.comm_stream
    chunk_events = workspace.chunk_events
    # Call seq - _NUM_SLOTS may still have pushes reading this Y; wait before
    # overwriting it.
    main_stream.wait_event(slot["push_read_done"])
    for i in range(1, world_size):
        p = (rank + i) % world_size
        gemm_block(p * chunk_rows, (p + 1) * chunk_rows)
        chunk_events[p].record(main_stream)
    gemm_block(row0, row0 + chunk_rows)
    for i in range(1, world_size):
        p = (rank + i) % world_size
        comm_stream.wait_event(chunk_events[p])
        with torch.cuda.stream(comm_stream):
            peer_recv[p].copy_(y[p * chunk_rows : (p + 1) * chunk_rows])
            torch.ops.symm_mem.stream_write_value32_(
                peer_signal[p], peer_arrival[p], seq
            )
    # Pushes done reading Y; waited on by call seq + _NUM_SLOTS (the slot's
    # next user, see above).
    slot["push_read_done"].record(comm_stream)
    return wait_reduce(push_srcs)
