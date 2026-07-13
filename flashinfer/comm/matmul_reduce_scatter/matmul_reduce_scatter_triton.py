# Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Triton implementation of compute-signal/pull-reduce matmul + reduce-scatter. For details of the algorithm, see matmul_reduce_scatter.py."""

import warnings
from typing import Literal, Optional

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import triton
import triton.language as tl

from flashinfer.utils import get_device_sm_count

from ..all_gather_matmul.configs import Configs

# Buffer/signal slots cycled by call index; tolerates _NUM_SLOTS - 1 calls of
# cross-rank drift without a barrier.
_NUM_SLOTS = 3

# tail/push crossover, measured on 4xH100 (K=8192, N=2048, ws=4).
# TODO(shape-aware): route on the GEMM/comm ratio (shifts with K, N, world_size).
_PUSH_MIN_TOKENS = 8192


# do_not_specialize: a monotonic seq would hit Triton's value%16 int
# specialization and recompile mid-serving.
@triton.jit(do_not_specialize=["seq"])
def wait_reduce_triton_kernel(
    srcs,  # tuple of (CHUNK_M, N) partial views; signal-arrival order, own last
    signal_ptr,  # local signal pad; slot i gates srcs[i]
    out_ptr,
    seq,
    CHUNK_M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_SRC: tl.constexpr,
):
    """Spin-wait per source, then sum this rank's (CHUNK_M, N) slice in fp32.
    The last source is the caller's own partial: stream-ordered, no wait.
    Volatile loads: sources may be remote."""
    num_tiles_n = tl.cdiv(N, BLOCK_N)
    total_tiles = tl.cdiv(CHUNK_M, BLOCK_M) * num_tiles_n
    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)

    tile_id = pid
    while tile_id < total_tiles:
        offs_m = (tile_id // num_tiles_n) * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = (tile_id % num_tiles_n) * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = (offs_m[:, None] < CHUNK_M) & (offs_n[None, :] < N)
        idx = offs_m[:, None].to(tl.int64) * N + offs_n[None, :]

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for i in tl.static_range(NUM_SRC):
            if i < NUM_SRC - 1:
                while tl.load(signal_ptr + i, volatile=True) < seq:
                    pass
            acc += tl.load(srcs[i] + idx, mask=mask, volatile=True).to(tl.float32)
        tl.store(out_ptr + idx, acc.to(out_ptr.dtype.element_ty), mask=mask)

        tile_id += num_programs


class MatmulReduceScatterWorkspace:
    """Symmetric-memory workspace for matmul_reduce_scatter."""

    def __init__(
        self,
        group: dist.ProcessGroup,
        max_M: int,
        N: int,
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
    ):
        Configs.initialize()
        if Configs.TRANSFER != "CE":
            raise NotImplementedError(
                "matmul_reduce_scatter only implements the copy-engine transport"
            )
        self.group = group
        self.world_size = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        if max_M % self.world_size != 0:
            raise ValueError("max_M must be divisible by world_size")
        self.max_M = max_M
        self.N = N
        self.dtype = dtype
        self.device = device or torch.device("cuda", torch.cuda.current_device())
        self.seq = 0
        self._destroyed = False

        ws, rank = self.world_size, self.rank
        max_chunk = max_M // ws
        self.slots = []
        for _ in range(_NUM_SLOTS):
            y = symm_mem.empty(max_M, N, device=self.device, dtype=dtype)
            hdl = symm_mem.rendezvous(y, group.group_name)
            # Pad slot i <- source (rank - 1 - i) % ws (signal-arrival order).
            signal_pad = hdl.get_signal_pad(rank, (ws,), Configs.SIGNAL_DTYPE, 0)
            y_remote = [
                hdl.get_remote_tensor((rank - 1 - i) % ws, y.shape, y.dtype)
                for i in range(ws - 1)
            ]
            # push receive buffer: slot i <- arrival-order source i.
            recv = symm_mem.empty(ws - 1, max_chunk, N, device=self.device, dtype=dtype)
            recv_hdl = symm_mem.rendezvous(recv, group.group_name)
            # Outgoing: block p -> peer_recv[p], then bump peer p's pad.
            peer_recv = [None] * ws
            peer_signal = [None] * ws
            peer_arrival = [0] * ws
            for p in range(ws):
                if p == rank:
                    continue
                peer_arrival[p] = (p - rank - 1) % ws
                peer_recv[p] = recv_hdl.get_remote_tensor(p, recv.shape, recv.dtype)[
                    peer_arrival[p]
                ]
                peer_signal[p] = hdl.get_signal_pad(p, (ws,), Configs.SIGNAL_DTYPE, 0)
            self.slots.append(
                dict(
                    y=y,
                    signal_pad=signal_pad,
                    y_remote=y_remote,
                    recv=recv,
                    peer_recv=peer_recv,
                    peer_signal=peer_signal,
                    peer_arrival=peer_arrival,
                    push_read_done=torch.cuda.Event(),
                )
            )

        self.comm_stream = torch.cuda.Stream()
        self.chunk_events = [torch.cuda.Event() for _ in range(ws)]
        # Persistent reduce grid: one CTA per SM (all_gather_matmul convention).
        self.num_sms = get_device_sm_count(self.device)
        # Per-M source/destination views (cheap tensor views, keyed by M).
        self._views: dict = {}

    def _sources(self, M):
        """Per-slot (pull_srcs, push_srcs, peer_recv) views for token count M."""
        views = self._views.get(M)
        if views is None:
            chunk = M // self.world_size
            row0 = self.rank * chunk
            views = []
            for slot in self.slots:
                y_own = slot["y"][row0 : row0 + chunk]
                pull_srcs = tuple(t[row0 : row0 + chunk] for t in slot["y_remote"]) + (
                    y_own,
                )
                push_srcs = tuple(
                    slot["recv"][i][:chunk] for i in range(self.world_size - 1)
                ) + (y_own,)
                peer_recv = [
                    None if t is None else t[:chunk] for t in slot["peer_recv"]
                ]
                views.append((pull_srcs, push_srcs, peer_recv))
            self._views[M] = views
        return views

    def destroy(self) -> None:
        """Release the symmetric buffers. Idempotent."""
        self.slots = []
        self._views = {}
        self._destroyed = True

    def __del__(self):
        if not self._destroyed:
            warnings.warn(
                f"{self.__class__.__name__} was not explicitly destroyed. "
                f"Call workspace.destroy() to ensure deterministic cleanup of "
                f"symmetric-memory resources.",
                ResourceWarning,
                stacklevel=2,
            )


def matmul_reduce_scatter_triton(
    inp: torch.Tensor,
    w: torch.Tensor,
    group: dist.ProcessGroup,
    workspace: MatmulReduceScatterWorkspace,
    *,
    verbose: bool = False,
    strategy: Literal["auto", "tail", "push"] = "auto",
):
    """Compute-signal/pull-reduce matmul + reduce-scatter; returns this rank's
    (M // world_size, N) slice. See matmul_reduce_scatter.py for details."""
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
        strategy = "push" if M >= _PUSH_MIN_TOKENS else "tail"
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

    tile_m, tile_n = 32, 256  # reduce tile; bandwidth-bound, larger adds nothing
    num_tiles = triton.cdiv(chunk_rows, tile_m) * triton.cdiv(N, tile_n)
    grid = (min(num_tiles, workspace.num_sms),)

    def wait_reduce(srcs):
        wait_reduce_triton_kernel[grid](
            srcs,
            slot["signal_pad"],
            out,
            seq,
            chunk_rows,
            N,
            BLOCK_M=tile_m,
            BLOCK_N=tile_n,
            NUM_SRC=world_size,
        )
        return out

    if verbose and rank == 0:
        print(
            f"matmul_reduce_scatter_triton: M={M}, N={N}, K={K}, "
            f"world_size={world_size}, chunk_rows={chunk_rows}, "
            f"strategy={strategy}, seq={seq}, grid={grid}"
        )

    main_stream = torch.cuda.current_stream()

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
