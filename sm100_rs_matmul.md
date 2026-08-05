# SM100 matmul + reduce-scatter calibration

## Result

For BF16 `input[M,8192] @ weight[8192,2048]` on four B200 GPUs:

- `strategy="auto"` uses **tail below 10,752 tokens** and **push at or above
  10,752 tokens**.
- The fused operation is faster than matmul + NCCL reduce-scatter through about
  2K tokens, slower around 3K-8K, and faster again from 10,752 tokens upward.
- The cuTile reduce remains a non-persistent `(16, 256)` tile. Persistent grids
  capped at the 152-SM count were slower overall.
- SM100 uses this one threshold value; Hopper keeps its existing
  world-size-specific thresholds.

These thresholds are shape-specific. Recalibrate if `K`, `N`, dtype, transport,
cuTile/compiler version, or topology changes.

## Settings and methodology

| Setting | Value |
|---|---|
| GPUs | 4 x NVIDIA GB200/B200, SM100, 152 SMs/GPU |
| Topology | NVLink 5; `nvidia-smi topo -m` reports `NV18` between every GPU pair |
| Driver | 580.105.08 |
| PyTorch | 2.11.0+cu130 |
| CUDA | 13.0 |
| cuda-tile | 1.5.0 |
| TileIRAS | 13.3.36 |
| dtype | BF16 |
| shape | `M x 8192` times `8192 x 2048` |
| transport | PyTorch symmetric memory, copy engine (`CE`) |
| timing | CUDA events; 20 warmups and 300 timed calls per implementation |
| ordering | Tail, push, and reference order rotated every iteration to remove clock/order bias |
| aggregation | `avg_mean`: mean of per-rank means; `max_mean`: slowest rank's mean |
| reference | `torch.matmul` followed by `torch.distributed.reduce_scatter_tensor` |

The checked-in 50-iteration benchmark was also run twice. Its 16K-64K and
4K-8K conclusions matched the longer run. Results at 1K-2K varied materially,
so the 300-iteration counterbalanced run below is authoritative for that range.

| Tokens | 50-iteration speedup, run 1 | run 2 |
|---:|---:|---:|
| 65,536 | 1.17x | 1.18x |
| 32,768 | 1.09x | 1.06x |
| 16,384 | 1.04x | 1.05x |
| 8,192 | 0.93x | 0.92x |
| 4,096 | 0.92x | 0.93x |
| 2,048 | 0.89x | 1.35x (outlier) |
| 1,024 | 1.00x | 1.01x |

## Fused versus NCCL

`overlap` is the strategy selected by the calibrated four-GPU `auto` policy.
Times are milliseconds.

```text
--------------------------------------------------------------------------------
         |           overlap           |             ref             |  speedup
# tokens |      avg_mean      max_mean |      avg_mean      max_mean |       avg
--------------------------------------------------------------------------------
   65536 |         1.628         1.688 |         1.933         1.959 |      1.19x
   57344 |         1.367         1.442 |         1.741         1.777 |      1.27x
   49152 |         1.228         1.270 |         1.440         1.464 |      1.17x
   40960 |         1.022         1.061 |         1.191         1.214 |      1.17x
   32768 |         0.910         0.949 |         0.979         1.004 |      1.08x
   24576 |         0.677         0.718 |         0.765         0.794 |      1.13x
   16384 |         0.477         0.505 |         0.540         0.560 |      1.13x
   12288 |         0.387         0.408 |         0.427         0.445 |      1.10x
   10752 |         0.372         0.380 |         0.380         0.393 |      1.02x
    8192 |         0.326         0.329 |         0.308         0.321 |      0.95x
    6144 |         0.272         0.275 |         0.259         0.274 |      0.95x
    4096 |         0.215         0.217 |         0.205         0.225 |      0.95x
    3072 |         0.185         0.186 |         0.180         0.201 |      0.97x
    2048 |         0.163         0.165 |         0.174         0.190 |      1.07x
    1536 |         0.160         0.164 |         0.167         0.179 |      1.04x
    1024 |         0.159         0.163 |         0.164         0.172 |      1.03x
     768 |         0.159         0.163 |         0.165         0.171 |      1.04x
     512 |         0.153         0.157 |         0.165         0.177 |      1.07x
--------------------------------------------------------------------------------
```

## Tail/push crossover

SM100 uses the single measured four-GPU threshold `10752`. The H100 values
remain `{2: 3072, 4: 6144, 8: 12288}`. A two-GPU exploratory sweep found a
5,120-token crossover, recorded below, but it is intentionally not another
production routing value.

Two GPUs, 300 counterbalanced iterations:

```text
      M   tail_avg tail_max   push_avg push_max    ref_avg  ref_max
   4096     0.1958   0.1970     0.2113   0.2123     0.1841   0.1846
   4352     0.1969   0.1972     0.2105   0.2145     0.1906   0.1951
   4608     0.1996   0.2000     0.2129   0.2167     0.1929   0.1976
   4864     0.2012   0.2019     0.2128   0.2165     0.1966   0.2015
   5120     0.2216   0.2221     0.2124   0.2173     0.2132   0.2197
   5376     0.2168   0.2170     0.2127   0.2177     0.2113   0.2168
   5632     0.2245   0.2246     0.2117   0.2174     0.2204   0.2266
   6144     0.2451   0.2459     0.2249   0.2312     0.2386   0.2448
```

Four GPUs, 300 counterbalanced iterations:

```text
      M   tail_avg tail_max   push_avg push_max    ref_avg  ref_max
  10240     0.3633   0.3675     0.3674   0.3782     0.3529   0.3628
  10752     0.3826   0.3866     0.3595   0.3676     0.3708   0.3791
  11264     0.4010   0.4048     0.3629   0.3759     0.3953   0.4055
  11776     0.4210   0.4269     0.3656   0.3838     0.4172   0.4333
  12288     0.4349   0.4389     0.3845   0.4039     0.4275   0.4453
```

## Reduce tile and grid

A preliminary sweep covered `tile_m={8,16,32,64}` and
`tile_n={128,256,512}` with full and SM-count-capped grids. The table below is
the 100-iteration, order-rotated shortlist. `(16,256)` non-persistent was kept
because it was balanced across the range and best on the important 32K push
case. The apparent small-shape differences among non-persistent tiles are only
a few microseconds.

```text
tile_m tile_n persistent      M strategy   avg_ms   max_ms
    16    256      False   1024     tail   0.1390   0.1428
    16    512      False   1024     tail   0.1351   0.1381
    32    256      False   1024     tail   0.1347   0.1384
    32    512      False   1024     tail   0.1353   0.1386
    32    512       True   1024     tail   0.1488   0.1518
    16    256      False   2048     tail   0.1437   0.1480
    16    512      False   2048     tail   0.1434   0.1463
    32    256      False   2048     tail   0.1432   0.1459
    32    512      False   2048     tail   0.1453   0.1482
    32    512       True   2048     tail   0.1456   0.1486
    16    256      False   8192     tail   0.3140   0.3168
    16    512      False   8192     tail   0.3139   0.3173
    32    256      False   8192     tail   0.3136   0.3170
    32    512      False   8192     tail   0.3159   0.3195
    32    512       True   8192     tail   0.3178   0.3205
    16    256      False  12288     push   0.3875   0.3916
    16    512      False  12288     push   0.3864   0.3894
    32    256      False  12288     push   0.3882   0.3895
    32    512      False  12288     push   0.3901   0.3919
    32    512       True  12288     push   0.3918   0.3947
    16    256      False  32768     push   0.9312   0.9342
    16    512      False  32768     push   0.9331   0.9345
    32    256      False  32768     push   0.9335   0.9360
    32    512      False  32768     push   0.9422   0.9436
    32    512       True  32768     push   0.9437   0.9456
    16    256      False  65536     tail   1.8387   1.8418
    16    512      False  65536     tail   1.8371   1.8400
    32    256      False  65536     tail   1.8385   1.8419
    32    512      False  65536     tail   1.8398   1.8435
    32    512       True  65536     tail   1.8455   1.8501
```

## Workspace and correctness

The SM100 cuTile path uses the caller-owned `MatmulReduceScatterWorkspace`.
It reuses three symmetric-memory slots containing GEMM intermediates, push
receive buffers, signal pads, remote views, a communication stream, and events.
The final output tensor is allocated per call. For `max_M=65536`, `N=2048`,
world size 4, and BF16, the symmetric buffers occupy about 1.31 GiB per GPU.

Validation performed on both four- and two-GPU configurations:

- Pytest BF16 and FP16: `2 passed` on four GPUs and `2 passed` on two GPUs.
- Forced delayed producer to exercise `atomic_add(+0, acquire.sys)` and the
  `cuStreamWriteValue32` publication pairing.
- Eight queued alternating tail/push calls, workspace slot reuse, and partial
  reduce tiles.
- Extended BF16 sweep: five sizes, both strategies, ten repetitions per size.
- Ruff lint/format, Python compile check, and `git diff --check`: passed.

## Reproduction script

Save the following as `bench_sm100_rs.py` in the repository root. It reproduces
the counterbalanced tail/push/reference measurements above against the current
production kernel. The tile/grid table used a temporary configurable launcher
and persistent stride loop that were removed after tuning; the production
script intentionally does not mutate the selected tile or grid.

```python
import argparse
import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from flashinfer.comm import MatmulReduceScatterWorkspace, matmul_reduce_scatter


K, N = 8192, 2048
DEFAULT_SIZES = [
    512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 10752,
    12288, 16384, 24576, 32768, 40960, 49152, 57344, 65536,
]


def free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("localhost", 0))
        return sock.getsockname()[1]


def reference(inp, weight, group):
    world_size = dist.get_world_size(group)
    partial = inp @ weight
    out = torch.empty(
        inp.shape[0] // world_size,
        weight.shape[1],
        dtype=inp.dtype,
        device=inp.device,
    )
    dist.reduce_scatter_tensor(out, partial, group=group)
    return out


def worker(rank, world_size, port, sizes, warmup, iterations):
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=world_size,
        device_id=device,
    )
    group = dist.group.WORLD
    torch.manual_seed(100 + rank)
    weight = torch.randn(K, N, dtype=torch.bfloat16, device=device)

    with MatmulReduceScatterWorkspace(
        group, max(sizes), N, dtype=torch.bfloat16, device=device
    ) as workspace:
        clock_input = torch.randn(16384, K, dtype=torch.bfloat16, device=device)
        matmul_reduce_scatter(
            clock_input, weight, group, workspace, strategy="tail"
        )
        for _ in range(30):
            torch.mm(clock_input, weight)
        torch.cuda.synchronize(device)

        if rank == 0:
            print("      M   tail_avg tail_max   push_avg push_max    ref_avg  ref_max")

        for m in sizes:
            inp = torch.randn(m, K, dtype=torch.bfloat16, device=device)
            functions = {
                "tail": lambda: matmul_reduce_scatter(
                    inp, weight, group, workspace, strategy="tail"
                ),
                "push": lambda: matmul_reduce_scatter(
                    inp, weight, group, workspace, strategy="push"
                ),
                "ref": lambda: reference(inp, weight, group),
            }
            names = list(functions)
            for name in names:
                for _ in range(warmup):
                    functions[name]()
            torch.cuda.synchronize(device)
            dist.barrier(group)

            samples = {name: [] for name in names}
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            for iteration in range(iterations):
                offset = iteration % len(names)
                for name in names[offset:] + names[:offset]:
                    start.record()
                    functions[name]()
                    end.record()
                    end.synchronize()
                    samples[name].append(start.elapsed_time(end))

            results = []
            for name in names:
                local = torch.tensor(
                    [sum(samples[name]) / iterations],
                    dtype=torch.float64,
                    device=device,
                )
                gathered = [torch.empty_like(local) for _ in range(world_size)]
                dist.all_gather(gathered, local, group=group)
                if rank == 0:
                    values = torch.cat(gathered).cpu()
                    results.append((values.mean().item(), values.max().item()))
            if rank == 0:
                print(
                    f"{m:7d} "
                    + " ".join(f"{value:8.4f}" for pair in results for value in pair)
                )

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    args = parser.parse_args()
    if torch.cuda.device_count() < args.world_size:
        raise RuntimeError("not enough visible CUDA devices")
    mp.spawn(
        worker,
        args=(
            args.world_size,
            free_port(),
            args.sizes,
            args.warmup,
            args.iterations,
        ),
        nprocs=args.world_size,
        join=True,
    )
```

Commands:

```bash
# Full four-GPU sweep.
PYTHONPATH=. /home/nvidia/venvs/ray-vllm/bin/python bench_sm100_rs.py

# Four-GPU threshold zoom.
PYTHONPATH=. /home/nvidia/venvs/ray-vllm/bin/python bench_sm100_rs.py \
  --sizes 10240 10752 11264 11776 12288

# Two-GPU threshold zoom.
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. \
  /home/nvidia/venvs/ray-vllm/bin/python bench_sm100_rs.py --world-size 2 \
  --sizes 4096 4352 4608 4864 5120 5376 5632 6144

# Checked-in concise auto-vs-reference table.
PYTHONPATH=. /home/nvidia/venvs/ray-vllm/bin/python \
  tests/comm/test_matmul_reduce_scatter.py --benchmark --dtype bfloat16

# Correctness and lint commands used.
PYTHONPATH=. /home/nvidia/venvs/ray-vllm/bin/python -m pytest \
  tests/comm/test_matmul_reduce_scatter.py -q
/home/nvidia/venvs/ray-vllm/bin/python -m ruff check \
  flashinfer/comm/matmul_reduce_scatter tests/comm/test_matmul_reduce_scatter.py
/home/nvidia/venvs/ray-vllm/bin/python -m ruff format --check \
  flashinfer/comm/matmul_reduce_scatter tests/comm/test_matmul_reduce_scatter.py
```
