"""Measure parameters, MACs and latency of VanillaNet-7 before and after fusion.

Weights are random: none of these metrics depend on their values, so no
trained checkpoint is needed.

Usage:
    python scripts/benchmark.py                  # CPU, batch 1, fp32
    python scripts/benchmark.py --device cuda    # GPU, batch 1 and 64, fp32 and fp16 weights
"""

import argparse
import platform

import torch
import torch.utils.benchmark as benchmark
from torch.utils.flop_counter import FlopCounterMode

from vanillanet_ships import VanillaNet7
from vanillanet_ships.fusion import fuse_model

# 192x128 grayscale images, as in the competition
IMAGE_SHAPE = (1, 128, 192)
WARMUP_ITERS = 10


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def count_macs(forward, x):
    # FlopCounterMode intercepts every PyTorch operation run inside the block
    # and adds up the FLOPs of convolutions and matrix products (BatchNorm,
    # ReLU and pooling are negligible and not counted). 1 MAC = 2 FLOPs.
    with FlopCounterMode(display=False) as counter:
        forward(x)
    return counter.get_total_flops() // 2


DTYPES = {"fp32": torch.float32, "fp16": torch.float16}


def measure_latency_ms(run, x, threads, min_run_time):
    # CUDA kernels are launched asynchronously: without synchronize() we would
    # only time the launches, not the computation.
    stmt = "run(x); torch.cuda.synchronize()" if x.is_cuda else "run(x)"

    # Warm-up: lets cuDNN pick its fastest algorithms and fills the caches
    for _ in range(WARMUP_ITERS):
        run(x)

    # torch.utils.benchmark.Timer repeats the call until `min_run_time` seconds
    # have passed; we keep the median, which is robust to the occasional slow
    # run caused by the OS.
    timer = benchmark.Timer(stmt=stmt, globals={"run": run, "x": x, "torch": torch}, num_threads=threads)
    return timer.blocked_autorange(min_run_time=min_run_time).median * 1e3


def build_variants():
    """Return {name: (model, forward function)} for each model to compare."""
    torch.manual_seed(0)
    full = VanillaNet7(num_classes=10, width=1.0).eval()
    half = VanillaNet7(num_classes=10, width=0.5).eval()
    return {
        "Training-time (width 1.0)": (full, lambda x: full(x, lmbd=1.0)),
        "Fused (width 1.0)": (fused_full := fuse_model(full), fused_full),
        "Fused (width 0.5)": (fused_half := fuse_model(half), fused_half),
    }


def ratio(previous, current, higher_is_better=False):
    """Format the change from the previous row, e.g. ' (÷2.4)' or ' (×2.4)'."""
    if previous is None:
        return ""
    return f" (×{current / previous:.1f})" if higher_is_better else f" (÷{previous / current:.1f})"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--batch-sizes", type=int, nargs="+", help="default: 1 on CPU, 1 and 64 on GPU")
    parser.add_argument("--threads", type=int, default=torch.get_num_threads(), help="CPU threads used by PyTorch")
    parser.add_argument("--min-run-time", type=float, default=3.0, help="seconds spent timing each setting")
    args = parser.parse_args()

    on_gpu = args.device == "cuda"
    if on_gpu:
        if not torch.cuda.is_available():
            parser.error("CUDA is not available")
        torch.backends.cudnn.benchmark = True
        device_name = torch.cuda.get_device_name()
    else:
        device_name = platform.processor()

    batch_sizes = args.batch_sizes or ([1, 64] if on_gpu else [1])
    precisions = ["fp32", "fp16"] if on_gpu else ["fp32"]
    settings = [(b, p) for b in batch_sizes for p in precisions]

    variants = build_variants()
    rows = []
    with torch.inference_mode():
        for name, (model, forward) in variants.items():
            print(f"Measuring {name}...")
            # Parameters and MACs do not depend on the device: count them once, on CPU
            row = {
                "name": name,
                "params": count_parameters(model),
                "macs": count_macs(forward, torch.randn(1, *IMAGE_SHAPE)),
            }
            for batch_size, precision in settings:
                # fp16 converts the weights themselves (pure half-precision
                # inference); autocast would re-cast them at every call
                dtype = DTYPES[precision]
                model.to(args.device, dtype)
                x = torch.randn(batch_size, *IMAGE_SHAPE, device=args.device, dtype=dtype)
                row[batch_size, precision] = measure_latency_ms(forward, x, args.threads, args.min_run_time)
            model.to("cpu", torch.float32)
            if on_gpu:
                torch.cuda.empty_cache()
            rows.append(row)

    print()
    print(f"Device: {device_name} | threads: {args.threads} | torch {torch.__version__}")
    if on_gpu and torch.backends.cudnn.allow_tf32:
        # PyTorch default: "fp32" convolutions actually run in TF32 on tensor cores
        print("fp32 convolutions use TF32 (PyTorch default for cuDNN).")
    print("Each row is compared with the previous one: first fusion, then halving the width.")

    print()
    print("| Model | Params | MACs (per image) |")
    print("|---|---|---|")
    previous = None
    for row in rows:
        print(
            f"| {row['name']} | {row['params'] / 1e6:.1f}M{ratio(previous and previous['params'], row['params'])} "
            f"| {row['macs'] / 1e9:.2f}G{ratio(previous and previous['macs'], row['macs'])} |"
        )
        previous = row

    # Batch 1 is reported as latency (ms), larger batches as throughput (images/s)
    headers = [
        f"{'latency' if b == 1 else 'throughput'}, batch {b}, {p}" for b, p in settings
    ]
    print()
    print(f"| Model | {' | '.join(headers)} |")
    print(f"|---|{'---|' * len(settings)}")
    previous = None
    for row in rows:
        cells = []
        for batch_size, precision in settings:
            ms = row[batch_size, precision]
            prev_ms = previous and previous[batch_size, precision]
            if batch_size == 1:
                cells.append(f"{ms:.2f} ms{ratio(prev_ms, ms)}" if ms < 10 else f"{ms:.1f} ms{ratio(prev_ms, ms)}")
            else:
                speedup = ratio(prev_ms and 1 / prev_ms, 1 / ms, higher_is_better=True)
                cells.append(f"{batch_size / ms * 1e3:.0f} img/s{speedup}")
        print(f"| {row['name']} | {' | '.join(cells)} |")
        previous = row


if __name__ == "__main__":
    main()
