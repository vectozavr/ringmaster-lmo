import argparse
import json
import os
import statistics
import time

import torch

from nanochat_async import NanochatLanguageModelFunction


DEFAULT_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "nanochat_gradient_timing.json")
DEFAULT_WARMUP_STEPS = 3
DEFAULT_MEASURE_STEPS = 10
DEFAULT_DEVICE_BATCH_SIZE = 2
DEFAULT_NUM_SHARDS = 10


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure Nanochat stochastic-gradient wall-clock time on the current GPU."
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help="Where to save the measured timing JSON.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=DEFAULT_WARMUP_STEPS,
        help="Number of warmup gradient calls before timing.",
    )
    parser.add_argument(
        "--measure-steps",
        type=int,
        default=DEFAULT_MEASURE_STEPS,
        help="Number of timed gradient calls.",
    )
    parser.add_argument(
        "--device-batch-size",
        type=int,
        default=DEFAULT_DEVICE_BATCH_SIZE,
        help="Training batch size used for the timing run.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=DEFAULT_NUM_SHARDS,
        help="Number of autoresearch shards to prepare/use.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for the timing run.",
    )
    parser.add_argument(
        "--compile-model",
        action="store_true",
        help="Enable torch.compile during the timing run.",
    )
    return parser.parse_args()


def synchronize_if_needed(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for gradient timing, but torch.cuda.is_available() is False.")

    function = NanochatLanguageModelFunction(
        seed=args.seed,
        is_cuda=True,
        num_shards=args.num_shards,
        device_batch_size=args.device_batch_size,
        compile_model=args.compile_model,
    )
    point = function.get_current_point()
    num_parameters = function.dim()

    print("Nanochat config:", function.parameter_metadata()["model_config"])
    print(f"Model size: {num_parameters:,} parameters ({num_parameters / 1e6:.2f}M)")
    print(f"Device: {function.device}")
    print(f"Warmup steps: {args.warmup_steps}")
    print(f"Measure steps: {args.measure_steps}")
    print(f"Device batch size: {args.device_batch_size}")

    for warmup_step in range(args.warmup_steps):
        synchronize_if_needed(function.device)
        function.gradient(point)
        synchronize_if_needed(function.device)
        print(f"Warmup {warmup_step + 1}/{args.warmup_steps}")

    durations = []
    for measure_step in range(args.measure_steps):
        synchronize_if_needed(function.device)
        start_time = time.perf_counter()
        function.gradient(point)
        synchronize_if_needed(function.device)
        duration = time.perf_counter() - start_time
        durations.append(duration)
        print(f"Step {measure_step + 1}/{args.measure_steps}: {duration:.6f}s")

    mean_seconds = float(statistics.mean(durations))
    median_seconds = float(statistics.median(durations))
    std_seconds = float(statistics.pstdev(durations)) if len(durations) > 1 else 0.0

    payload = {
        "mean_seconds": mean_seconds,
        "median_seconds": median_seconds,
        "std_seconds": std_seconds,
        "durations_seconds": [float(duration) for duration in durations],
        "device": str(function.device),
        "device_name": torch.cuda.get_device_name(function.device) if function.device.type == "cuda" else "cpu",
        "device_batch_size": args.device_batch_size,
        "num_shards": args.num_shards,
        "seed": args.seed,
        "compile_model": bool(args.compile_model),
        "num_parameters": int(num_parameters),
        "model_config": function.parameter_metadata()["model_config"],
    }

    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, sort_keys=True)

    print("---")
    print(f"Mean gradient time:   {mean_seconds:.6f}s")
    print(f"Median gradient time: {median_seconds:.6f}s")
    print(f"Std gradient time:    {std_seconds:.6f}s")
    print(f"Saved timing to {args.output}")


if __name__ == "__main__":
    main()
