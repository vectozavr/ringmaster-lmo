import argparse
import json
import os

import matplotlib

if os.environ.get("DISPLAY"):
    matplotlib.use("TkAgg")
else:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


sns.set(style="whitegrid", context="talk", font_scale=1.1, palette=sns.color_palette("bright"), color_codes=False)
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = "DejaVu Sans"
matplotlib.rcParams["mathtext.fontset"] = "cm"
matplotlib.rcParams["figure.figsize"] = (10, 7)
matplotlib.rcParams["text.usetex"] = False


DEFAULT_TRACE_PATH = os.path.join(os.path.dirname(__file__), "ringmaster_nanochat_traces.json")
DEFAULT_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "ringmaster_nanochat_replot.pdf")

COLORS = {
    "RingmasterMuonASGD": "#C00000",
    "ParameterAgnosticRingmasterMuonASGD": "#B8860B",
    "RennalaMuonSGD": "#006666",
    "DelayAdaptiveMuonASGD": "#4682B4",
}

MARKERS = {
    "RingmasterMuonASGD": "*",
    "ParameterAgnosticRingmasterMuonASGD": "^",
    "RennalaMuonSGD": "H",
    "DelayAdaptiveMuonASGD": "o",
}

MARKER_SIZES = {
    "RingmasterMuonASGD": 18,
    "ParameterAgnosticRingmasterMuonASGD": 16,
    "RennalaMuonSGD": 16,
    "DelayAdaptiveMuonASGD": 14,
}

PLOTTING_ORDER = [
    "RingmasterMuonASGD",
    "ParameterAgnosticRingmasterMuonASGD",
    "RennalaMuonSGD",
    "DelayAdaptiveMuonASGD",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot Nanochat async comparison curves from a saved trace JSON."
    )
    parser.add_argument(
        "--trace-file",
        default=DEFAULT_TRACE_PATH,
        help="Path to the saved trace JSON file.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help="Path to the output plot file.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional custom title for the plot.",
    )
    parser.add_argument(
        "--yscale",
        choices=["log", "linear"],
        default="log",
        help="Y-axis scale.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=1,
        help="Optional moving-average smoothing window. Use 1 to disable smoothing.",
    )
    parser.add_argument(
        "--max-time",
        type=float,
        default=None,
        help="Optional x-axis cutoff. Defaults to the time limit stored in the trace file.",
    )
    return parser.parse_args()


def moving_average(values, window):
    if window <= 1:
        return np.asarray(values, dtype=np.float64)
    window = min(window, len(values))
    kernel = np.ones(window, dtype=np.float64) / window
    smoothed = np.convolve(np.asarray(values, dtype=np.float64), kernel, mode="valid")
    prefix = np.asarray(values[: window - 1], dtype=np.float64)
    return np.concatenate([prefix, smoothed])


def load_trace_payload(trace_file):
    with open(trace_file, "r", encoding="utf-8") as source:
        return json.load(source)


def build_title(payload, custom_title=None):
    if custom_title is not None:
        return custom_title
    title_suffix = payload.get("_meta", {}).get("title_suffix", "")
    title = "Nanochat Muon Async Training Comparison"
    if title_suffix:
        title = f"{title}: {title_suffix}"
    return title


def main():
    args = parse_args()
    payload = load_trace_payload(args.trace_file)
    traces_by_method = payload["traces_by_method"]
    time_limit = args.max_time
    if time_limit is None:
        time_limit = payload.get("_meta", {}).get("time_limit")

    plt.figure()
    plot_fn = plt.semilogy if args.yscale == "log" else plt.plot

    for method_name in PLOTTING_ORDER:
        if method_name not in traces_by_method:
            continue
        trace = traces_by_method[method_name]
        runtime = np.asarray(trace["runtime"], dtype=np.float64)
        loss = moving_average(trace["latest_train_loss"], args.smooth_window)

        plot_fn(
            runtime,
            loss,
            label=trace["label"],
            linestyle="solid",
            marker=MARKERS.get(method_name, "o"),
            markersize=MARKER_SIZES.get(method_name, 12),
            markevery=max(1, len(runtime) // 10),
            color=COLORS.get(method_name),
        )

    plt.legend(loc="upper right", prop={"size": 14})
    if time_limit is not None:
        plt.xlim(0, time_limit)
    plt.xlabel("Runtime (seconds)")
    plt.ylabel("Latest Minibatch Loss")
    plt.title(build_title(payload, custom_title=args.title))
    plt.tight_layout()
    plt.savefig(args.output, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
