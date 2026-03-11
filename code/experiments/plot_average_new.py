# plot_average.py
import os
import glob
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

METHOD_PRETTY = {
    "ringleader": "Ringleader ASGD",
    "malenia":    "Malenia SGD",
    "iaasgd":     r"$\mathrm{IA^2SGD}$",
}
METHOD_COLOR = {
    "ringleader": "#B8860B",
    "malenia":    "#006666",
    "iaasgd":     "#C00000",
}

# ---------------- smoothing & utility helpers ----------------
def _ensure_odd(n: int) -> int:
    return n if (n % 2 == 1) else n + 1

def moving_avg(y, window_pts=101):
    """
    Centered moving average on a uniform index grid (linear space).
    O(n) via prefix sums. Window is made odd and capped to length.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n == 0:
        return y
    w = _ensure_odd(int(max(1, window_pts)))
    if w > n:
        w = n if (n % 2 == 1) else n - 1
        if w < 1:
            return y
    half = w // 2
    ps = np.empty(n + 1, dtype=float)
    ps[0] = 0.0
    ps[1:] = np.cumsum(y)
    out = np.empty_like(y)
    for k in range(n):
        L = max(0, k - half)
        R = min(n - 1, k + half)
        s = ps[R + 1] - ps[L]
        out[k] = s / (R - L + 1)
    return out

def moving_avg_log(y, window_pts=101, eps=1e-20):
    """
    Centered moving average in log-space (semilogy-friendly). Applies log(y+eps).
    """
    y = np.asarray(y, dtype=float)
    y_log = np.log(np.maximum(y, eps))
    y_log_s = moving_avg(y_log, window_pts=window_pts)
    return np.exp(y_log_s)

def ffill_zeros(y, floor):
    """
    Forward-fill values <= floor with the last positive sample, then floor.
    Useful if plotting on log scales and occasional zeros appear.
    """
    y = np.asarray(y, dtype=float).copy()
    mask = y <= floor
    if mask.any():
        valid_idx = np.nonzero(~mask)[0]
        last = floor
        if valid_idx.size:
            last = max(y[valid_idx[0]], floor)
        for i in range(y.size):
            if y[i] > floor:
                last = y[i]
            else:
                y[i] = last
    return np.maximum(y, floor)

# ---------------- I/O helpers ----------------
def load_runs(run_dir):
    """
    Returns:
      by_method: dict method -> list of dicts { "times":..., "loss":..., "grad2":..., "path":... }
      meta_any:  last seen meta (for reference)
    """
    paths = glob.glob(os.path.join(run_dir, "*.npz"))
    by_method = {}
    meta_any = None
    for p in paths:
        with np.load(p, allow_pickle=True) as z:
            times = z["times"]
            # loss may or may not exist depending on your saver
            loss = z["loss"] if "loss" in z.files else None
            grad2 = z["grads"] if "grads" in z.files else None
            meta = json.loads(str(z["meta"]))
            m = meta["method"]
            by_method.setdefault(m, []).append({
                "times": times,
                "loss": loss,
                "grad2": grad2,
                "path": p,
                "meta": meta,
            })
            meta_any = meta
    return by_method, meta_any

def make_time_grid(by_method, num_points=1001):
    # Choose a time grid that covers the max time across all runs
    t_max = 0.0
    for runs in by_method.values():
        for run in runs:
            t = run["times"]
            if t.size > 0:
                t_max = max(t_max, float(t[-1]))
    if t_max <= 0:
        t_max = 1.0
    return np.linspace(0.0, t_max, int(num_points))

def interp_on_grid(t, y, grid):
    # Forward-fill edges: use first/last values outside range
    y0 = float(y[0])
    yN = float(y[-1])
    return np.interp(grid, t, y, left=y0, right=yN)

# ---------------- main ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, default="runs",
                        help="Directory containing saved *.npz runs.")
    parser.add_argument("--num_points", type=int, default=1001,
                        help="Points on the common time grid.")
    parser.add_argument("--outfile", type=str, default=None,
                        help="Optional path to save the figure (e.g., avg_loss.png).")

    # What to plot
    parser.add_argument("--metric", choices=["loss", "grad2", "auto"], default="loss",
                        help="'loss' preferred; 'grad2' for gradient norm²; 'auto' uses loss if present else grad2.")

    # Per-run smoothing (before averaging)
    parser.add_argument("--smooth_window_pts", type=int, default=101,
                        help="Window size (points on grid) for moving average; made odd and capped by length.")
    parser.add_argument("--smooth_in_log", action="store_true",
                        help="Smooth in log-space (useful if you plan to plot on log scales).")

    # Robust aggregation & bands
    parser.add_argument("--band", choices=["std", "iqr", "p10p90", "none"], default="iqr",
                        help="Uncertainty band type across runs: std, interquartile range, 10–90%%, or none.")
    parser.add_argument("--geom_stats", action="store_true",
                        help="Aggregate in log-space (geometric stats). Often good for grad2; optional for loss.")

    # Y handling: zeros & scale
    parser.add_argument("--y_floor", type=float, default=1e-12,
                        help="Minimum positive floor to avoid zeros on log scales.")
    parser.add_argument("--ffill_zeros", action="store_true",
                        help="Forward-fill values <= y_floor before flooring (for log plots).")
    parser.add_argument("--scale", choices=["linear", "semilogy", "symlog", "log1p"], default="linear",
                        help="Y-axis style. For loss, linear is a nice default.")
    parser.add_argument("--symlog_linthresh", type=float, default=1e-10,
                        help="Linear threshold for symlog scale (around zero).")

    args = parser.parse_args()

    by_method, meta_any = load_runs(args.run_dir)
    if not by_method:
        raise SystemExit(f"No runs found in {args.run_dir}")

    # build grid
    grid = make_time_grid(by_method, num_points=args.num_points)

    fig, ax = plt.subplots(figsize=(10, 7))

    # choose which metric we'll actually use per run
    use_loss = (args.metric == "loss")
    use_grad2 = (args.metric == "grad2")
    auto = (args.metric == "auto")

    for method, runs in by_method.items():
        per_run = []
        used_metric = None  # keep a note for legend if you want

        for run in runs:
            t = run["times"]

            # pick metric for this run
            if auto:
                if run["loss"] is not None:
                    y_raw = run["loss"]; used_metric = "loss"
                elif run["grad2"] is not None:
                    y_raw = run["grad2"]; used_metric = "grad2"
                else:
                    # skip runs with neither
                    continue
            else:
                if use_loss:
                    if run["loss"] is None:
                        # skip if loss not available
                        continue
                    y_raw = run["loss"]; used_metric = "loss"
                else:  # grad2
                    if run["grad2"] is None:
                        continue
                    y_raw = run["grad2"]; used_metric = "grad2"

            # interpolate onto common grid
            y = interp_on_grid(t, y_raw, grid)

            # if plotting on a log-like scale, keep values positive & optionally ffill
            if args.scale in ("semilogy", "symlog", "log1p"):
                if args.ffill_zeros:
                    y = ffill_zeros(y, args.y_floor)
                else:
                    y = np.maximum(y, args.y_floor)

            # per-run smoothing (linear or log space)
            if args.smooth_in_log:
                y = moving_avg_log(y, window_pts=args.smooth_window_pts, eps=args.y_floor)
            else:
                y = moving_avg(y, window_pts=args.smooth_window_pts)

            per_run.append(y)

        if not per_run:
            # nothing usable for this method with current metric choice
            continue

        curves = np.stack(per_run, axis=0)  # (num_runs, T)

        # ---- aggregate across runs ----
        if args.geom_stats:
            logs = np.log(np.maximum(curves, args.y_floor))
            # pair center with band: median for quantile bands, mean for std
            if args.band in ("iqr", "p10p90"):
                center_log = np.quantile(logs, 0.50, axis=0)  # median
            else:  # "std" or "none"
                center_log = logs.mean(axis=0)               # mean of logs

            if args.band == "std":
                s_log = logs.std(axis=0)
                lower_log = center_log - s_log
                upper_log = center_log + s_log
            elif args.band == "iqr":
                lower_log = np.quantile(logs, 0.25, axis=0)
                upper_log = np.quantile(logs, 0.75, axis=0)
            elif args.band == "p10p90":
                lower_log = np.quantile(logs, 0.10, axis=0)
                upper_log = np.quantile(logs, 0.90, axis=0)
            else:
                lower_log = upper_log = center_log

            center = np.exp(center_log)
            lower  = np.exp(lower_log)
            upper  = np.exp(upper_log)
        else:
            # linear-space aggregation
            if args.band in ("iqr", "p10p90"):
                center = np.quantile(curves, 0.50, axis=0)  # median
            else:
                center = curves.mean(axis=0)                # mean

            if args.band == "std":
                s = curves.std(axis=0)
                lower = center - s
                upper = center + s
            elif args.band == "iqr":
                lower = np.quantile(curves, 0.25, axis=0)
                upper = np.quantile(curves, 0.75, axis=0)
            elif args.band == "p10p90":
                lower = np.quantile(curves, 0.10, axis=0)
                upper = np.quantile(curves, 0.90, axis=0)
            else:
                lower = upper = center

        # keep sensible values for log-like scales
        if args.scale in ("semilogy", "symlog", "log1p"):
            center = np.maximum(center, args.y_floor)
            lower  = np.maximum(lower,  args.y_floor)
            upper  = np.maximum(upper,  args.y_floor)

        label = METHOD_PRETTY.get(method, method)
        color = METHOD_COLOR.get(method, None)

        # ---- plot with chosen scale ----
        if args.scale == "semilogy":
            ax.semilogy(grid, center, label=label, color=color)
            ax.fill_between(grid, lower, upper, alpha=0.2, color=color)
            ax.set_ylabel("Training loss (center ± band)")
        elif args.scale == "symlog":
            ax.plot(grid, center, label=label, color=color)
            ax.fill_between(grid, lower, upper, alpha=0.2, color=color)
            ax.set_yscale("symlog", linthresh=args.symlog_linthresh)
            ax.set_ylabel("Training loss (center ± band)")
        elif args.scale == "log1p":
            eps = args.y_floor
            T = lambda x: np.log10(x + eps)
            ax.plot(grid, T(center), label=label, color=color)
            ax.fill_between(grid, T(lower), T(upper), alpha=0.2, color=color)
            ax.set_ylabel(r"$\log_{10}(\mathrm{loss}+\epsilon)$")
        else:  # linear
            ax.plot(grid, center, label=label, color=color)
            ax.fill_between(grid, lower, upper, alpha=0.2, color=color)
            ax.set_ylabel("Training loss (center ± band)")

    ax.legend(loc='upper right', prop={'size': 16})
    ax.set_xlim(0, grid[-1])
    ax.set_xlabel("Runtime (seconds)")
    plt.tight_layout()

    if args.outfile:
        plt.savefig(args.outfile, dpi=200)
        print(f"Saved: {args.outfile}")
    else:
        plt.show()

if __name__ == "__main__":
    main()
