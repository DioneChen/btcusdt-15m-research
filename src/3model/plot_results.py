import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


MODELS = ["xgboost", "lightgbm", "lstm"]


def plot_combined_curves(output_dir, figure_dir):
    os.makedirs(figure_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5))
    found = False
    for model in MODELS:
        path = os.path.join(output_dir, "%s_test_strategy.csv" % model)
        if os.path.exists(path):
            df = pd.read_csv(path)
            if "net_ret" in df.columns:
                curve = (1 + df["net_ret"].fillna(0)).cumprod()
                ax.plot(curve, label=model)
                found = True
    if not found:
        raise FileNotFoundError("No *_test_strategy.csv files found in %s" % output_dir)

    ax.set_title("BTCUSDT 15m | Test cumulative return comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(figure_dir, "test_cumulative_return_comparison.png"), dpi=150)
    plt.close(fig)


def plot_metric_bars(summary_csv, figure_dir):
    os.makedirs(figure_dir, exist_ok=True)
    df = pd.read_csv(summary_csv)
    cols = ["test_ic", "test_rank_ic", "test_direction_acc", "test_sharpe", "test_total_return"]

    fig, axes = plt.subplots(len(cols), 1, figsize=(10, 14), sharex=True)
    for ax, col in zip(axes, cols):
        ax.bar(df["model"], df[col])
        ax.set_title(col)
        ax.axhline(0, linewidth=1)
    fig.tight_layout()
    fig.savefig(os.path.join(figure_dir, "model_metric_comparison.png"), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--summary-csv", default="results/model_compare_summary.csv")
    parser.add_argument("--figure-dir", default="assets/figures")
    args = parser.parse_args()

    if os.path.exists(args.summary_csv):
        plot_metric_bars(args.summary_csv, args.figure_dir)
    if os.path.isdir(args.output_dir):
        try:
            plot_combined_curves(args.output_dir, args.figure_dir)
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
