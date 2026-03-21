import os

import matplotlib.pyplot as plt
import pandas as pd

from .config import FEE_BPS, HOLD_BARS, TARGET_COL
from .utils import max_drawdown_from_returns, regression_metrics


def strategy_metrics(pred, fwd_ret, hold_bars=16, fee_bps=5.0, threshold=None):
    df = pd.DataFrame({"pred": pred, "fwd_ret": fwd_ret}).dropna().copy()
    df = df.iloc[::hold_bars].copy()

    if threshold is None:
        threshold = float(df["pred"].abs().quantile(0.6))

    df["position"] = 0
    df.loc[df["pred"] > threshold, "position"] = 1
    df.loc[df["pred"] < -threshold, "position"] = -1

    turnover = df["position"].diff().abs().fillna(df["position"].abs())
    fee = turnover * (fee_bps / 10000.0)

    df["gross_ret"] = df["position"] * df["fwd_ret"]
    df["net_ret"] = df["gross_ret"] - fee

    periods_per_year = 6 * 365
    mean_ret = df["net_ret"].mean()
    std_ret = df["net_ret"].std(ddof=0)
    sharpe = 0.0 if std_ret == 0 else (mean_ret / std_ret) * (periods_per_year ** 0.5)

    curve = (1 + df["net_ret"]).cumprod()
    total_return = float(curve.iloc[-1] - 1) if len(curve) else 0.0
    annual_return = float((1 + total_return) ** (periods_per_year / max(len(df), 1)) - 1) if len(df) else 0.0
    mdd = max_drawdown_from_returns(df["net_ret"])
    trade_ratio = float((df["position"] != 0).mean()) if len(df) else 0.0

    metrics = {
        "threshold": float(threshold),
        "total_return": total_return,
        "annual_return": annual_return,
        "sharpe": float(sharpe),
        "max_drawdown": mdd,
        "trade_ratio": trade_ratio,
        "n_trades": int((turnover > 0).sum()),
    }
    return metrics, df


def evaluate_one_model(model_name, pred_df, output_dir):
    val_df = pred_df[pred_df["split"] == "val"].copy()
    test_df = pred_df[pred_df["split"] == "test"].copy()

    val_reg = regression_metrics(val_df[TARGET_COL], val_df["pred"])
    test_reg = regression_metrics(test_df[TARGET_COL], test_df["pred"])

    val_strat_metrics, val_strat_df = strategy_metrics(
        pred=val_df["pred"],
        fwd_ret=val_df[TARGET_COL],
        hold_bars=HOLD_BARS,
        fee_bps=FEE_BPS,
        threshold=None,
    )
    test_strat_metrics, test_strat_df = strategy_metrics(
        pred=test_df["pred"],
        fwd_ret=test_df[TARGET_COL],
        hold_bars=HOLD_BARS,
        fee_bps=FEE_BPS,
        threshold=val_strat_metrics["threshold"],
    )

    result = {
        "model": model_name,
        **{"val_" + k: v for k, v in val_reg.items()},
        **{"test_" + k: v for k, v in test_reg.items()},
        **{"val_" + k: v for k, v in val_strat_metrics.items()},
        **{"test_" + k: v for k, v in test_strat_metrics.items()},
    }

    val_strat_df.to_csv(os.path.join(output_dir, "%s_val_strategy.csv" % model_name))
    test_strat_df.to_csv(os.path.join(output_dir, "%s_test_strategy.csv" % model_name))
    pred_df.to_csv(os.path.join(output_dir, "%s_predictions.csv" % model_name), index=False)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot((1 + test_strat_df["net_ret"]).cumprod(), label="%s strategy" % model_name)
    ax.set_title("%s | Test cumulative return" % model_name)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "%s_test_curve.png" % model_name), dpi=150)
    plt.close(fig)

    return result
