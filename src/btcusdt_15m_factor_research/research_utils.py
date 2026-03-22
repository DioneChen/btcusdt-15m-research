
from typing import Dict, List

import numpy as np
import pandas as pd


def factor_ic_table(data: pd.DataFrame, factors: List[str], target: str, min_n: int = 500) -> pd.DataFrame:
    rows = []
    for f in factors:
        sub = data[[f, target]].dropna()
        if len(sub) < min_n:
            continue
        rows.append({
            "factor": f,
            "n": len(sub),
            "pearson_ic": sub[f].corr(sub[target], method="pearson"),
            "spearman_ic": sub[f].corr(sub[target], method="spearman"),
        })
    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out
    return out.sort_values("spearman_ic", ascending=False).reset_index(drop=True)


def monthly_rank_ic_series(data: pd.DataFrame, factor: str, target: str) -> pd.Series:
    tmp = data[[factor, target]].dropna().copy()
    tmp["month"] = tmp.index.tz_localize(None).to_period("M")
    out = tmp.groupby("month").apply(lambda x: x[factor].corr(x[target], method="spearman"))
    out.index = out.index.astype(str)
    return out


def monthly_rank_ic_summary(data: pd.DataFrame, factors: List[str], target: str) -> pd.DataFrame:
    rows = []
    for f in factors:
        s = monthly_rank_ic_series(data, f, target).dropna()
        if len(s) < 3:
            continue
        mean_ic = s.mean()
        std_ic = s.std()
        ir = mean_ic / std_ic if pd.notna(std_ic) and std_ic > 0 else np.nan
        same_sign_ratio = (np.sign(s) == np.sign(mean_ic)).mean() if pd.notna(mean_ic) and abs(mean_ic) > 1e-12 else np.nan
        rows.append({
            "factor": f,
            "months": len(s),
            "monthly_mean_ic": mean_ic,
            "monthly_std_ic": std_ic,
            "monthly_ic_ir": ir,
            "same_sign_ratio": same_sign_ratio,
            "abs_monthly_mean_ic": abs(mean_ic),
            "abs_monthly_ic_ir": abs(ir) if pd.notna(ir) else np.nan,
        })
    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out
    return out.sort_values("abs_monthly_mean_ic", ascending=False).reset_index(drop=True)


def quantile_test(data: pd.DataFrame, factor: str, target: str, q: int = 5) -> pd.DataFrame:
    tmp = data[[factor, target]].dropna().copy()
    tmp["bucket"] = pd.qcut(tmp[factor], q=q, labels=False, duplicates="drop")
    return tmp.groupby("bucket")[target].agg(["mean", "median", "count"])


def calc_aligned_ic(data: pd.DataFrame, factor: str, target: str, sign: int) -> float:
    sub = data[[factor, target]].dropna().copy()
    if len(sub) < 30:
        return np.nan
    aligned = sign * sub[factor]
    return aligned.corr(sub[target], method="spearman")
