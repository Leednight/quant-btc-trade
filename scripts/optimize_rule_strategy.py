from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df.sort_index()


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    drawdown = equity / peak - 1
    return float(drawdown.min())


def build_position(df: pd.DataFrame, ema_fast: int, ema_slow: int, atr_stop: float, volume_filter: float) -> pd.Series:
    close = df["close"]
    fast = close.ewm(span=ema_fast, adjust=False).mean()
    slow = close.ewm(span=ema_slow, adjust=False).mean()

    trend_1d = df["trend_1d_ema_gap"] > 0
    trend_4h = df["trend_4h_ema_gap"] > 0
    momentum = df["macd_hist"] > 0
    volume_ok = df["volume_ratio"] > volume_filter
    raw_long = (fast > slow) & trend_1d & trend_4h & momentum & volume_ok

    position = pd.Series(0.0, index=df.index)
    in_pos = False
    entry = np.nan
    stop = np.nan

    for i, ts in enumerate(df.index):
        if i == 0:
            continue

        if in_pos:
            low = df.at[ts, "low"]
            if low <= stop or not raw_long.iat[i]:
                in_pos = False
                entry = np.nan
                stop = np.nan

        if not in_pos and raw_long.iat[i]:
            in_pos = True
            entry = df.at[ts, "close"]
            stop = entry - atr_stop * df.at[ts, "atr14"]

        position.iat[i] = 1.0 if in_pos else 0.0

    return position.shift(1).fillna(0.0)


def evaluate(df: pd.DataFrame, position: pd.Series, fee_bps: float) -> dict[str, float]:
    returns = df["close"].pct_change().fillna(0.0)
    turnover = position.diff().abs().fillna(position.abs())
    strategy_return = position * returns - turnover * fee_bps / 10000
    equity = (1 + strategy_return).cumprod()

    trades = int((position.diff() > 0).sum())
    years = max((df.index[-1] - df.index[0]).days / 365.25, 1 / 365.25)
    total_return = float(equity.iloc[-1] - 1)
    annual_return = float(equity.iloc[-1] ** (1 / years) - 1)
    sharpe = float(strategy_return.mean() / strategy_return.std() * np.sqrt(24 * 365)) if strategy_return.std() > 0 else 0.0

    selected = df.loc[position > 0, "close"].pct_change().dropna()
    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "max_drawdown": max_drawdown(equity),
        "sharpe": sharpe,
        "trades": trades,
        "exposure": float(position.mean()),
        "avg_bar_return_when_in_position": float(selected.mean()) if len(selected) else 0.0,
    }


def year_report(df: pd.DataFrame, position: pd.Series, fee_bps: float) -> pd.DataFrame:
    rows = []
    for year, group in df.groupby(df.index.year):
        pos = position.reindex(group.index)
        metrics = evaluate(group, pos, fee_bps)
        metrics["year"] = year
        rows.append(metrics)
    return pd.DataFrame(rows).set_index("year")


def optimize(df: pd.DataFrame, fee_bps: float) -> tuple[pd.DataFrame, dict[str, float]]:
    rows = []
    grid = {
        "ema_fast": [12, 20, 30],
        "ema_slow": [48, 60, 96],
        "atr_stop": [1.5, 2.0, 2.5, 3.0],
        "volume_filter": [-0.2, 0.0, 0.2],
    }

    for values in itertools.product(*grid.values()):
        params = dict(zip(grid.keys(), values))
        if params["ema_fast"] >= params["ema_slow"]:
            continue

        position = build_position(df, **params)
        metrics = evaluate(df, position, fee_bps)
        rows.append({**params, **metrics})

    result = pd.DataFrame(rows)
    result["score"] = result["annual_return"] + result["max_drawdown"] * 0.5 + result["sharpe"] * 0.03
    result = result.sort_values("score", ascending=False)
    best = result.iloc[0][list(grid.keys())].to_dict()
    return result, best


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize a simple long-only BTC/ETH trend rule strategy.")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", default="reports/rule_strategy_optimization.csv", type=Path)
    parser.add_argument("--fee-bps", default=5.0, type=float, help="Round-trip-ish one-side fee/slippage estimate in bps.")
    args = parser.parse_args()

    df = load_dataset(args.data)
    required = [
        "close",
        "low",
        "atr14",
        "volume_ratio",
        "macd_hist",
        "trend_1d_ema_gap",
        "trend_4h_ema_gap",
    ]
    df = df.dropna(subset=required).copy()

    result, best = optimize(df, args.fee_bps)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)

    best_position = build_position(df, **best)
    yearly = year_report(df, best_position, args.fee_bps)
    yearly_path = args.out.with_name(args.out.stem + "_yearly.csv")
    yearly.to_csv(yearly_path)

    print("Best params")
    print(best)
    print("\nTop 10")
    print(result.head(10).to_string(index=False))
    print(f"\nSaved optimization to {args.out}")
    print(f"Saved yearly report to {yearly_path}")


if __name__ == "__main__":
    main()
