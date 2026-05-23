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


def build_position(
    df: pd.DataFrame,
    ema_fast: int,
    ema_slow: int,
    trend_threshold: float,
    use_macd: int,
    volume_filter: float,
    funding_z_cap: float,
    allow_short: int,
    regime_filter: int = 1,
    regime_fast: int = 100,
    regime_slow: int = 200,
) -> pd.Series:
    close = df["close"]
    fast = close.ewm(span=ema_fast, adjust=False).mean()
    slow = close.ewm(span=ema_slow, adjust=False).mean()

    daily_close = close.resample("1d", label="left", closed="left").last()
    daily_fast = daily_close.ewm(span=regime_fast, adjust=False).mean()
    daily_slow = daily_close.ewm(span=regime_slow, adjust=False).mean()
    daily_regime_gap = (daily_fast / daily_slow - 1).shift(1).reindex(df.index, method="ffill")
    bull_regime = daily_regime_gap > 0
    bear_regime = daily_regime_gap < 0
    if not regime_filter:
        bull_regime = pd.Series(True, index=df.index)
        bear_regime = pd.Series(True, index=df.index)

    trend_1d_long = df["trend_1d_ema_gap"] > trend_threshold
    trend_4h_long = df["trend_4h_ema_gap"] > trend_threshold
    trend_1d_short = df["trend_1d_ema_gap"] < -trend_threshold
    trend_4h_short = df["trend_4h_ema_gap"] < -trend_threshold

    macd_long = df["macd_hist"] > 0
    macd_short = df["macd_hist"] < 0
    if not use_macd:
        macd_long = pd.Series(True, index=df.index)
        macd_short = pd.Series(True, index=df.index)

    volume_ok = df["volume_ratio"] > volume_filter
    funding_ok = df["funding_rate_zscore"].abs() < funding_z_cap

    raw_long = (fast > slow) & bull_regime & trend_1d_long & trend_4h_long & macd_long & volume_ok & funding_ok
    raw_short = (fast < slow) & bear_regime & trend_1d_short & trend_4h_short & macd_short & volume_ok & funding_ok

    position = pd.Series(0.0, index=df.index)
    position.loc[raw_long] = 1.0
    if allow_short:
        position.loc[raw_short] = -1.0

    return position.shift(1).fillna(0.0)


def evaluate(df: pd.DataFrame, position: pd.Series, fee_bps: float) -> dict[str, float]:
    returns = df["close"].pct_change().fillna(0.0)
    turnover = position.diff().abs().fillna(position.abs())
    strategy_return = position * returns - turnover * fee_bps / 10000
    equity = (1 + strategy_return).cumprod()

    entries = (position != 0) & (position.shift(1).fillna(0.0) != position)
    long_entries = entries & (position > 0)
    short_entries = entries & (position < 0)
    years = max((df.index[-1] - df.index[0]).days / 365.25, 1 / 365.25)
    total_return = float(equity.iloc[-1] - 1)
    annual_return = float(equity.iloc[-1] ** (1 / years) - 1)
    sharpe = float(strategy_return.mean() / strategy_return.std() * np.sqrt(24 * 365)) if strategy_return.std() > 0 else 0.0

    active_returns = strategy_return.loc[position != 0]
    long_return = (1 + strategy_return.where(position > 0, 0.0)).prod() - 1
    short_return = (1 + strategy_return.where(position < 0, 0.0)).prod() - 1
    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "max_drawdown": max_drawdown(equity),
        "sharpe": sharpe,
        "trades": int(entries.sum()),
        "long_trades": int(long_entries.sum()),
        "short_trades": int(short_entries.sum()),
        "exposure": float(position.mean()),
        "abs_exposure": float(position.abs().mean()),
        "long_exposure": float((position > 0).mean()),
        "short_exposure": float((position < 0).mean()),
        "long_total_return": float(long_return),
        "short_total_return": float(short_return),
        "avg_bar_return_when_in_position": float(active_returns.mean()) if len(active_returns) else 0.0,
    }


def year_report(df: pd.DataFrame, position: pd.Series, fee_bps: float) -> pd.DataFrame:
    rows = []
    for year, group in df.groupby(df.index.year):
        pos = position.reindex(group.index)
        metrics = evaluate(group, pos, fee_bps)
        metrics["year"] = year
        rows.append(metrics)
    return pd.DataFrame(rows).set_index("year")


def optimize(
    df: pd.DataFrame,
    fee_bps: float,
    short_mode: str,
    regime_filter: int,
    regime_fast: int,
    regime_slow: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    rows = []
    allow_short_values = [0, 1] if short_mode == "both" else [1]
    grid = {
        "ema_fast": [12, 20, 30, 48],
        "ema_slow": [60, 96, 120, 200],
        "trend_threshold": [0.0, 0.0025, 0.005, 0.01],
        "use_macd": [0, 1],
        "volume_filter": [-999.0, -0.2, 0.0, 0.2],
        "funding_z_cap": [0.75, 1.5, 2.5, 999.0],
        "allow_short": allow_short_values,
        "regime_filter": [regime_filter],
        "regime_fast": [regime_fast],
        "regime_slow": [regime_slow],
    }

    for values in itertools.product(*grid.values()):
        params = dict(zip(grid.keys(), values))
        if params["ema_fast"] >= params["ema_slow"]:
            continue

        position = build_position(df, **params)
        metrics = evaluate(df, position, fee_bps)
        rows.append({**params, **metrics})

    result = pd.DataFrame(rows)
    result = result[result["trades"] >= 20].copy()
    if short_mode == "long_short":
        result = result[result["short_trades"] >= 10].copy()
    result["score"] = result["annual_return"] + result["max_drawdown"] * 0.5 + result["sharpe"] * 0.05
    result = result.sort_values("score", ascending=False)
    best = result.iloc[0][list(grid.keys())].to_dict()
    return result, best


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize BTC/ETH multi-timeframe EMA trend rules.")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", default="reports/rule_strategy_optimization.csv", type=Path)
    parser.add_argument("--fee-bps", default=5.0, type=float, help="Round-trip-ish one-side fee/slippage estimate in bps.")
    parser.add_argument(
        "--short-mode",
        choices=["both", "long_short"],
        default="long_short",
        help="both searches long-only and long-short; long_short forces allow_short=1.",
    )
    parser.add_argument("--regime-filter", type=int, default=1, help="1 enables daily EMA regime filter, 0 disables it.")
    parser.add_argument("--regime-fast", type=int, default=100, help="Fast daily EMA for bull/bear regime.")
    parser.add_argument("--regime-slow", type=int, default=200, help="Slow daily EMA for bull/bear regime.")
    args = parser.parse_args()

    df = load_dataset(args.data)
    required = [
        "close",
        "low",
        "atr14",
        "volume_ratio",
        "macd_hist",
        "funding_rate_zscore",
        "trend_1d_ema_gap",
        "trend_4h_ema_gap",
    ]
    df = df.dropna(subset=required).copy()

    result, best = optimize(df, args.fee_bps, args.short_mode, args.regime_filter, args.regime_fast, args.regime_slow)
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
