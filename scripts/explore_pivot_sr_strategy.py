from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from optimize_rule_strategy import load_dataset, max_drawdown


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def daily_bars(df: pd.DataFrame) -> pd.DataFrame:
    return df.resample("1d", label="left", closed="left").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "atr14": "last",
        }
    ).dropna(subset=["open", "high", "low", "close"])


def pivothigh(series: pd.Series, left: int, right: int) -> pd.Series:
    out = pd.Series(False, index=series.index)
    values = series.to_numpy()
    for i in range(left, len(series) - right):
        window = values[i - left : i + right + 1]
        out.iat[i] = values[i] == np.nanmax(window) and values[i] > np.nanmax(values[i - left : i])
    return out


def pivotlow(series: pd.Series, left: int, right: int) -> pd.Series:
    out = pd.Series(False, index=series.index)
    values = series.to_numpy()
    for i in range(left, len(series) - right):
        window = values[i - left : i + right + 1]
        out.iat[i] = values[i] == np.nanmin(window) and values[i] < np.nanmin(values[i - left : i])
    return out


def filter_alternating_pivots(pivots: list[dict]) -> list[dict]:
    filtered: list[dict] = []
    for pivot in sorted(pivots, key=lambda item: item["pivot_time"]):
        if not filtered:
            filtered.append(pivot)
            continue

        last = filtered[-1]
        if pivot["type"] != last["type"]:
            filtered.append(pivot)
            continue

        if pivot["type"] == "resistance" and pivot["level"] > last["level"]:
            filtered[-1] = pivot
        elif pivot["type"] == "support" and pivot["level"] < last["level"]:
            filtered[-1] = pivot

    return filtered


def build_levels(df: pd.DataFrame, ema_len: int, left: int, right: int) -> pd.DataFrame:
    d = daily_bars(df)
    d["pivot_ema"] = d["close"].ewm(span=ema_len, adjust=False).mean()
    d["pivot_high"] = pivothigh(d["pivot_ema"], left, right)
    d["pivot_low"] = pivotlow(d["pivot_ema"], left, right)

    pivots = []
    for ts, row in d.iterrows():
        confirm_ts = ts + pd.Timedelta(days=right)
        if row["pivot_high"]:
            pivots.append(
                {
                    "pivot_time": ts,
                    "confirm_time": confirm_ts,
                    "type": "resistance",
                    "level": row["high"],
                    "pivot_ema": row["pivot_ema"],
                }
            )
        if row["pivot_low"]:
            pivots.append(
                {
                    "pivot_time": ts,
                    "confirm_time": confirm_ts,
                    "type": "support",
                    "level": row["low"],
                    "pivot_ema": row["pivot_ema"],
                }
            )

    levels = filter_alternating_pivots(pivots)
    if not levels:
        return pd.DataFrame(index=df.index, columns=["support", "resistance", "sr_position"])

    level_df = pd.DataFrame(levels).sort_values("confirm_time")
    hourly = pd.DataFrame(index=df.index)

    for level_type, col in [("support", "support"), ("resistance", "resistance")]:
        events = level_df[level_df["type"] == level_type].copy()
        if events.empty:
            hourly[col] = np.nan
            hourly[f"{col}_age_hours"] = np.nan
            continue

        level_series = pd.Series(events["level"].to_numpy(), index=pd.DatetimeIndex(events["confirm_time"]))
        full_index = hourly.index.union(level_series.index)
        hourly[col] = level_series.reindex(full_index).sort_index().ffill().reindex(hourly.index)

        time_series = pd.Series(level_series.index, index=level_series.index)
        last_time = time_series.reindex(full_index).sort_index().ffill().reindex(hourly.index)
        hourly[f"{col}_age_hours"] = (hourly.index.to_series() - last_time).dt.total_seconds() / 3600

    type_map = {"support": -1, "resistance": 1}
    type_series = pd.Series(
        level_df["type"].map(type_map).to_numpy(),
        index=pd.DatetimeIndex(level_df["confirm_time"]),
    )
    hourly["last_pivot_type"] = type_series.reindex(hourly.index.union(type_series.index)).sort_index().ffill().reindex(hourly.index).fillna(0)

    range_width = hourly["resistance"] - hourly["support"]
    hourly["sr_range_pct"] = range_width / df["close"]
    hourly["sr_position"] = (df["close"] - hourly["support"]) / range_width
    hourly["dist_support_pct"] = df["close"] / hourly["support"] - 1
    hourly["dist_resistance_pct"] = hourly["resistance"] / df["close"] - 1
    hourly["dist_support_atr"] = (df["close"] - hourly["support"]) / df["atr14"]
    hourly["dist_resistance_atr"] = (hourly["resistance"] - df["close"]) / df["atr14"]
    return hourly


def factor_report(df: pd.DataFrame, levels: pd.DataFrame) -> dict[str, float]:
    data = df.join(levels, how="left")
    data["future_return_24h"] = data["close"].shift(-24) / data["close"] - 1
    factor_cols = ["sr_position", "dist_support_atr", "dist_resistance_atr", "sr_range_pct"]
    report = {}
    for col in factor_cols:
        sample = data[[col, "future_return_24h"]].replace([np.inf, -np.inf], np.nan).dropna()
        report[f"{col}_ic_24h"] = float(sample[col].corr(sample["future_return_24h"], method="spearman")) if len(sample) > 30 else np.nan
    return report


def backtest_sr(
    df: pd.DataFrame,
    levels: pd.DataFrame,
    proximity_atr: float,
    stop_atr: float,
    max_hold_hours: int,
    fee_bps: float,
) -> dict[str, float]:
    data = df.join(levels, how="left")
    position = 0
    entry_price = np.nan
    entry_time = None
    entry_side = 0
    trade_returns = []
    trades = []
    equity_curve = pd.Series(1.0, index=data.index)
    equity = 1.0

    for ts, row in data.iterrows():
        if pd.isna(row["atr14"]) or row["atr14"] <= 0:
            equity_curve.at[ts] = equity
            continue

        if position != 0:
            held = int((ts - entry_time).total_seconds() // 3600)
            stop_price = entry_price - entry_side * stop_atr * row["atr14"]
            hit_stop = row["low"] <= stop_price if entry_side > 0 else row["high"] >= stop_price
            hit_target = row["high"] >= row["resistance"] if entry_side > 0 and pd.notna(row["resistance"]) else False
            hit_target = hit_target or (
                row["low"] <= row["support"] if entry_side < 0 and pd.notna(row["support"]) else False
            )
            timed_out = held >= max_hold_hours
            if hit_stop or hit_target or timed_out:
                if hit_stop:
                    exit_price = stop_price
                elif hit_target:
                    exit_price = row["resistance"] if entry_side > 0 else row["support"]
                else:
                    exit_price = row["close"]
                gross = entry_side * (exit_price / entry_price - 1)
                net = gross - 2 * fee_bps / 10000
                trade_returns.append(net)
                trades.append({"entry_time": entry_time, "exit_time": ts, "side": entry_side, "return": net})
                equity *= 1 + net
                position = 0

        if position == 0:
            near_support = pd.notna(row["support"]) and row["low"] <= row["support"] + proximity_atr * row["atr14"]
            near_resistance = pd.notna(row["resistance"]) and row["high"] >= row["resistance"] - proximity_atr * row["atr14"]
            valid_range = pd.notna(row["support"]) and pd.notna(row["resistance"]) and row["resistance"] > row["support"]

            if valid_range and near_support and row["close"] > row["support"]:
                position = 1
                entry_side = 1
                entry_price = row["close"]
                entry_time = ts
            elif valid_range and near_resistance and row["close"] < row["resistance"]:
                position = -1
                entry_side = -1
                entry_price = row["close"]
                entry_time = ts

        equity_curve.at[ts] = equity

    trade_returns = pd.Series(trade_returns, dtype=float)
    if trade_returns.empty:
        return {
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "sharpe": 0.0,
            "trades": 0,
            "win_rate": 0.0,
            "avg_trade": 0.0,
        }

    years = max((data.index[-1] - data.index[0]).days / 365.25, 1 / 365.25)
    hourly_returns = equity_curve.pct_change().fillna(0.0)
    sharpe = hourly_returns.mean() / hourly_returns.std() * np.sqrt(24 * 365) if hourly_returns.std() > 0 else 0.0
    return {
        "total_return": float(equity_curve.iloc[-1] - 1),
        "annual_return": float(equity_curve.iloc[-1] ** (1 / years) - 1),
        "max_drawdown": max_drawdown(equity_curve),
        "sharpe": float(sharpe),
        "trades": int(len(trade_returns)),
        "win_rate": float((trade_returns > 0).mean()),
        "avg_trade": float(trade_returns.mean()),
    }


def evaluate_grid(
    df: pd.DataFrame,
    ema_lengths: list[int],
    left_windows: list[int],
    right: int,
    proximity_atr: float,
    stop_atr: float,
    max_hold_hours: int,
    fee_bps: float,
) -> pd.DataFrame:
    rows = []
    train = df.loc[: "2023-12-31 23:59:59"]
    test = df.loc["2024-01-01":]
    for ema_len, left in itertools.product(ema_lengths, left_windows):
        levels = build_levels(df, ema_len, left, right)
        train_metrics = backtest_sr(train, levels.reindex(train.index), proximity_atr, stop_atr, max_hold_hours, fee_bps)
        test_metrics = backtest_sr(test, levels.reindex(test.index), proximity_atr, stop_atr, max_hold_hours, fee_bps)
        train_factors = factor_report(train, levels.reindex(train.index))
        test_factors = factor_report(test, levels.reindex(test.index))
        rows.append(
            {
                "ema_len": ema_len,
                "left": left,
                "right": right,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"test_{k}": v for k, v in test_metrics.items()},
                **{f"train_{k}": v for k, v in train_factors.items()},
                **{f"test_{k}": v for k, v in test_factors.items()},
                "score": test_metrics["annual_return"] + 0.5 * test_metrics["max_drawdown"] + 0.05 * test_metrics["sharpe"],
            }
        )
    return pd.DataFrame(rows).sort_values("score", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Explore 1D EMA pivot support/resistance parameters.")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--ema-lengths", default="5,7,10,14,20,30")
    parser.add_argument("--left-windows", default="5,8,10,12,15,20,30")
    parser.add_argument("--right", default=1, type=int)
    parser.add_argument("--proximity-atr", default=0.5, type=float)
    parser.add_argument("--stop-atr", default=1.0, type=float)
    parser.add_argument("--max-hold-hours", default=120, type=int)
    parser.add_argument("--fee-bps", default=5.0, type=float)
    args = parser.parse_args()

    df = load_dataset(args.data).dropna(subset=["open", "high", "low", "close", "atr14"]).copy()
    result = evaluate_grid(
        df,
        parse_int_list(args.ema_lengths),
        parse_int_list(args.left_windows),
        args.right,
        args.proximity_atr,
        args.stop_atr,
        args.max_hold_hours,
        args.fee_bps,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    print(f"Saved result to {args.out}")
    print(result.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
