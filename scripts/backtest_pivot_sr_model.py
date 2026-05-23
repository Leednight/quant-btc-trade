from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

try:
    from explore_pivot_sr_strategy import build_levels
except ModuleNotFoundError:
    from scripts.explore_pivot_sr_strategy import build_levels
from optimize_rule_strategy import load_dataset, max_drawdown


PIVOT_FACTOR_COLUMNS = [
    "sr_range_pct",
    "sr_position",
    "dist_support_pct",
    "dist_resistance_pct",
    "dist_support_atr",
    "dist_resistance_atr",
    "last_pivot_type",
    "support_age_hours",
    "resistance_age_hours",
]


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def build_model_data(df: pd.DataFrame, model_bundle: dict) -> pd.DataFrame:
    levels = build_levels(
        df,
        int(model_bundle["ema_len"]),
        int(model_bundle["left"]),
        int(model_bundle["right"]),
    )
    factor_cols = model_bundle.get("pivot_factor_columns", PIVOT_FACTOR_COLUMNS)
    data = df.join(levels[factor_cols], how="left")
    return data


def make_position(
    pred: pd.Series,
    long_threshold: float,
    short_threshold: float,
    mode: str,
    position_mode: str,
    base_size: float,
    mid_size: float,
    extreme_size: float,
    max_size: float,
    long_extreme_threshold: float,
    short_extreme_threshold: float,
    long_max_threshold: float,
    short_max_threshold: float,
    leverage: float,
) -> pd.Series:
    position = pd.Series(0.0, index=pred.index)
    if position_mode == "fixed":
        long_size = max_size
        short_size = max_size
        position.loc[pred > long_threshold] = long_size
        if mode == "long_short":
            position.loc[pred < short_threshold] = -short_size
        return position.shift(1).fillna(0.0) * leverage

    long_size = pd.Series(base_size, index=pred.index)
    long_size.loc[pred > long_extreme_threshold] = mid_size
    if long_max_threshold > 0:
        long_size.loc[pred > long_max_threshold] = extreme_size
    short_size = pd.Series(base_size, index=pred.index)
    short_size.loc[pred < short_extreme_threshold] = mid_size
    if short_max_threshold > 0:
        short_size.loc[pred < short_max_threshold] = extreme_size
    position.loc[pred > long_threshold] = long_size.loc[pred > long_threshold].clip(upper=max_size)
    if mode == "long_short":
        position.loc[pred < short_threshold] = -short_size.loc[pred < short_threshold].clip(upper=max_size)
    return position.shift(1).fillna(0.0) * leverage


def apply_btc_trend_filter(
    desired: pd.Series,
    btc_trend: pd.Series | None,
    mode: str,
    countertrend_size: float,
) -> pd.Series:
    if btc_trend is None or mode == "none":
        return desired

    trend = btc_trend.shift(1).reindex(desired.index).ffill()
    filtered = desired.copy()
    long_against_btc = (filtered > 0) & (trend <= 0)
    short_against_btc = (filtered < 0) & (trend >= 0)
    countertrend = long_against_btc | short_against_btc

    if mode == "confirm":
        filtered.loc[countertrend] = 0.0
    elif mode == "reduce":
        filtered.loc[countertrend] = np.sign(filtered.loc[countertrend]) * np.minimum(
            filtered.loc[countertrend].abs(),
            countertrend_size,
        )
    return filtered


def apply_execution_constraints(
    df: pd.DataFrame,
    pred: pd.Series,
    desired: pd.Series,
    long_threshold: float,
    short_threshold: float,
    min_hold_hours: int,
    max_consecutive_losses: int,
    cooldown_hours: int,
    fee_bps: float,
    slippage_bps: float,
    atr_stop_mult: float,
    max_trade_loss: float,
    max_position_loss: float,
    short_rebound_stop: float,
    drawdown_reduce_threshold: float,
    drawdown_reduce_factor: float,
    drawdown_pause_threshold: float,
    drawdown_cooldown_hours: int,
    loss_reduce_after: int,
    loss_reduced_size: float,
    vol_pause_quantile: float,
    vol_pause_hours: int,
    tighten_after_losses: int,
    tightened_long_threshold: float,
    tightened_short_threshold: float,
    tighten_reset_hours: int,
    daily_loss_limit: float,
    rolling_loss_window_hours: int,
    rolling_loss_limit: float,
    rolling_loss_cooldown_hours: int,
    loss_streak_loss_limit: float,
    loss_streak_cooldown_hours: int,
) -> pd.Series:
    position = pd.Series(0.0, index=desired.index)
    current = 0.0
    entry_time = None
    entry_price = np.nan
    entry_atr = np.nan
    consecutive_losses = 0
    loss_streak_loss = 0.0
    last_loss_time = None
    daily_realized = 0.0
    realized_history = []
    equity = 1.0
    peak_equity = 1.0
    current_day = None
    cooldown_until = pd.Timestamp.min.tz_localize("UTC") if desired.index.tz is not None else pd.Timestamp.min
    vol_pause_until = cooldown_until
    rolling_loss_cooldown_until = cooldown_until
    drawdown_cooldown_until = cooldown_until
    loss_streak_cooldown_until = cooldown_until
    drawdown_pause_used = False
    vol_threshold = np.inf
    if vol_pause_quantile > 0 and "volatility_24" in df.columns:
        vol_threshold = float(df["volatility_24"].dropna().quantile(vol_pause_quantile))
    signal_pred = pred.shift(1).reindex(desired.index)

    for ts in desired.index:
        target = desired.at[ts]
        held_hours = 0 if entry_time is None else int((ts - entry_time).total_seconds() // 3600)
        can_change = current == 0 or held_hours >= min_hold_hours

        if current != 0 and pd.notna(entry_price):
            close = df.at[ts, "close"]
            high = df.at[ts, "high"] if "high" in df.columns else close
            low = df.at[ts, "low"] if "low" in df.columns else close
            stop_price = np.nan
            if atr_stop_mult > 0 and pd.notna(entry_atr) and entry_atr > 0:
                stop_price = entry_price - np.sign(current) * atr_stop_mult * entry_atr
            if max_trade_loss > 0:
                loss_stop_price = entry_price * (1 - max_trade_loss) if current > 0 else entry_price * (1 + max_trade_loss)
                if pd.isna(stop_price):
                    stop_price = loss_stop_price
                elif current > 0:
                    stop_price = max(stop_price, loss_stop_price)
                else:
                    stop_price = min(stop_price, loss_stop_price)
            if max_position_loss > 0 and abs(current) > 0:
                position_loss_pct = max_position_loss / abs(current)
                account_stop_price = entry_price * (1 - position_loss_pct) if current > 0 else entry_price * (1 + position_loss_pct)
                if pd.isna(stop_price):
                    stop_price = account_stop_price
                elif current > 0:
                    stop_price = max(stop_price, account_stop_price)
                else:
                    stop_price = min(stop_price, account_stop_price)

            intrabar_stop = False
            if pd.notna(stop_price):
                intrabar_stop = low <= stop_price if current > 0 else high >= stop_price
            rebound_stop = current < 0 and short_rebound_stop > 0 and high / entry_price - 1 >= short_rebound_stop
            if (intrabar_stop or rebound_stop) and can_change:
                target = 0.0

        day = ts.date()
        if current_day != day:
            current_day = day
            daily_realized = 0.0
        if (
            tighten_reset_hours > 0
            and current == 0
            and last_loss_time is not None
            and consecutive_losses > 0
            and ts - last_loss_time >= pd.Timedelta(hours=tighten_reset_hours)
        ):
            consecutive_losses = 0
            loss_streak_loss = 0.0
            last_loss_time = None

        extreme_vol = pd.notna(df.at[ts, "volatility_24"]) and df.at[ts, "volatility_24"] > vol_threshold
        if extreme_vol and current == 0 and vol_pause_hours > 0:
            vol_pause_until = ts + pd.Timedelta(hours=vol_pause_hours)

        drawdown = equity / peak_equity - 1 if peak_equity > 0 else 0.0
        if drawdown_reduce_threshold > 0 and drawdown > -drawdown_reduce_threshold:
            drawdown_pause_used = False
        if (
            current == 0
            and target != 0
            and drawdown_pause_threshold > 0
            and drawdown <= -drawdown_pause_threshold
            and not drawdown_pause_used
        ):
            drawdown_cooldown_until = ts + pd.Timedelta(hours=drawdown_cooldown_hours)
            drawdown_pause_used = True
        if (
            ts < cooldown_until
            or ts < vol_pause_until
            or ts < rolling_loss_cooldown_until
            or ts < drawdown_cooldown_until
            or ts < loss_streak_cooldown_until
        ) and current == 0:
            target = 0.0

        if current == 0 and target != 0 and daily_loss_limit > 0 and daily_realized <= -daily_loss_limit:
            target = 0.0
        if current == 0 and target != 0 and drawdown_reduce_threshold > 0 and drawdown <= -drawdown_reduce_threshold:
            target *= drawdown_reduce_factor

        if current == 0 and target != 0 and tighten_after_losses > 0 and consecutive_losses >= tighten_after_losses:
            signal_prob = signal_pred.at[ts]
            if pd.isna(signal_prob):
                target = 0.0
            elif target > 0 and signal_prob <= tightened_long_threshold:
                target = 0.0
            elif target < 0 and signal_prob >= tightened_short_threshold:
                target = 0.0

        if current == 0 and target != 0 and loss_reduce_after > 0 and consecutive_losses >= loss_reduce_after:
            target = np.sign(target) * min(abs(target), loss_reduced_size)

        direction_change = np.sign(target) != np.sign(current)
        size_change = target != current
        if size_change and can_change:
            if current != 0 and pd.notna(entry_price):
                exit_price = df.at[ts, "close"]
                high = df.at[ts, "high"] if "high" in df.columns else exit_price
                low = df.at[ts, "low"] if "low" in df.columns else exit_price
                stop_price = np.nan
                if atr_stop_mult > 0 and pd.notna(entry_atr) and entry_atr > 0:
                    stop_price = entry_price - np.sign(current) * atr_stop_mult * entry_atr
                if max_trade_loss > 0:
                    loss_stop_price = entry_price * (1 - max_trade_loss) if current > 0 else entry_price * (1 + max_trade_loss)
                    if pd.isna(stop_price):
                        stop_price = loss_stop_price
                    elif current > 0:
                        stop_price = max(stop_price, loss_stop_price)
                    else:
                        stop_price = min(stop_price, loss_stop_price)
                if max_position_loss > 0 and abs(current) > 0:
                    position_loss_pct = max_position_loss / abs(current)
                    account_stop_price = entry_price * (1 - position_loss_pct) if current > 0 else entry_price * (1 + position_loss_pct)
                    if pd.isna(stop_price):
                        stop_price = account_stop_price
                    elif current > 0:
                        stop_price = max(stop_price, account_stop_price)
                    else:
                        stop_price = min(stop_price, account_stop_price)
                stop_hit = False
                if pd.notna(stop_price):
                    if current > 0 and low <= stop_price:
                        exit_price = stop_price
                        stop_hit = True
                    elif current < 0 and high >= stop_price:
                        exit_price = stop_price
                        stop_hit = True
                if current < 0 and short_rebound_stop > 0 and not stop_hit:
                    rebound_price = entry_price * (1 + short_rebound_stop)
                    if high >= rebound_price:
                        exit_price = rebound_price
                gross = current * (exit_price / entry_price - 1)
                net = gross - 2 * (fee_bps + slippage_bps) / 10000
                equity *= max(1 + net, 1e-9)
                peak_equity = max(peak_equity, equity)
                daily_realized += net
                if rolling_loss_window_hours > 0 and rolling_loss_limit > 0:
                    realized_history.append((ts, net))
                    cutoff = ts - pd.Timedelta(hours=rolling_loss_window_hours)
                    realized_history = [(trade_ts, trade_net) for trade_ts, trade_net in realized_history if trade_ts >= cutoff]
                    rolling_realized = sum(trade_net for _, trade_net in realized_history)
                    if rolling_realized <= -rolling_loss_limit:
                        rolling_loss_cooldown_until = ts + pd.Timedelta(hours=rolling_loss_cooldown_hours)
                if direction_change and net < 0:
                    consecutive_losses += 1
                    loss_streak_loss += net
                    last_loss_time = ts
                elif direction_change:
                    consecutive_losses = 0
                    loss_streak_loss = 0.0
                    last_loss_time = None

                if max_consecutive_losses > 0 and consecutive_losses >= max_consecutive_losses:
                    cooldown_until = ts + pd.Timedelta(hours=cooldown_hours)
                    target = 0.0
                if loss_streak_loss_limit > 0 and loss_streak_loss <= -loss_streak_loss_limit:
                    loss_streak_cooldown_until = ts + pd.Timedelta(hours=loss_streak_cooldown_hours)
                    target = 0.0
                if target != 0 and daily_loss_limit > 0 and daily_realized <= -daily_loss_limit:
                    target = 0.0
                if target != 0 and ts < rolling_loss_cooldown_until:
                    target = 0.0

            current = target
            if current != 0:
                entry_time = ts
                entry_price = df.at[ts, "close"]
                entry_atr = df.at[ts, "atr14"] if "atr14" in df.columns else np.nan
            else:
                entry_time = None
                entry_price = np.nan
                entry_atr = np.nan

        position.at[ts] = current

    return position


def funding_cost_series(df: pd.DataFrame, position: pd.Series) -> pd.Series:
    if "funding_rate" not in df.columns:
        return pd.Series(0.0, index=df.index)

    funding_rate = df["funding_rate"].fillna(0.0)
    funding_event = funding_rate.ne(funding_rate.shift(1))
    # Positive funding means longs pay shorts. Negative funding means shorts pay longs.
    return position.shift(1).fillna(0.0) * funding_rate.where(funding_event, 0.0)


def extract_trades(
    df: pd.DataFrame,
    position: pd.Series,
    fee_bps: float,
    slippage_bps: float,
    include_funding: bool,
    atr_stop_mult: float = 0.0,
    max_trade_loss: float = 0.0,
    max_position_loss: float = 0.0,
    short_rebound_stop: float = 0.0,
) -> pd.DataFrame:
    trades = []
    current = 0.0
    entry_time = None
    entry_price = np.nan
    entry_size = 0.0
    entry_atr = np.nan

    def resolve_exit_price(ts, side, fallback_price):
        if side == 0 or pd.isna(entry_price):
            return fallback_price
        high = df.at[ts, "high"] if "high" in df.columns else fallback_price
        low = df.at[ts, "low"] if "low" in df.columns else fallback_price
        stop_price = np.nan
        if atr_stop_mult > 0 and pd.notna(entry_atr) and entry_atr > 0:
            stop_price = entry_price - np.sign(side) * atr_stop_mult * entry_atr
        if max_trade_loss > 0:
            loss_stop_price = entry_price * (1 - max_trade_loss) if side > 0 else entry_price * (1 + max_trade_loss)
            if pd.isna(stop_price):
                stop_price = loss_stop_price
            elif side > 0:
                stop_price = max(stop_price, loss_stop_price)
            else:
                stop_price = min(stop_price, loss_stop_price)
        if max_position_loss > 0 and abs(side) > 0:
            position_loss_pct = max_position_loss / abs(side)
            account_stop_price = entry_price * (1 - position_loss_pct) if side > 0 else entry_price * (1 + position_loss_pct)
            if pd.isna(stop_price):
                stop_price = account_stop_price
            elif side > 0:
                stop_price = max(stop_price, account_stop_price)
            else:
                stop_price = min(stop_price, account_stop_price)
        if pd.notna(stop_price):
            if side > 0 and low <= stop_price:
                return stop_price
            if side < 0 and high >= stop_price:
                return stop_price
        if side < 0 and short_rebound_stop > 0:
            rebound_price = entry_price * (1 + short_rebound_stop)
            if high >= rebound_price:
                return rebound_price
        return fallback_price

    for ts in position.index:
        pos = position.at[ts]
        if current == 0 and pos != 0:
            current = pos
            entry_size = abs(pos)
            entry_time = ts
            entry_price = df.at[ts, "close"]
            entry_atr = df.at[ts, "atr14"] if "atr14" in df.columns else np.nan
        elif current != 0 and np.sign(pos) != np.sign(current):
            exit_time = ts
            exit_price = resolve_exit_price(ts, current, df.at[ts, "close"])
            gross = current * (exit_price / entry_price - 1)
            turnover_cost = abs(current) * 2 * (fee_bps + slippage_bps) / 10000
            funding_cost = 0.0
            if include_funding:
                segment = position.loc[entry_time:exit_time]
                funding_cost = float(funding_cost_series(df.loc[segment.index], segment).sum())
            net = gross - turnover_cost - funding_cost
            trades.append(
                {
                    "entry_time": entry_time,
                    "exit_time": exit_time,
                    "side": "long" if current > 0 else "short",
                    "size": entry_size,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "bars": int((exit_time - entry_time).total_seconds() // 3600),
                    "gross_return": gross,
                    "turnover_cost": turnover_cost,
                    "funding_cost": funding_cost,
                    "net_return": net,
                }
            )
            current = 0.0
            entry_time = None
            entry_price = np.nan
            entry_size = 0.0
            entry_atr = np.nan
            if pos != 0:
                current = pos
                entry_size = abs(pos)
                entry_time = ts
                entry_price = df.at[ts, "close"]
                entry_atr = df.at[ts, "atr14"] if "atr14" in df.columns else np.nan

    if current != 0 and entry_time is not None:
        exit_time = position.index[-1]
        exit_price = resolve_exit_price(exit_time, current, df.at[exit_time, "close"])
        gross = current * (exit_price / entry_price - 1)
        turnover_cost = abs(current) * 2 * (fee_bps + slippage_bps) / 10000
        funding_cost = 0.0
        if include_funding:
            segment = position.loc[entry_time:exit_time]
            funding_cost = float(funding_cost_series(df.loc[segment.index], segment).sum())
        net = gross - turnover_cost - funding_cost
        trades.append(
            {
                "entry_time": entry_time,
                "exit_time": exit_time,
                "side": "long" if current > 0 else "short",
                "size": entry_size,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "bars": int((exit_time - entry_time).total_seconds() // 3600),
                "gross_return": gross,
                "turnover_cost": turnover_cost,
                "funding_cost": funding_cost,
                "net_return": net,
            }
        )

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        return trades_df
    loss_group = (trades_df["net_return"] >= 0).cumsum()
    trades_df["loss_streak_id"] = loss_group.where(trades_df["net_return"] < 0)
    trades_df["consecutive_loss_count"] = trades_df.groupby("loss_streak_id").cumcount() + 1
    trades_df.loc[trades_df["net_return"] >= 0, "consecutive_loss_count"] = 0
    return trades_df


def evaluate(
    df: pd.DataFrame,
    position: pd.Series,
    fee_bps: float,
    slippage_bps: float = 0.0,
    include_funding: bool = False,
    initial_capital: float = 1.0,
) -> dict[str, float]:
    returns = df["close"].pct_change().fillna(0.0)
    turnover = position.diff().abs().fillna(position.abs())
    trade_cost = turnover * (fee_bps + slippage_bps) / 10000
    funding_cost = funding_cost_series(df, position) if include_funding else 0.0
    strategy_return = position * returns - trade_cost - funding_cost
    equity = (1 + strategy_return).cumprod()

    entries = (position != 0) & (position.shift(1).fillna(0.0) != position)
    long_entries = entries & (position > 0)
    short_entries = entries & (position < 0)
    years = max((df.index[-1] - df.index[0]).days / 365.25, 1 / 365.25)
    sharpe = strategy_return.mean() / strategy_return.std() * np.sqrt(24 * 365) if strategy_return.std() > 0 else 0.0
    long_return = (1 + strategy_return.where(position > 0, 0.0)).prod() - 1
    short_return = (1 + strategy_return.where(position < 0, 0.0)).prod() - 1
    active_returns = strategy_return.loc[position != 0]

    return {
        "total_return": float(equity.iloc[-1] - 1),
        "initial_capital": float(initial_capital),
        "final_equity": float(initial_capital * equity.iloc[-1]),
        "profit": float(initial_capital * (equity.iloc[-1] - 1)),
        "annual_return": float(equity.iloc[-1] ** (1 / years) - 1),
        "max_drawdown": max_drawdown(equity),
        "sharpe": float(sharpe),
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


def threshold_grid_report(
    df: pd.DataFrame,
    pred: pd.Series,
    long_thresholds: list[float],
    short_thresholds: list[float],
    mode: str,
    fee_bps: float,
    slippage_bps: float,
    include_funding: bool,
    min_hold_hours: int,
    max_consecutive_losses: int,
    cooldown_hours: int,
    atr_stop_mult: float,
    max_trade_loss: float,
    max_position_loss: float,
    short_rebound_stop: float,
    drawdown_reduce_threshold: float,
    drawdown_reduce_factor: float,
    drawdown_pause_threshold: float,
    drawdown_cooldown_hours: int,
    loss_reduce_after: int,
    loss_reduced_size: float,
    vol_pause_quantile: float,
    vol_pause_hours: int,
    position_mode: str,
    base_size: float,
    mid_size: float,
    extreme_size: float,
    max_size: float,
    long_extreme_threshold: float,
    short_extreme_threshold: float,
    long_max_threshold: float,
    short_max_threshold: float,
    leverage: float,
    btc_trend: pd.Series | None,
    btc_trend_filter: str,
    btc_countertrend_size: float,
    initial_capital: float,
    tighten_after_losses: int,
    tightened_long_threshold: float,
    tightened_short_threshold: float,
    tighten_reset_hours: int,
    daily_loss_limit: float,
    rolling_loss_window_hours: int,
    rolling_loss_limit: float,
    rolling_loss_cooldown_hours: int,
    loss_streak_loss_limit: float,
    loss_streak_cooldown_hours: int,
) -> pd.DataFrame:
    rows = []
    for long_threshold in long_thresholds:
        if mode == "long_only":
            desired = make_position(
                pred, long_threshold, 0.0, mode, position_mode, base_size, mid_size, extreme_size, max_size,
                long_extreme_threshold, short_extreme_threshold, long_max_threshold, short_max_threshold, leverage
            )
            desired = apply_btc_trend_filter(desired, btc_trend, btc_trend_filter, btc_countertrend_size * leverage)
            position = apply_execution_constraints(
                df, pred, desired, long_threshold, 0.0, min_hold_hours, max_consecutive_losses, cooldown_hours, fee_bps, slippage_bps,
                atr_stop_mult, max_trade_loss, max_position_loss, short_rebound_stop,
                drawdown_reduce_threshold, drawdown_reduce_factor, drawdown_pause_threshold, drawdown_cooldown_hours,
                loss_reduce_after, loss_reduced_size,
                vol_pause_quantile, vol_pause_hours, tighten_after_losses, tightened_long_threshold,
                tightened_short_threshold, tighten_reset_hours, daily_loss_limit, rolling_loss_window_hours,
                rolling_loss_limit, rolling_loss_cooldown_hours, loss_streak_loss_limit, loss_streak_cooldown_hours
            )
            rows.append(
                {
                    "mode": mode,
                    "long_threshold": long_threshold,
                    "short_threshold": np.nan,
                    **evaluate(df, position, fee_bps, slippage_bps, include_funding, initial_capital),
                }
            )
        else:
            for short_threshold in short_thresholds:
                if short_threshold >= long_threshold:
                    continue
                desired = make_position(
                    pred, long_threshold, short_threshold, mode, position_mode, base_size, mid_size, extreme_size, max_size,
                    long_extreme_threshold, short_extreme_threshold, long_max_threshold, short_max_threshold, leverage
                )
                desired = apply_btc_trend_filter(desired, btc_trend, btc_trend_filter, btc_countertrend_size * leverage)
                position = apply_execution_constraints(
                    df, pred, desired, long_threshold, short_threshold, min_hold_hours, max_consecutive_losses,
                    cooldown_hours, fee_bps, slippage_bps,
                    atr_stop_mult, max_trade_loss, max_position_loss, short_rebound_stop,
                    drawdown_reduce_threshold, drawdown_reduce_factor, drawdown_pause_threshold, drawdown_cooldown_hours,
                    loss_reduce_after, loss_reduced_size,
                    vol_pause_quantile, vol_pause_hours, tighten_after_losses, tightened_long_threshold,
                    tightened_short_threshold, tighten_reset_hours, daily_loss_limit, rolling_loss_window_hours,
                    rolling_loss_limit, rolling_loss_cooldown_hours, loss_streak_loss_limit, loss_streak_cooldown_hours
                )
                rows.append(
                    {
                        "mode": mode,
                        "long_threshold": long_threshold,
                        "short_threshold": short_threshold,
                        **evaluate(df, position, fee_bps, slippage_bps, include_funding, initial_capital),
                    }
                )
    result = pd.DataFrame(rows)
    result["score"] = result["annual_return"] + 0.5 * result["max_drawdown"] + 0.05 * result["sharpe"]
    return result.sort_values("score", ascending=False)


def year_report(
    df: pd.DataFrame,
    pred: pd.Series,
    long_threshold: float,
    short_threshold: float,
    mode: str,
    fee_bps: float,
    slippage_bps: float,
    include_funding: bool,
    min_hold_hours: int,
    max_consecutive_losses: int,
    cooldown_hours: int,
    atr_stop_mult: float,
    max_trade_loss: float,
    max_position_loss: float,
    short_rebound_stop: float,
    drawdown_reduce_threshold: float,
    drawdown_reduce_factor: float,
    drawdown_pause_threshold: float,
    drawdown_cooldown_hours: int,
    loss_reduce_after: int,
    loss_reduced_size: float,
    vol_pause_quantile: float,
    vol_pause_hours: int,
    position_mode: str,
    base_size: float,
    mid_size: float,
    extreme_size: float,
    max_size: float,
    long_extreme_threshold: float,
    short_extreme_threshold: float,
    long_max_threshold: float,
    short_max_threshold: float,
    leverage: float,
    btc_trend: pd.Series | None,
    btc_trend_filter: str,
    btc_countertrend_size: float,
    initial_capital: float,
    tighten_after_losses: int,
    tightened_long_threshold: float,
    tightened_short_threshold: float,
    tighten_reset_hours: int,
    daily_loss_limit: float,
    rolling_loss_window_hours: int,
    rolling_loss_limit: float,
    rolling_loss_cooldown_hours: int,
    loss_streak_loss_limit: float,
    loss_streak_cooldown_hours: int,
) -> pd.DataFrame:
    rows = []
    desired = make_position(
        pred, long_threshold, short_threshold, mode, position_mode, base_size, mid_size, extreme_size, max_size,
        long_extreme_threshold, short_extreme_threshold, long_max_threshold, short_max_threshold, leverage
    )
    desired = apply_btc_trend_filter(desired, btc_trend, btc_trend_filter, btc_countertrend_size * leverage)
    position = apply_execution_constraints(
        df, pred, desired, long_threshold, short_threshold, min_hold_hours, max_consecutive_losses,
        cooldown_hours, fee_bps, slippage_bps,
        atr_stop_mult, max_trade_loss, max_position_loss, short_rebound_stop,
        drawdown_reduce_threshold, drawdown_reduce_factor, drawdown_pause_threshold, drawdown_cooldown_hours,
        loss_reduce_after, loss_reduced_size,
        vol_pause_quantile, vol_pause_hours, tighten_after_losses, tightened_long_threshold,
        tightened_short_threshold, tighten_reset_hours, daily_loss_limit, rolling_loss_window_hours,
        rolling_loss_limit, rolling_loss_cooldown_hours, loss_streak_loss_limit, loss_streak_cooldown_hours
    )
    for year, group in df.groupby(df.index.year):
        pos = position.reindex(group.index).fillna(0.0)
        metrics = evaluate(group, pos, fee_bps, slippage_bps, include_funding, initial_capital)
        metrics["year"] = year
        rows.append(metrics)
    return pd.DataFrame(rows).set_index("year")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest pivot SR LightGBM probability signals.")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2024-12-31 23:59:59")
    parser.add_argument("--select-start", default=None)
    parser.add_argument("--select-end", default=None)
    parser.add_argument("--test-start", default=None)
    parser.add_argument("--test-end", default=None)
    parser.add_argument("--yearly-start", default="2020-01-01")
    parser.add_argument("--yearly-end", default="2024-12-31 23:59:59")
    parser.add_argument("--mode", choices=["long_only", "long_short"], default="long_short")
    parser.add_argument("--long-thresholds", default="0.52,0.55,0.58,0.60,0.62")
    parser.add_argument("--short-thresholds", default="0.48,0.45,0.42,0.40,0.38")
    parser.add_argument("--fee-bps", default=5.0, type=float)
    parser.add_argument("--slippage-bps", default=2.0, type=float)
    parser.add_argument("--include-funding", action="store_true")
    parser.add_argument("--min-hold-hours", default=0, type=int)
    parser.add_argument("--max-consecutive-losses", default=0, type=int)
    parser.add_argument("--cooldown-hours", default=24, type=int)
    parser.add_argument("--atr-stop-mult", default=0.0, type=float)
    parser.add_argument("--max-trade-loss", default=0.0, type=float)
    parser.add_argument("--max-position-loss", default=0.0, type=float)
    parser.add_argument("--short-rebound-stop", default=0.0, type=float)
    parser.add_argument("--drawdown-reduce-threshold", default=0.0, type=float)
    parser.add_argument("--drawdown-reduce-factor", default=0.5, type=float)
    parser.add_argument("--drawdown-pause-threshold", default=0.0, type=float)
    parser.add_argument("--drawdown-cooldown-hours", default=72, type=int)
    parser.add_argument("--loss-reduce-after", default=0, type=int)
    parser.add_argument("--loss-reduced-size", default=0.25, type=float)
    parser.add_argument("--vol-pause-quantile", default=0.0, type=float)
    parser.add_argument("--vol-pause-hours", default=24, type=int)
    parser.add_argument("--position-mode", choices=["fixed", "probability"], default="fixed")
    parser.add_argument("--base-size", default=0.5, type=float)
    parser.add_argument("--mid-size", default=0.65, type=float)
    parser.add_argument("--extreme-size", default=0.8, type=float)
    parser.add_argument("--max-size", default=0.8, type=float)
    parser.add_argument("--initial-capital", default=1.0, type=float)
    parser.add_argument("--leverage", default=1.0, type=float)
    parser.add_argument("--btc-trend-data", default=None, type=Path)
    parser.add_argument("--btc-trend-column", default="trend_1d_ema_gap")
    parser.add_argument("--btc-trend-filter", choices=["none", "confirm", "reduce"], default="none")
    parser.add_argument("--btc-countertrend-size", default=0.25, type=float)
    parser.add_argument("--long-extreme-threshold", default=0.58, type=float)
    parser.add_argument("--short-extreme-threshold", default=0.42, type=float)
    parser.add_argument("--long-max-threshold", default=0.65, type=float)
    parser.add_argument("--short-max-threshold", default=0.35, type=float)
    parser.add_argument("--fixed-long-threshold", default=None, type=float)
    parser.add_argument("--fixed-short-threshold", default=None, type=float)
    parser.add_argument("--tighten-after-losses", default=0, type=int)
    parser.add_argument("--tightened-long-threshold", default=0.55, type=float)
    parser.add_argument("--tightened-short-threshold", default=0.38, type=float)
    parser.add_argument("--tighten-reset-hours", default=0, type=int)
    parser.add_argument("--daily-loss-limit", default=0.0, type=float)
    parser.add_argument("--rolling-loss-window-hours", default=0, type=int)
    parser.add_argument("--rolling-loss-limit", default=0.0, type=float)
    parser.add_argument("--rolling-loss-cooldown-hours", default=48, type=int)
    parser.add_argument("--loss-streak-loss-limit", default=0.0, type=float)
    parser.add_argument("--loss-streak-cooldown-hours", default=96, type=int)
    args = parser.parse_args()

    bundle = joblib.load(args.model)
    model = bundle["model"]
    features = bundle["features"]

    df = load_dataset(args.data)
    data = build_model_data(df, bundle)
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=features + ["close"])
    pred = pd.Series(model.predict_proba(data[features])[:, 1], index=data.index, name="pred")
    btc_trend = None
    if args.btc_trend_data is not None and args.btc_trend_filter != "none":
        btc_data = load_dataset(args.btc_trend_data)
        if args.btc_trend_column not in btc_data.columns:
            raise ValueError(f"BTC trend column not found: {args.btc_trend_column}")
        btc_trend = btc_data[args.btc_trend_column]

    if args.select_start and args.select_end and args.test_start and args.test_end:
        select = data.loc[args.select_start : args.select_end]
        select_pred = pred.reindex(select.index).dropna()
        select = select.reindex(select_pred.index)
        selection_report = threshold_grid_report(
            select,
            select_pred,
            parse_float_list(args.long_thresholds),
            parse_float_list(args.short_thresholds),
            args.mode,
            args.fee_bps,
            args.slippage_bps,
            args.include_funding,
            args.min_hold_hours,
            args.max_consecutive_losses,
            args.cooldown_hours,
            args.atr_stop_mult,
            args.max_trade_loss,
            args.max_position_loss,
            args.short_rebound_stop,
            args.drawdown_reduce_threshold,
            args.drawdown_reduce_factor,
            args.drawdown_pause_threshold,
            args.drawdown_cooldown_hours,
            args.loss_reduce_after,
            args.loss_reduced_size,
            args.vol_pause_quantile,
            args.vol_pause_hours,
            args.position_mode,
            args.base_size,
            args.mid_size,
            args.extreme_size,
            args.max_size,
            args.long_extreme_threshold,
            args.short_extreme_threshold,
            args.long_max_threshold,
            args.short_max_threshold,
            args.leverage,
            btc_trend,
            args.btc_trend_filter,
            args.btc_countertrend_size,
            args.initial_capital,
            args.tighten_after_losses,
            args.tightened_long_threshold,
            args.tightened_short_threshold,
            args.tighten_reset_hours,
            args.daily_loss_limit,
            args.rolling_loss_window_hours,
            args.rolling_loss_limit,
            args.rolling_loss_cooldown_hours,
            args.loss_streak_loss_limit,
            args.loss_streak_cooldown_hours,
        )
        best = selection_report.iloc[0]
        if args.fixed_long_threshold is not None:
            best["long_threshold"] = args.fixed_long_threshold
        if args.fixed_short_threshold is not None:
            best["short_threshold"] = args.fixed_short_threshold

        test = data.loc[args.test_start : args.test_end]
        test_pred = pred.reindex(test.index).dropna()
        test = test.reindex(test_pred.index)
        report = threshold_grid_report(
            test,
            test_pred,
            [float(best["long_threshold"])],
            [0.0 if pd.isna(best["short_threshold"]) else float(best["short_threshold"])],
            args.mode,
            args.fee_bps,
            args.slippage_bps,
            args.include_funding,
            args.min_hold_hours,
            args.max_consecutive_losses,
            args.cooldown_hours,
            args.atr_stop_mult,
            args.max_trade_loss,
            args.max_position_loss,
            args.short_rebound_stop,
            args.drawdown_reduce_threshold,
            args.drawdown_reduce_factor,
            args.drawdown_pause_threshold,
            args.drawdown_cooldown_hours,
            args.loss_reduce_after,
            args.loss_reduced_size,
            args.vol_pause_quantile,
            args.vol_pause_hours,
            args.position_mode,
            args.base_size,
            args.mid_size,
            args.extreme_size,
            args.max_size,
            args.long_extreme_threshold,
            args.short_extreme_threshold,
            args.long_max_threshold,
            args.short_max_threshold,
            args.leverage,
            btc_trend,
            args.btc_trend_filter,
            args.btc_countertrend_size,
            args.initial_capital,
            args.tighten_after_losses,
            args.tightened_long_threshold,
            args.tightened_short_threshold,
            args.tighten_reset_hours,
            args.daily_loss_limit,
            args.rolling_loss_window_hours,
            args.rolling_loss_limit,
            args.rolling_loss_cooldown_hours,
            args.loss_streak_loss_limit,
            args.loss_streak_cooldown_hours,
        )
        selection_out = args.out.with_name(args.out.stem + "_selection.csv")
        selection_report.to_csv(selection_out, index=False)
    else:
        test = data.loc[args.start : args.end]
        test_pred = pred.reindex(test.index).dropna()
        test = test.reindex(test_pred.index)
        selection_report = None

        if args.fixed_long_threshold is not None:
            long_thresholds = [args.fixed_long_threshold]
            short_thresholds = [args.fixed_short_threshold if args.fixed_short_threshold is not None else 0.0]
        else:
            long_thresholds = parse_float_list(args.long_thresholds)
            short_thresholds = parse_float_list(args.short_thresholds)

        report = threshold_grid_report(
            test,
            test_pred,
            long_thresholds,
            short_thresholds,
            args.mode,
            args.fee_bps,
            args.slippage_bps,
            args.include_funding,
            args.min_hold_hours,
            args.max_consecutive_losses,
            args.cooldown_hours,
            args.atr_stop_mult,
            args.max_trade_loss,
            args.max_position_loss,
            args.short_rebound_stop,
            args.drawdown_reduce_threshold,
            args.drawdown_reduce_factor,
            args.drawdown_pause_threshold,
            args.drawdown_cooldown_hours,
            args.loss_reduce_after,
            args.loss_reduced_size,
            args.vol_pause_quantile,
            args.vol_pause_hours,
            args.position_mode,
            args.base_size,
            args.mid_size,
            args.extreme_size,
            args.max_size,
            args.long_extreme_threshold,
            args.short_extreme_threshold,
            args.long_max_threshold,
            args.short_max_threshold,
            args.leverage,
            btc_trend,
            args.btc_trend_filter,
            args.btc_countertrend_size,
            args.initial_capital,
            args.tighten_after_losses,
            args.tightened_long_threshold,
            args.tightened_short_threshold,
            args.tighten_reset_hours,
            args.daily_loss_limit,
            args.rolling_loss_window_hours,
            args.rolling_loss_limit,
            args.rolling_loss_cooldown_hours,
            args.loss_streak_loss_limit,
            args.loss_streak_cooldown_hours,
        )
        best = report.iloc[0]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.out, index=False)

    yearly_data = data.loc[args.yearly_start : args.yearly_end]
    yearly_pred = pred.reindex(yearly_data.index).dropna()
    yearly_data = yearly_data.reindex(yearly_pred.index)
    yearly = year_report(
        yearly_data,
        yearly_pred,
        float(best["long_threshold"]),
        0.0 if pd.isna(best["short_threshold"]) else float(best["short_threshold"]),
        args.mode,
        args.fee_bps,
        args.slippage_bps,
        args.include_funding,
        args.min_hold_hours,
        args.max_consecutive_losses,
        args.cooldown_hours,
        args.atr_stop_mult,
        args.max_trade_loss,
        args.max_position_loss,
        args.short_rebound_stop,
        args.drawdown_reduce_threshold,
        args.drawdown_reduce_factor,
        args.drawdown_pause_threshold,
        args.drawdown_cooldown_hours,
        args.loss_reduce_after,
        args.loss_reduced_size,
        args.vol_pause_quantile,
        args.vol_pause_hours,
        args.position_mode,
        args.base_size,
        args.mid_size,
        args.extreme_size,
        args.max_size,
        args.long_extreme_threshold,
        args.short_extreme_threshold,
        args.long_max_threshold,
        args.short_max_threshold,
        args.leverage,
        btc_trend,
        args.btc_trend_filter,
        args.btc_countertrend_size,
        args.initial_capital,
        args.tighten_after_losses,
        args.tightened_long_threshold,
        args.tightened_short_threshold,
        args.tighten_reset_hours,
        args.daily_loss_limit,
        args.rolling_loss_window_hours,
        args.rolling_loss_limit,
        args.rolling_loss_cooldown_hours,
        args.loss_streak_loss_limit,
        args.loss_streak_cooldown_hours,
    )
    yearly_out = args.out.with_name(args.out.stem + "_yearly.csv")
    yearly.to_csv(yearly_out)

    pred_out = args.out.with_name(args.out.stem + "_predictions.csv")
    test.assign(pred=test_pred).to_csv(pred_out, index_label="timestamp")

    best_position = make_position(
        yearly_pred,
        float(best["long_threshold"]),
        0.0 if pd.isna(best["short_threshold"]) else float(best["short_threshold"]),
        args.mode,
        args.position_mode,
        args.base_size,
        args.mid_size,
        args.extreme_size,
        args.max_size,
        args.long_extreme_threshold,
        args.short_extreme_threshold,
        args.long_max_threshold,
        args.short_max_threshold,
        args.leverage,
    )
    best_position = apply_btc_trend_filter(
        best_position,
        btc_trend,
        args.btc_trend_filter,
        args.btc_countertrend_size * args.leverage,
    )
    best_position = apply_execution_constraints(
        yearly_data,
        yearly_pred,
        best_position,
        float(best["long_threshold"]),
        0.0 if pd.isna(best["short_threshold"]) else float(best["short_threshold"]),
        args.min_hold_hours,
        args.max_consecutive_losses,
        args.cooldown_hours,
        args.fee_bps,
        args.slippage_bps,
        args.atr_stop_mult,
        args.max_trade_loss,
        args.max_position_loss,
        args.short_rebound_stop,
        args.drawdown_reduce_threshold,
        args.drawdown_reduce_factor,
        args.drawdown_pause_threshold,
        args.drawdown_cooldown_hours,
        args.loss_reduce_after,
        args.loss_reduced_size,
        args.vol_pause_quantile,
        args.vol_pause_hours,
        args.tighten_after_losses,
        args.tightened_long_threshold,
        args.tightened_short_threshold,
        args.tighten_reset_hours,
        args.daily_loss_limit,
        args.rolling_loss_window_hours,
        args.rolling_loss_limit,
        args.rolling_loss_cooldown_hours,
        args.loss_streak_loss_limit,
        args.loss_streak_cooldown_hours,
    )
    trades = extract_trades(
        yearly_data,
        best_position,
        args.fee_bps,
        args.slippage_bps,
        args.include_funding,
        args.atr_stop_mult,
        args.max_trade_loss,
        args.max_position_loss,
        args.short_rebound_stop,
    )
    trades_out = args.out.with_name(args.out.stem + "_trades.csv")
    trades.to_csv(trades_out, index=False)

    worst_trades_out = args.out.with_name(args.out.stem + "_worst_trades.csv")
    if not trades.empty:
        trades.sort_values("net_return").head(30).to_csv(worst_trades_out, index=False)
        loss_streaks = (
            trades[trades["net_return"] < 0]
            .groupby("loss_streak_id")
            .agg(
                start=("entry_time", "first"),
                end=("exit_time", "last"),
                count=("net_return", "size"),
                total_loss=("net_return", "sum"),
                worst_trade=("net_return", "min"),
            )
            .sort_values(["count", "total_loss"], ascending=[False, True])
        )
    else:
        pd.DataFrame().to_csv(worst_trades_out, index=False)
        loss_streaks = pd.DataFrame()
    loss_streaks_out = args.out.with_name(args.out.stem + "_loss_streaks.csv")
    loss_streaks.to_csv(loss_streaks_out)

    period_label = f"{args.test_start} -> {args.test_end}" if args.test_start else f"{args.start} -> {args.end}"
    print(f"Prediction summary for {period_label}")
    print(test_pred.describe(percentiles=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]).to_string())
    if selection_report is not None:
        print("\nSelected threshold on validation")
        print(selection_report.head(10).to_string(index=False))
    print("\nTop threshold results")
    print(report.head(20).to_string(index=False))
    print(f"\nSaved report to {args.out}")
    if selection_report is not None:
        print(f"Saved selection report to {selection_out}")
    print(f"Saved yearly report to {yearly_out}")
    print(f"Saved predictions to {pred_out}")
    print(f"Saved trades to {trades_out}")
    print(f"Saved worst trades to {worst_trades_out}")
    print(f"Saved loss streaks to {loss_streaks_out}")
    if not trades.empty:
        print("\nWorst trades")
        print(trades.sort_values("net_return").head(10).to_string(index=False))
        print("\nWorst loss streaks")
        print(loss_streaks.head(10).to_string())


if __name__ == "__main__":
    main()
