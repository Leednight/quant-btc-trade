from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from optimize_rule_strategy import build_position, evaluate, load_dataset
    from train_lightgbm import select_feature_columns
except ModuleNotFoundError:
    from scripts.optimize_rule_strategy import build_position, evaluate, load_dataset
    from scripts.train_lightgbm import select_feature_columns


DEFAULT_RULES = {
    "BTCUSDT": {
        "ema_fast": 12,
        "ema_slow": 96,
        "trend_threshold": 0.0,
        "use_macd": 0,
        "volume_filter": -999.0,
        "funding_z_cap": 999.0,
        "allow_short": 0,
    },
    "ETHUSDT": {
        "ema_fast": 48,
        "ema_slow": 200,
        "trend_threshold": 0.0,
        "use_macd": 0,
        "volume_filter": -999.0,
        "funding_z_cap": 999.0,
        "allow_short": 0,
    },
}

RULE_GRIDS = {
    "BTCUSDT": [
        {"ema_fast": 12, "ema_slow": 96},
        {"ema_fast": 20, "ema_slow": 96},
        {"ema_fast": 30, "ema_slow": 60},
        {"ema_fast": 30, "ema_slow": 96},
        {"ema_fast": 12, "ema_slow": 120},
        {"ema_fast": 30, "ema_slow": 48},
        {"ema_fast": 48, "ema_slow": 96},
    ],
    "ETHUSDT": [
        {"ema_fast": 48, "ema_slow": 200},
        {"ema_fast": 30, "ema_slow": 200},
        {"ema_fast": 20, "ema_slow": 200},
        {"ema_fast": 20, "ema_slow": 60},
        {"ema_fast": 30, "ema_slow": 60},
        {"ema_fast": 48, "ema_slow": 96},
        {"ema_fast": 20, "ema_slow": 96},
    ],
}

RULE_DEFAULTS = {
    "trend_threshold": 0.0,
    "use_macd": 0,
    "volume_filter": -999.0,
    "funding_z_cap": 999.0,
    "allow_short": 0,
}


def infer_symbol(df: pd.DataFrame, path: Path) -> str:
    if "symbol" in df.columns and df["symbol"].notna().any():
        return str(df["symbol"].dropna().iloc[0]).upper()
    name = path.name.upper()
    if "ETHUSDT" in name:
        return "ETHUSDT"
    return "BTCUSDT"


def add_horizon_filter_label(df: pd.DataFrame, horizon: int, atr_mult: float) -> pd.DataFrame:
    df = df.copy()
    df["filter_future_return"] = df["close"].shift(-horizon) / df["close"] - 1
    df["filter_atr_return"] = df["atr14_pct"] * atr_mult
    df["filter_label"] = (df["filter_future_return"] > df["filter_atr_return"]).astype(int)
    return df


def add_trade_labels(
    df: pd.DataFrame,
    position: pd.Series,
    fee_bps: float,
    min_trade_return: float,
    atr_mult: float,
    rule_name: str = "base",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    df = df.copy()
    position = position.reindex(df.index).fillna(0.0)
    trade_id = pd.Series(np.nan, index=df.index)
    rows = []
    current_id = 0
    start = None
    side = 0.0

    for ts, pos in position.items():
        if side == 0 and pos != 0:
            start = ts
            side = pos
            current_id += 1
        elif side != 0 and pos != side:
            end = position.index[position.index.get_loc(ts) - 1]
            rows.append((current_id, start, end, side))
            start = ts if pos != 0 else None
            side = pos
            if side != 0:
                current_id += 1

        if side != 0:
            trade_id.loc[ts] = current_id

    if side != 0 and start is not None:
        rows.append((current_id, start, position.index[-1], side))

    trade_rows = []
    for tid, entry_ts, exit_ts, trade_side in rows:
        entry_close = df.at[entry_ts, "close"]
        exit_close = df.at[exit_ts, "close"]
        gross_return = trade_side * (exit_close / entry_close - 1)
        net_return = gross_return - 2 * fee_bps / 10000
        atr_threshold = float(df.at[entry_ts, "atr14_pct"]) * atr_mult
        label_threshold = max(min_trade_return, atr_threshold)
        trade_rows.append(
            {
                "trade_id": tid,
                "entry_time": entry_ts,
                "exit_time": exit_ts,
                "side": trade_side,
                "trade_return": net_return,
                "trade_bars": int(df.loc[entry_ts:exit_ts].shape[0]),
                "filter_atr_return": atr_threshold,
                "filter_label": int(net_return > label_threshold),
                "rule_name": rule_name,
            }
        )

    trades = pd.DataFrame(trade_rows)
    if trades.empty:
        raise ValueError("No trades found from the base rule position.")

    entry_features = df.reindex(trades["entry_time"]).copy()
    entry_features.index = pd.DatetimeIndex(trades["entry_time"])
    entry_features["trade_id"] = trades["trade_id"].to_numpy()
    entry_features["side"] = trades["side"].to_numpy()
    entry_features["trade_return"] = trades["trade_return"].to_numpy()
    entry_features["trade_bars"] = trades["trade_bars"].to_numpy()
    entry_features["filter_atr_return"] = trades["filter_atr_return"].to_numpy()
    entry_features["filter_label"] = trades["filter_label"].to_numpy()
    entry_features["rule_name"] = trades["rule_name"].to_numpy()
    return entry_features, trades, trade_id


def make_rule_params(base: dict[str, float], override: dict[str, float]) -> dict[str, float]:
    params = {**RULE_DEFAULTS, **base, **override}
    return params


def build_multi_rule_trade_samples(
    df: pd.DataFrame,
    symbol: str,
    fee_bps: float,
    min_trade_return: float,
    atr_mult: float,
) -> pd.DataFrame:
    samples = []
    base = DEFAULT_RULES[symbol]
    for override in RULE_GRIDS[symbol]:
        params = make_rule_params(base, override)
        if params["ema_fast"] >= params["ema_slow"]:
            continue
        position = build_position(df, **params)
        rule_name = f"ema{int(params['ema_fast'])}_{int(params['ema_slow'])}"
        entry_features, _trades, _trade_id = add_trade_labels(
            df,
            position,
            fee_bps=fee_bps,
            min_trade_return=min_trade_return,
            atr_mult=atr_mult,
            rule_name=rule_name,
        )
        entry_features["rule_ema_fast"] = params["ema_fast"]
        entry_features["rule_ema_slow"] = params["ema_slow"]
        entry_features["rule_ema_ratio"] = params["ema_fast"] / params["ema_slow"]
        samples.append(entry_features)

    if not samples:
        raise ValueError(f"No multi-rule samples were generated for {symbol}.")

    data = pd.concat(samples).sort_index()
    return data


def split_walk_forward(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df.loc["2020-01-01":"2022-12-31 23:59:59"]
    valid = df.loc["2023-01-01":"2023-12-31 23:59:59"]
    test = df.loc["2024-01-01":"2024-12-31 23:59:59"]
    return train, valid, test


def apply_probability_filter(position: pd.Series, pred: pd.Series, threshold: float) -> pd.Series:
    filtered = position.copy()
    filtered.loc[pred.reindex(filtered.index).fillna(0.0) < threshold] = 0.0
    return filtered


def apply_trade_probability_filter(
    position: pd.Series,
    trade_id: pd.Series,
    entry_pred: pd.Series,
    threshold: float,
) -> pd.Series:
    filtered = position.copy()
    pred_by_trade = {}
    for entry_time, probability in entry_pred.items():
        tid = trade_id.get(entry_time, np.nan)
        if pd.notna(tid):
            pred_by_trade[tid] = probability

    for tid in pd.Series(trade_id.dropna().unique()).sort_values():
        probability = pred_by_trade.get(tid, 0.0)
        if probability < threshold:
            filtered.loc[trade_id == tid] = 0.0
    return filtered


def select_filter_features(df: pd.DataFrame) -> list[str]:
    blocked = {
        "filter_label",
        "filter_future_return",
        "filter_atr_return",
        "rule_position",
        "trade_id",
        "side",
        "trade_return",
        "trade_bars",
        "rule_name",
    }
    return [col for col in select_feature_columns(df) if col not in blocked]


def threshold_report(
    df: pd.DataFrame,
    base_position: pd.Series,
    trade_id: pd.Series,
    pred: pd.Series,
    thresholds: list[float],
    fee_bps: float,
    trade_level: bool,
) -> pd.DataFrame:
    rows = []
    base_metrics = evaluate(df, base_position, fee_bps)
    rows.append({"threshold": "rule_only", **base_metrics})
    for threshold in thresholds:
        if trade_level:
            filtered = apply_trade_probability_filter(base_position, trade_id, pred, threshold)
        else:
            filtered = apply_probability_filter(base_position, pred, threshold)
        rows.append({"threshold": threshold, **evaluate(df, filtered, fee_bps)})
    return pd.DataFrame(rows)


def safe_auc(y_true: pd.Series, pred: pd.Series) -> float:
    from sklearn.metrics import roc_auc_score

    if y_true.nunique() < 2:
        return float("nan")
    return float(roc_auc_score(y_true, pred))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a LightGBM filter for an EMA trend rule strategy.")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--model-out", default=None, type=Path)
    parser.add_argument("--report-out", default=None, type=Path)
    parser.add_argument("--horizon", default=24, type=int, help="Filter label horizon in 1H bars.")
    parser.add_argument("--label-mode", choices=["trade", "horizon"], default="trade")
    parser.add_argument("--sample-mode", choices=["single_rule", "multi_rule"], default="multi_rule")
    parser.add_argument("--atr-mult", default=0.0, type=float, help="Positive label threshold as ATR percent multiple.")
    parser.add_argument("--min-trade-return", default=0.0, type=float, help="Minimum net trade return for a positive trade label.")
    parser.add_argument("--fee-bps", default=5.0, type=float)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.40, 0.45, 0.50, 0.55, 0.60])
    args = parser.parse_args()

    import joblib
    import lightgbm as lgb
    from sklearn.metrics import classification_report

    df = load_dataset(args.data)
    symbol = (args.symbol or infer_symbol(df, args.data)).upper()
    if symbol not in DEFAULT_RULES:
        raise ValueError(f"No default rule params for {symbol}. Pass BTCUSDT or ETHUSDT.")

    rule_params = DEFAULT_RULES[symbol]
    base_position = build_position(df, **rule_params)
    df["rule_position"] = base_position

    if args.label_mode == "trade":
        if args.sample_mode == "multi_rule":
            candidates = build_multi_rule_trade_samples(
                df,
                symbol,
                fee_bps=args.fee_bps,
                min_trade_return=args.min_trade_return,
                atr_mult=args.atr_mult,
            )
            _base_candidates, _base_trades, trade_id = add_trade_labels(
                df,
                base_position,
                fee_bps=args.fee_bps,
                min_trade_return=args.min_trade_return,
                atr_mult=args.atr_mult,
                rule_name="base",
            )
        else:
            candidates, _trades, trade_id = add_trade_labels(
                df,
                base_position,
                fee_bps=args.fee_bps,
                min_trade_return=args.min_trade_return,
                atr_mult=args.atr_mult,
                rule_name="base",
            )
    else:
        df = add_horizon_filter_label(df, args.horizon, args.atr_mult)
        trade_id = pd.Series(np.nan, index=df.index)
        candidates = df.loc[df["rule_position"] != 0].copy()

    feature_cols = select_filter_features(df)
    feature_cols = [col for col in feature_cols if col in candidates.columns]
    drop_cols = feature_cols + ["filter_label", "filter_atr_return"]
    if args.label_mode == "horizon":
        drop_cols.append("filter_future_return")
    candidates = candidates.dropna(subset=drop_cols).copy()
    train, valid, test = split_walk_forward(candidates)

    if train.empty or valid.empty or test.empty:
        raise ValueError("Empty train/valid/test candidate split. Check data range and rule params.")

    x_train, y_train = train[feature_cols], train["filter_label"]
    x_valid, y_valid = valid[feature_cols], valid["filter_label"]
    x_test, y_test = test[feature_cols], test["filter_label"]

    scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=1000,
        learning_rate=0.02,
        num_leaves=7 if args.label_mode == "trade" else 15,
        max_depth=3 if args.label_mode == "trade" else 4,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=5 if args.label_mode == "trade" else 80,
        reg_alpha=0.2,
        reg_lambda=2.0,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        x_train,
        y_train,
        eval_set=[(x_valid, y_valid)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(80), lgb.log_evaluation(50)],
    )

    valid_pred = pd.Series(model.predict_proba(x_valid)[:, 1], index=valid.index)
    test_pred = pd.Series(model.predict_proba(x_test)[:, 1], index=test.index)
    test_class = (test_pred > 0.5).astype(int)

    print("\nCandidate Dataset")
    print(f"Symbol: {symbol}")
    print(f"Label mode: {args.label_mode}")
    print(f"Sample mode: {args.sample_mode}")
    print(f"Rule params: {rule_params}")
    print(f"Train rows: {len(train):,}, positive ratio: {y_train.mean():.4f}")
    print(f"Valid rows: {len(valid):,}, positive ratio: {y_valid.mean():.4f}")
    print(f"Test rows:  {len(test):,}, positive ratio: {y_test.mean():.4f}")
    print(f"Features: {len(feature_cols)}")

    print("\nFilter Metrics")
    print(f"Valid AUC: {safe_auc(y_valid, valid_pred):.4f}")
    print(f"Test AUC:  {safe_auc(y_test, test_pred):.4f}")
    print(classification_report(y_test, test_class, digits=4, zero_division=0))
    print("\nPrediction probability summary")
    print("Valid")
    print(valid_pred.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).to_string())
    print("Test")
    print(test_pred.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).to_string())

    full_pred = pd.Series(np.nan, index=df.index)
    full_pred.loc[valid_pred.index] = valid_pred
    full_pred.loc[test_pred.index] = test_pred

    test_df = df.loc["2024-01-01":"2024-12-31 23:59:59"].dropna(subset=["close"])
    test_base_position = base_position.reindex(test_df.index).fillna(0.0)
    test_trade_id = trade_id.reindex(test_df.index)
    test_report = threshold_report(
        test_df,
        test_base_position,
        test_trade_id,
        test_pred,
        args.thresholds,
        args.fee_bps,
        trade_level=args.label_mode == "trade",
    )
    print("\n2024 Rule Filter Report")
    print(test_report.to_string(index=False))

    model_out = args.model_out or Path(f"models/{symbol.lower()}_rule_filter_lgbm.joblib")
    report_out = args.report_out or Path(f"reports/{symbol.lower()}_rule_filter_report.csv")
    model_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "features": feature_cols,
            "rule_params": rule_params,
            "horizon": args.horizon,
            "label_mode": args.label_mode,
            "sample_mode": args.sample_mode,
            "atr_mult": args.atr_mult,
            "min_trade_return": args.min_trade_return,
            "symbol": symbol,
        },
        model_out,
    )
    test_report.to_csv(report_out, index=False)

    importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print(f"\nSaved model to {model_out}")
    print(f"Saved report to {report_out}")
    print("\nTop 20 features")
    print(importance.head(20).to_string())


if __name__ == "__main__":
    main()
