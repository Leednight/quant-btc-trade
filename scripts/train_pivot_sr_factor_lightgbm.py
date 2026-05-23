from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    from explore_pivot_sr_strategy import build_levels
    from optimize_rule_strategy import load_dataset
    from train_lightgbm import select_feature_columns
except ModuleNotFoundError:
    from scripts.explore_pivot_sr_strategy import build_levels
    from scripts.optimize_rule_strategy import load_dataset
    from scripts.train_lightgbm import select_feature_columns


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


def split_walk_forward(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df.loc["2020-01-01":"2022-12-31 23:59:59"]
    valid = df.loc["2023-01-01":"2023-12-31 23:59:59"]
    test = df.loc["2024-01-01":"2024-12-31 23:59:59"]
    return train, valid, test


def add_label(df: pd.DataFrame, horizon: int, threshold: float) -> pd.DataFrame:
    df = df.copy()
    df["pivot_future_return"] = df["close"].shift(-horizon) / df["close"] - 1
    df["pivot_label"] = (df["pivot_future_return"] > threshold).astype(int)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LightGBM with pivot support/resistance factors.")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--ema-len", type=int, default=None)
    parser.add_argument("--left", type=int, default=None)
    parser.add_argument("--right", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--threshold", type=float, default=0.004)
    parser.add_argument("--model-out", type=Path, default=None)
    args = parser.parse_args()

    import joblib
    import lightgbm as lgb
    from sklearn.metrics import classification_report, roc_auc_score

    symbol = args.symbol.upper()
    if args.ema_len is None:
        args.ema_len = 14 if symbol == "BTCUSDT" else 5
    if args.left is None:
        args.left = 15 if symbol == "BTCUSDT" else 8

    df = load_dataset(args.data)
    levels = build_levels(df, args.ema_len, args.left, args.right)
    data = df.join(levels[PIVOT_FACTOR_COLUMNS], how="left")
    data = add_label(data, args.horizon, args.threshold)

    base_features = select_feature_columns(data)
    feature_cols = [col for col in base_features if col not in {"pivot_label", "pivot_future_return"}]
    feature_cols = [col for col in feature_cols if col in data.columns]
    data = data.replace([float("inf"), float("-inf")], pd.NA).dropna(subset=feature_cols + ["pivot_label", "pivot_future_return"])

    train, valid, test = split_walk_forward(data)
    x_train, y_train = train[feature_cols], train["pivot_label"]
    x_valid, y_valid = valid[feature_cols], valid["pivot_label"]
    x_test, y_test = test[feature_cols], test["pivot_label"]

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=100,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_valid, y_valid)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)],
    )

    valid_pred = pd.Series(model.predict_proba(x_valid)[:, 1], index=valid.index)
    test_pred = pd.Series(model.predict_proba(x_test)[:, 1], index=test.index)
    test_class = (test_pred > 0.5).astype(int)

    print("\nDataset")
    print(f"Symbol: {symbol}")
    print(f"Pivot params: EMA{args.ema_len}, left={args.left}, right={args.right}")
    print(f"Train rows: {len(train):,}, positive ratio: {y_train.mean():.4f}")
    print(f"Valid rows: {len(valid):,}, positive ratio: {y_valid.mean():.4f}")
    print(f"Test rows:  {len(test):,}, positive ratio: {y_test.mean():.4f}")
    print(f"Features: {len(feature_cols)}")

    print("\nMetrics")
    print(f"Valid AUC: {roc_auc_score(y_valid, valid_pred):.4f}")
    print(f"Test AUC:  {roc_auc_score(y_test, test_pred):.4f}")
    print(classification_report(y_test, test_class, digits=4, zero_division=0))

    importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\nPivot factor importance")
    print(importance.reindex(PIVOT_FACTOR_COLUMNS).dropna().sort_values(ascending=False).to_string())
    print("\nTop 20 features")
    print(importance.head(20).to_string())

    model_out = args.model_out or Path(f"models/{symbol.lower()}_pivot_sr_factor_lgbm.joblib")
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "features": feature_cols,
            "pivot_factor_columns": PIVOT_FACTOR_COLUMNS,
            "ema_len": args.ema_len,
            "left": args.left,
            "right": args.right,
            "symbol": symbol,
        },
        model_out,
    )
    print(f"\nSaved model to {model_out}")


if __name__ == "__main__":
    main()
