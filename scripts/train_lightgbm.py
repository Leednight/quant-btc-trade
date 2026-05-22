from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


EXCLUDE_COLUMNS = {
    "symbol",
    "label",
    "future_return",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "trade_count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ema20",
    "ema60",
    "rolling_high_24",
    "rolling_low_24",
    "tr",
    "atr14",
    "volume_ma20",
    "quote_volume_ma20",
    "trade_count_ma20",
}

EXCLUDE_PREFIXES = (
    "mark_open",
    "mark_high",
    "mark_low",
    "mark_close",
    "mark_volume",
    "mark_quote_volume",
    "mark_trade_count",
    "mark_taker_buy_volume",
    "mark_taker_buy_quote_volume",
    "index_open",
    "index_high",
    "index_low",
    "index_close",
    "index_volume",
    "index_quote_volume",
    "index_trade_count",
    "index_taker_buy_volume",
    "index_taker_buy_quote_volume",
    "premium_open",
    "premium_high",
    "premium_low",
    "premium_volume",
    "premium_quote_volume",
    "premium_trade_count",
    "premium_taker_buy_volume",
    "premium_taker_buy_quote_volume",
)


def load_dataset(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp")

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df.sort_index()


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for col in df.columns:
        if col in EXCLUDE_COLUMNS:
            continue
        if col.startswith(EXCLUDE_PREFIXES):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def split_2020(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df.loc["2020-01-01":"2020-09-30 23:59:59"]
    valid = df.loc["2020-10-01":"2020-11-30 23:59:59"]
    test = df.loc["2020-12-01":"2020-12-31 23:59:59"]
    return train, valid, test


def print_signal_report(name: str, data: pd.DataFrame, pred: pd.Series, threshold: float) -> None:
    selected = data.loc[pred > threshold]
    print(f"\n{name} signal report, pred > {threshold:.2f}")
    print(f"Signals: {len(selected):,} / {len(data):,}")
    if len(selected) == 0:
        return
    print(f"Average future return: {selected['future_return'].mean():.4%}")
    print(f"Median future return: {selected['future_return'].median():.4%}")
    print(f"Win rate over label threshold: {selected['label'].mean():.4%}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LightGBM trend model.")
    parser.add_argument("--data", default="data/processed/btcusdt_1h_features.parquet", type=Path)
    parser.add_argument("--model-out", default="models/btcusdt_lgbm_1h.joblib", type=Path)
    parser.add_argument("--features-out", default="models/btcusdt_lgbm_1h_features.txt", type=Path)
    parser.add_argument("--signal-threshold", default=0.58, type=float)
    args = parser.parse_args()

    import joblib
    import lightgbm as lgb
    from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

    df = load_dataset(args.data)
    feature_cols = select_feature_columns(df)
    data = df.dropna(subset=feature_cols + ["label", "future_return"]).copy()

    train, valid, test = split_2020(data)
    if train.empty or valid.empty or test.empty:
        raise ValueError(
            "Train/valid/test split is empty. Make sure the dataset covers 2020-01 through 2020-12."
        )

    x_train, y_train = train[feature_cols], train["label"]
    x_valid, y_valid = valid[feature_cols], valid["label"]
    x_test, y_test = test[feature_cols], test["label"]

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
    print(f"Train rows: {len(train):,}, positive ratio: {y_train.mean():.4f}")
    print(f"Valid rows: {len(valid):,}, positive ratio: {y_valid.mean():.4f}")
    print(f"Test rows:  {len(test):,}, positive ratio: {y_test.mean():.4f}")
    print(f"Features: {len(feature_cols)}")

    print("\nMetrics")
    print(f"Valid AUC: {roc_auc_score(y_valid, valid_pred):.4f}")
    print(f"Test AUC:  {roc_auc_score(y_test, test_pred):.4f}")
    print(f"Test accuracy at 0.50: {accuracy_score(y_test, test_class):.4f}")
    print(classification_report(y_test, test_class, digits=4))

    print_signal_report("Valid", valid, valid_pred, args.signal_threshold)
    print_signal_report("Test", test, test_pred, args.signal_threshold)

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": feature_cols}, args.model_out)
    args.features_out.write_text("\n".join(feature_cols), encoding="utf-8")
    print(f"\nSaved model to {args.model_out}")
    print(f"Saved feature list to {args.features_out}")

    importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\nTop 20 features")
    print(importance.head(20).to_string())


if __name__ == "__main__":
    main()
