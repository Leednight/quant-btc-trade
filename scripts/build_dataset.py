from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trade_count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
]


def read_csv_or_zip(path: Path, names: list[str] | None = None) -> pd.DataFrame:
    return pd.read_csv(path, header=None, names=names, compression="infer")


def list_data_files(root: Path, symbol: str | None = None) -> list[Path]:
    files = []
    for pattern in ("*.zip", "*.csv"):
        files.extend(root.rglob(pattern))
    if symbol:
        symbol_upper = symbol.upper()
        files = [path for path in files if path.name.upper().startswith(symbol_upper)]
    return sorted(files)


def find_data_dir(raw_dir: Path, candidates: list[str]) -> Path:
    for candidate in candidates:
        path = raw_dir / candidate
        if path.exists():
            return path

    lower_to_path = {path.name.lower(): path for path in raw_dir.iterdir() if path.is_dir()}
    for candidate in candidates:
        path = lower_to_path.get(candidate.lower())
        if path is not None:
            return path

    return raw_dir / candidates[0]


def read_kline_dir(root: Path, symbol: str, prefix: str = "") -> pd.DataFrame:
    files = list_data_files(root, symbol)
    if not files:
        raise FileNotFoundError(f"No csv/zip files found in {root}")

    frames = []
    for path in files:
        df = read_csv_or_zip(path, KLINE_COLUMNS)
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)

    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "trade_count",
        "taker_buy_volume",
        "taker_buy_quote_volume",
    ]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["timestamp", "open", "high", "low", "close"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    df = df.set_index("timestamp")

    keep = numeric_cols
    if prefix:
        rename = {col: f"{prefix}_{col}" for col in keep}
        df = df[keep].rename(columns=rename)
    else:
        df = df[keep]
    return df


def read_funding_dir(root: Path, symbol: str) -> pd.DataFrame:
    files = list_data_files(root, symbol)
    if not files:
        raise FileNotFoundError(f"No csv/zip files found in {root}")

    frames = []
    for path in files:
        raw = pd.read_csv(path, compression="infer")
        cols = {c.lower(): c for c in raw.columns}

        if "calc_time" in cols:
            time_col = cols["calc_time"]
        elif "fundingtime" in cols:
            time_col = cols["fundingtime"]
        else:
            time_col = raw.columns[0]

        rate_col = None
        for candidate in ("last_funding_rate", "fundingrate", "funding_rate"):
            if candidate in cols:
                rate_col = cols[candidate]
                break
        if rate_col is None:
            rate_col = raw.columns[-1]

        df = raw[[time_col, rate_col]].copy()
        df.columns = ["timestamp", "funding_rate"]
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = pd.to_datetime(pd.to_numeric(df["timestamp"], errors="coerce"), unit="ms", utc=True)
    df["funding_rate"] = pd.to_numeric(df["funding_rate"], errors="coerce")
    df = df.dropna(subset=["timestamp", "funding_rate"])
    return df.sort_values("timestamp").drop_duplicates("timestamp").set_index("timestamp")


def resample_ohlcv(df: pd.DataFrame, rule: str = "1h") -> pd.DataFrame:
    aggregations = {}
    for col in df.columns:
        if col.endswith("_open") or col == "open":
            aggregations[col] = "first"
        elif col.endswith("_high") or col == "high":
            aggregations[col] = "max"
        elif col.endswith("_low") or col == "low":
            aggregations[col] = "min"
        elif col.endswith("_close") or col == "close":
            aggregations[col] = "last"
        elif any(token in col for token in ["volume", "quote_volume", "trade_count"]):
            aggregations[col] = "sum"
        else:
            aggregations[col] = "last"

    return df.resample(rule, label="left", closed="left").agg(aggregations).dropna(how="all")


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for n in [1, 3, 6, 12, 24, 48, 72]:
        df[f"ret_{n}"] = df["close"].pct_change(n)

    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema60"] = df["close"].ewm(span=60, adjust=False).mean()
    df["close_ema20_gap"] = df["close"] / df["ema20"] - 1
    df["close_ema60_gap"] = df["close"] / df["ema60"] - 1
    df["ema20_ema60_gap"] = df["ema20"] / df["ema60"] - 1

    df["rolling_high_24"] = df["high"].rolling(24).max()
    df["rolling_low_24"] = df["low"].rolling(24).min()
    df["close_high24_gap"] = df["close"] / df["rolling_high_24"] - 1
    df["close_low24_gap"] = df["close"] / df["rolling_low_24"] - 1

    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - df["close"].shift(1)).abs()
    tr3 = (df["low"] - df["close"].shift(1)).abs()
    df["tr"] = np.maximum.reduce([tr1, tr2, tr3])
    df["atr14"] = df["tr"].rolling(14).mean()
    df["atr14_pct"] = df["atr14"] / df["close"]
    df["volatility_24"] = df["ret_1"].rolling(24).std()
    df["range_pct"] = df["high"] / df["low"] - 1

    df["volume_ma20"] = df["volume"].rolling(20).mean()
    df["quote_volume_ma20"] = df["quote_volume"].rolling(20).mean()
    df["trade_count_ma20"] = df["trade_count"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_ma20"] - 1
    df["quote_volume_ratio"] = df["quote_volume"] / df["quote_volume_ma20"] - 1
    df["trade_count_ratio"] = df["trade_count"] / df["trade_count_ma20"] - 1
    df["taker_buy_ratio"] = df["taker_buy_volume"] / df["volume"]
    df["taker_buy_quote_ratio"] = df["taker_buy_quote_volume"] / df["quote_volume"]

    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd_dif"] = ema12 - ema26
    df["macd_dea"] = df["macd_dif"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd_dif"] - df["macd_dea"]
    df["macd_hist_change"] = df["macd_hist"].diff()
    df["macd_dif_positive"] = (df["macd_dif"] > 0).astype(int)

    return df


def add_auxiliary_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "funding_rate" in df.columns:
        df["funding_rate_ma24"] = df["funding_rate"].rolling(24, min_periods=1).mean()
        std = df["funding_rate"].rolling(72, min_periods=12).std()
        df["funding_rate_zscore"] = (df["funding_rate"] - df["funding_rate"].rolling(72, min_periods=12).mean()) / std

    if "mark_close" in df.columns:
        df["close_mark_gap"] = df["close"] / df["mark_close"] - 1
        df["mark_ret_1"] = df["mark_close"].pct_change(1)

    if "index_close" in df.columns:
        df["close_index_gap"] = df["close"] / df["index_close"] - 1
        df["index_ret_1"] = df["index_close"].pct_change(1)

    if {"mark_close", "index_close"}.issubset(df.columns):
        df["mark_index_gap"] = df["mark_close"] / df["index_close"] - 1

    if "premium_close" in df.columns:
        df["premium_ma24"] = df["premium_close"].rolling(24, min_periods=1).mean()
        premium_std = df["premium_close"].rolling(72, min_periods=12).std()
        premium_mean = df["premium_close"].rolling(72, min_periods=12).mean()
        df["premium_zscore"] = (df["premium_close"] - premium_mean) / premium_std

    return df


def add_higher_timeframe_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    close_4h = df["close"].resample("4h", label="left", closed="left").last()
    ema20_4h = close_4h.ewm(span=20, adjust=False).mean()
    ema60_4h = close_4h.ewm(span=60, adjust=False).mean()
    features_4h = pd.DataFrame(
        {
            "trend_4h_ret_1": close_4h.pct_change(1),
            "trend_4h_ema_gap": ema20_4h / ema60_4h - 1,
            "trend_4h_close_ema20_gap": close_4h / ema20_4h - 1,
        }
    )
    # At 00:00/04:00/etc. the just-labeled 4H bar has not been available during
    # that bar. Shift one completed 4H bar forward before filling to 1H rows.
    features_4h = features_4h.shift(1).reindex(df.index, method="ffill")

    close_1d = df["close"].resample("1d", label="left", closed="left").last()
    ema20_1d = close_1d.ewm(span=20, adjust=False).mean()
    ema60_1d = close_1d.ewm(span=60, adjust=False).mean()
    features_1d = pd.DataFrame(
        {
            "trend_1d_ret_1": close_1d.pct_change(1),
            "trend_1d_ema_gap": ema20_1d / ema60_1d - 1,
            "trend_1d_is_bull": (ema20_1d > ema60_1d).astype(int),
        }
    )
    # Daily features are only known after the UTC day is complete, so every 1H
    # row uses the previous completed daily bar.
    features_1d = features_1d.shift(1).reindex(df.index, method="ffill")

    return pd.concat([df, features_4h, features_1d], axis=1)


def add_label(df: pd.DataFrame, horizon: int, threshold: float) -> pd.DataFrame:
    df = df.copy()
    df["future_return"] = df["close"].shift(-horizon) / df["close"] - 1
    df["label"] = (df["future_return"] > threshold).astype(int)
    return df


def build_dataset(raw_dir: Path, symbol: str, horizon: int, threshold: float) -> pd.DataFrame:
    symbol = symbol.upper()
    klines = read_kline_dir(find_data_dir(raw_dir, ["klines", "Kline", "kline"]), symbol)
    base = resample_ohlcv(klines, "1h")

    optional_kline_dirs = {
        "markPriceKlines": "mark",
        "premiumIndexKlines": "premium",
        "indexPriceKlines": "index",
    }
    for dirname, prefix in optional_kline_dirs.items():
        path = find_data_dir(raw_dir, [dirname])
        if path.exists() and list_data_files(path, symbol):
            aux = read_kline_dir(path, symbol, prefix=prefix)
            base = base.join(resample_ohlcv(aux, "1h"), how="left")

    funding_path = find_data_dir(raw_dir, ["fundingRate", "fundingrate"])
    if funding_path.exists() and list_data_files(funding_path, symbol):
        funding = read_funding_dir(funding_path, symbol)
        base = base.join(funding.resample("1h").last(), how="left")
        base["funding_rate"] = base["funding_rate"].ffill()

    base = add_technical_features(base)
    base = add_auxiliary_features(base)
    base = add_higher_timeframe_features(base)
    base = add_label(base, horizon=horizon, threshold=threshold)
    base.insert(0, "symbol", symbol)
    return base


def save_dataset(df: pd.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".parquet":
        df.to_parquet(out)
    else:
        df.to_csv(out, index_label="timestamp")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build 1H BTCUSDT feature dataset from Binance Data Vision files.")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--raw-dir", default="data/raw", type=Path)
    parser.add_argument("--out", default="data/processed/btcusdt_1h_features.parquet", type=Path)
    parser.add_argument("--horizon", default=12, type=int, help="Prediction horizon in 1H bars.")
    parser.add_argument("--threshold", default=0.004, type=float, help="Positive label threshold, e.g. 0.004 means 0.4 percent.")
    args = parser.parse_args()

    df = build_dataset(args.raw_dir, args.symbol, args.horizon, args.threshold)
    save_dataset(df, args.out)
    print(f"Saved {len(df):,} rows to {args.out}")
    print(f"Time range: {df.index.min()} -> {df.index.max()}")
    print(f"Positive label ratio: {df['label'].mean():.4f}")


if __name__ == "__main__":
    main()
