from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


BASE_URL = "https://data.binance.vision/data/futures/um/monthly"

DATASETS = {
    "klines": {
        "remote": "klines",
        "local_candidates": ["Kline", "klines"],
        "filename": "{symbol}-1m-{year}-{month:02d}.zip",
        "remote_path": "{remote}/{symbol}/1m/{filename}",
    },
    "markPriceKlines": {
        "remote": "markPriceKlines",
        "local_candidates": ["markPriceKlines"],
        "filename": "{symbol}-1m-{year}-{month:02d}.zip",
        "remote_path": "{remote}/{symbol}/1m/{filename}",
    },
    "premiumIndexKlines": {
        "remote": "premiumIndexKlines",
        "local_candidates": ["premiumIndexKlines"],
        "filename": "{symbol}-1m-{year}-{month:02d}.zip",
        "remote_path": "{remote}/{symbol}/1m/{filename}",
    },
    "indexPriceKlines": {
        "remote": "indexPriceKlines",
        "local_candidates": ["indexPriceKlines"],
        "filename": "{symbol}-1m-{year}-{month:02d}.zip",
        "remote_path": "{remote}/{symbol}/1m/{filename}",
    },
    "fundingRate": {
        "remote": "fundingRate",
        "local_candidates": ["fundingrate", "fundingRate"],
        "filename": "{symbol}-fundingRate-{year}-{month:02d}.zip",
        "remote_path": "{remote}/{symbol}/{filename}",
    },
}


def choose_local_dir(raw_dir: Path, candidates: list[str]) -> Path:
    for candidate in candidates:
        path = raw_dir / candidate
        if path.exists():
            return path
    return raw_dir / candidates[0]


def has_existing_file(local_dir: Path, symbol: str, year: int, month: int, dataset: str) -> bool:
    if dataset == "fundingRate":
        stem = f"{symbol}-fundingRate-{year}-{month:02d}"
    else:
        stem = f"{symbol}-1m-{year}-{month:02d}"
    return any(path.suffix != ".part" and not path.name.endswith(".part") for path in local_dir.rglob(f"{stem}.*"))


def download_file(url: str, out_path: Path, retries: int, sleep_seconds: float) -> str:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")

    for attempt in range(1, retries + 1):
        try:
            with urlopen(url, timeout=60) as response:
                if response.status != 200:
                    raise RuntimeError(f"HTTP {response.status}")
                with tmp_path.open("wb") as fh:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        fh.write(chunk)
            for replace_attempt in range(1, 11):
                try:
                    os.replace(tmp_path, out_path)
                    break
                except (FileNotFoundError, PermissionError):
                    if replace_attempt == 10:
                        raise
                    time.sleep(0.5)
            return "downloaded"
        except HTTPError as exc:
            if exc.code == 404:
                if tmp_path.exists():
                    tmp_path.unlink()
                return "missing"
            last_error = exc
        except (OSError, URLError, TimeoutError, RuntimeError) as exc:
            last_error = exc

        if attempt < retries:
            time.sleep(sleep_seconds)

    if tmp_path.exists():
        tmp_path.unlink()
    raise RuntimeError(f"Failed to download {url}: {last_error}")


def iter_jobs(symbols: list[str], datasets: list[str], start_year: int, end_year: int):
    for symbol in symbols:
        symbol = symbol.upper()
        for dataset in datasets:
            spec = DATASETS[dataset]
            for year in range(start_year, end_year + 1):
                for month in range(1, 13):
                    filename = spec["filename"].format(symbol=symbol, year=year, month=month)
                    remote_path = spec["remote_path"].format(
                        remote=spec["remote"],
                        symbol=symbol,
                        year=year,
                        month=month,
                        filename=filename,
                    )
                    yield symbol, dataset, filename, f"{BASE_URL}/{remote_path}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Binance USDT-M monthly data from data.binance.vision.")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--start-year", type=int, default=2020)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    invalid = sorted(set(args.datasets) - set(DATASETS))
    if invalid:
        raise ValueError(f"Unknown datasets: {invalid}. Valid datasets: {sorted(DATASETS)}")

    counts = {"downloaded": 0, "skipped": 0, "missing": 0}
    for symbol, dataset, filename, url in iter_jobs(args.symbols, args.datasets, args.start_year, args.end_year):
        local_dir = choose_local_dir(args.raw_dir, DATASETS[dataset]["local_candidates"])
        out_path = local_dir / filename

        if not args.force and has_existing_file(local_dir, symbol, int(filename.split("-")[-2]), int(filename.split("-")[-1].split(".")[0]), dataset):
            counts["skipped"] += 1
            print(f"skip {dataset} {symbol} {filename}")
            continue

        status = download_file(url, out_path, args.retries, args.sleep)
        counts[status] += 1
        print(f"{status} {dataset} {symbol} {filename}")

    print(f"Done: {counts}")


if __name__ == "__main__":
    main()
