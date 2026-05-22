# quant-btc-trade
auto trade to get time

这个项目用于整理 Binance Data Vision 下载的 BTCUSDT 1m 数据，并训练一个 1H 趋势波段 LightGBM 模型。

## 数据目录

把 2020 年 1 月到 12 月下载的数据放到下面目录。支持 `.zip` 和 `.csv`。

```text
data/raw/klines/
data/raw/fundingRate/
data/raw/markPriceKlines/
data/raw/premiumIndexKlines/
data/raw/indexPriceKlines/
```

推荐文件名保持 Binance 原始命名，例如：

```text
BTCUSDT-1m-2020-01.zip
BTCUSDT-fundingRate-2020-01.zip
```

## 安装依赖

```powershell
python -m pip install -r requirements.txt
```

## 构建特征数据

```powershell
python scripts/build_dataset.py --symbol BTCUSDT --raw-dir data/raw --out data/processed/btcusdt_1h_features.parquet
```

如果没有安装 parquet 引擎，可以输出 csv：

```powershell
python scripts/build_dataset.py --symbol BTCUSDT --raw-dir data/raw --out data/processed/btcusdt_1h_features.csv
```

## 训练模型

```powershell
python scripts/train_lightgbm.py --data data/processed/btcusdt_1h_features.parquet
```

2020 年数据默认切分：

```text
2020-01-01 到 2020-09-30：训练集
2020-10-01 到 2020-11-30：验证集
2020-12-01 到 2020-12-31：测试集
```

模型会输出到：

```text
models/btcusdt_lgbm_1h.joblib
models/btcusdt_lgbm_1h_features.txt
```

