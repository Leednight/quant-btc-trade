# 数字货币趋势策略研究

本项目用于研究 BTCUSDT / ETHUSDT 的 1H 趋势波段策略。当前流程包括：

```text
下载 Binance 历史数据
从 1m 聚合到 1H
构建多周期特征
优化 EMA 趋势规则策略
训练 LightGBM 规则过滤模型
输出回测和模型报告
```

当前推荐路线是：

```text
先用规则策略生成候选交易
再用 LightGBM 判断候选交易质量
最后用回测结果决定是否过滤信号
```

不要让模型直接决定所有买卖。对 BTC/ETH 这类强噪声资产，规则先行、模型辅助更稳。

## 环境准备

推荐在 Windows + Conda 下使用 Python 3.10：

```powershell
conda create -n trade_lgbm python=3.10 -y
conda activate trade_lgbm
python -m pip install -r requirements.txt
```

如果 conda 报 `NoWritablePkgsDirError`，可以把包缓存目录放到用户目录：

```powershell
mkdir $env:USERPROFILE\.conda_pkgs
conda config --add pkgs_dirs $env:USERPROFILE\.conda_pkgs
```

## 数据来源

数据来自 Binance 官方 Data Vision：

[https://data.binance.vision/](https://data.binance.vision/)

本项目使用 USDT-M 永续合约月度数据：

```text
klines
fundingRate
markPriceKlines
premiumIndexKlines
indexPriceKlines
```

数据默认放在：

```text
data/raw/
```

## 下载数据

下载 BTCUSDT 和 ETHUSDT 的 2020-2024 数据：

```powershell
python scripts/download_binance_data.py --symbols BTCUSDT ETHUSDT --start-year 2020 --end-year 2024 --raw-dir data/raw
```

下载脚本会自动跳过已经存在的文件，所以中途断掉后可以直接重新运行。

## 构建 1H 特征数据

BTC：

```powershell
python scripts/build_dataset.py --symbol BTCUSDT --raw-dir data/raw --out data/processed/btcusdt_1h_features_2020_2024.csv
```

ETH：

```powershell
python scripts/build_dataset.py --symbol ETHUSDT --raw-dir data/raw --out data/processed/ethusdt_1h_features_2020_2024.csv
```

特征构建逻辑：

```text
以 klines 为主表
1m K 线聚合为 1H
合并 fundingRate、markPrice、indexPrice、premiumIndex
构建 EMA、MACD、ATR、波动率、成交量、资金费率、溢价、指数偏离等特征
4H 和 1D 特征会 shift 一个完整周期，避免未来函数
```

生成文件：

```text
data/processed/btcusdt_1h_features_2020_2024.csv
data/processed/ethusdt_1h_features_2020_2024.csv
```

## 优化规则策略

运行 EMA 趋势规则网格搜索：

```powershell
python scripts/optimize_rule_strategy.py --data data/processed/btcusdt_1h_features_2020_2024.csv --out reports/btcusdt_rule_strategy_optimization_v2.csv
python scripts/optimize_rule_strategy.py --data data/processed/ethusdt_1h_features_2020_2024.csv --out reports/ethusdt_rule_strategy_optimization_v2.csv
```

强制搜索多空双向策略：

```powershell
python scripts/optimize_rule_strategy.py --data data/processed/btcusdt_1h_features_2020_2024.csv --out reports/btcusdt_rule_strategy_long_short.csv --short-mode long_short
python scripts/optimize_rule_strategy.py --data data/processed/ethusdt_1h_features_2020_2024.csv --out reports/ethusdt_rule_strategy_long_short.csv --short-mode long_short
```

使用 1D EMA100 / EMA200 判断牛熊，并在牛市只做多、熊市只做空：

```powershell
python scripts/optimize_rule_strategy.py --data data/processed/btcusdt_1h_features_2020_2024.csv --out reports/btcusdt_rule_strategy_regime_ema100_200.csv --short-mode long_short --regime-filter 1 --regime-fast 100 --regime-slow 200
python scripts/optimize_rule_strategy.py --data data/processed/ethusdt_1h_features_2020_2024.csv --out reports/ethusdt_rule_strategy_regime_ema100_200.csv --short-mode long_short --regime-filter 1 --regime-fast 100 --regime-slow 200
```

如果想使用 EMA99 / EMA200，把 `--regime-fast 100` 改成：

```powershell
--regime-fast 99
```

当前基准规则：

```text
BTCUSDT：EMA 12 / EMA 96，只做多
ETHUSDT：EMA 48 / EMA 200，只做多
```

这些参数来自 2020-2023 训练期，2024 作为样本外测试。它们不是最终实盘参数，只是当前研究基准。

当前多空测试结论：

```text
BTC/ETH 的空头规则在 2024 样本外明显拖累收益
多空双向策略的回撤更大，Sharpe 更低
当前更稳的基准仍然是只做多趋势策略
1D EMA100/EMA200 牛熊过滤能减少错误空头，但 ETH 空头仍然拖累明显
```

## 训练 LightGBM 规则过滤模型

推荐使用这个模型方向。

过滤模型不是直接预测涨跌，而是：

```text
规则策略先产生候选持仓
默认会使用多组 EMA 规则生成交易样本池
每一笔规则交易只生成一条训练样本
模型只在开仓点判断这笔交易是否值得做
评估时默认过滤当前基准规则，并保留或删除整笔交易
```

BTC：

```powershell
python scripts/train_rule_filter_lightgbm.py --data data/processed/btcusdt_1h_features_2020_2024.csv --symbol BTCUSDT
```

ETH：

```powershell
python scripts/train_rule_filter_lightgbm.py --data data/processed/ethusdt_1h_features_2020_2024.csv --symbol ETHUSDT
```

默认训练切分：

```text
2020-2022：训练集
2023：验证集
2024：测试集
```

默认标签：

```text
交易级标签：这笔规则交易的净收益 > 0
```

净收益会扣除开仓和平仓的估算交易成本。默认 `--fee-bps 5.0`，约等于单边 5 bps 的费用和滑点估计。

默认样本模式：

```text
--sample-mode multi_rule
```

这会使用多组 EMA 参数生成训练样本，缓解单一规则交易笔数太少的问题。评估报告仍然针对当前基准规则，便于和 `rule_only` 对比。

如果只想复现单一规则过滤实验：

```powershell
python scripts/train_rule_filter_lightgbm.py --data data/processed/btcusdt_1h_features_2020_2024.csv --symbol BTCUSDT --sample-mode single_rule
```

可以提高标签质量门槛：

```powershell
python scripts/train_rule_filter_lightgbm.py --data data/processed/btcusdt_1h_features_2020_2024.csv --symbol BTCUSDT --min-trade-return 0.005
```

也可以要求交易收益超过开仓时的 ATR 阈值：

```powershell
python scripts/train_rule_filter_lightgbm.py --data data/processed/btcusdt_1h_features_2020_2024.csv --symbol BTCUSDT --atr-mult 0.3
```

旧版小时级标签仍然保留，只建议做对照实验：

```powershell
python scripts/train_rule_filter_lightgbm.py --data data/processed/btcusdt_1h_features_2020_2024.csv --symbol BTCUSDT --label-mode horizon --horizon 24 --atr-mult 0.6
```

输出文件：

```text
models/btcusdt_rule_filter_lgbm.joblib
models/ethusdt_rule_filter_lgbm.joblib
reports/btcusdt_rule_filter_report.csv
reports/ethusdt_rule_filter_report.csv
```

报告里会比较：

```text
rule_only
模型概率 > 0.40
模型概率 > 0.45
模型概率 > 0.50
模型概率 > 0.55
模型概率 > 0.60
```

一个有价值的过滤模型应该至少满足其中之一：

```text
降低最大回撤
提高 Sharpe
减少明显低质量交易
在不大幅牺牲收益的情况下降低持仓暴露
```

## 重要：避免标签泄露

训练过滤模型时，下面这些字段绝对不能进入特征：

```text
filter_label
filter_future_return
filter_atr_return
future_return
label
```

如果模型输出里出现：

```text
Top features: filter_label
Valid AUC: 1.0000
Test AUC: 1.0000
```

这通常不是模型很强，而是发生了标签泄露。

当前脚本已经排除了这些字段，但每次修改特征逻辑后，仍然要检查 `Top 20 features`。

## 直接 LightGBM 基线

旧版直接分类模型仍然保留：

```powershell
python scripts/train_lightgbm.py --data data/processed/btcusdt_1h_features_2020_2024.csv
```

这个模型直接预测未来收益是否超过阈值，只适合作为研究基线。当前更推荐：

```text
规则策略 + LightGBM 过滤器
```

## 当前注意事项

当前回测仍然是研究版本，还不是实盘级回测。

尚未完整处理：

```text
真实盘口滑点
资金费率扣除
止损触发成交价
限价单未成交
强平风险
交易所断连
发明者量化实盘执行细节
```

下一步建议：

```text
加入资金费率成本
加入 ATR 止损和真实出场价格
增加仓位管理
输出逐笔交易明细
再把策略迁移到发明者量化平台
```
## 支撑压力 Pivot 因子

可以用 1D EMA 识别局部高低点，并做交替筛选：

```text
连续高点只保留最高点
连续低点只保留最低点
最终高点和低点交替出现
高点作为压力线
低点作为支撑线
```

参数探索：

```powershell
python scripts/explore_pivot_sr_strategy.py --data data/processed/btcusdt_1h_features_2020_2024.csv --out reports/btcusdt_pivot_sr_alternating_factor_exploration_fast.csv --ema-lengths 5,7,10,14 --left-windows 5,8,10,12,15

python scripts/explore_pivot_sr_strategy.py --data data/processed/ethusdt_1h_features_2020_2024.csv --out reports/ethusdt_pivot_sr_alternating_factor_exploration_fast.csv --ema-lengths 5,7,10,14 --left-windows 5,8,10,12,15
```

当前简化回测里的较优参数：

```text
BTCUSDT: EMA14, left=15, right=1
ETHUSDT: EMA5, left=8, right=1
```

支撑压力会生成这些因子：

```text
sr_range_pct
sr_position
dist_support_pct
dist_resistance_pct
dist_support_atr
dist_resistance_atr
last_pivot_type
support_age_hours
resistance_age_hours
```

训练 Pivot 因子模型：

```powershell
python scripts/train_pivot_sr_factor_lightgbm.py --data data/processed/btcusdt_1h_features_2020_2024.csv --symbol BTCUSDT --ema-len 14 --left 15

python scripts/train_pivot_sr_factor_lightgbm.py --data data/processed/ethusdt_1h_features_2020_2024.csv --symbol ETHUSDT --ema-len 5 --left 8
```

这个模型用于检验支撑压力相对位置是否能成为有效因子。重点看：

```text
Pivot factor importance
Valid AUC
Test AUC
```
## Pivot 模型概率回测

## 当前固定方案：ETH 100U 5x 激进版

当前固定主线是 ETH 100U 小资金激进方案，不加 BTC 趋势过滤，使用 5 倍名义杠杆和账户级风控。

核心参数：

```text
交易标的：ETHUSDT
初始资金：100U
名义杠杆：5x
固定阈值：long_threshold = 0.52, short_threshold = 0.40
普通信号：0.5 * 5 = 2.5x
强信号：0.72 * 5 = 3.6x
极强信号：0.8 * 5 = 4.0x
账户级单笔最大亏损：9%
连续亏损超过 20%：暂停开新仓 96 小时
回撤超过 20%：新开仓降到 70%
回撤超过 35%：暂停开新仓 72 小时
最近 168 小时净亏损超过 3%：暂停开新仓 96 小时
```

2024-01-01 到 2026-04-30 回测结果：

```text
100U -> 2,096,329U
total_return: +2,096,229%
max_drawdown: -34.04%
worst_trade: -9.80%
worst_loss_streak: -30.09%
```

2026-01-01 到 2026-04-30 最新样本单独回测：

```text
100U -> 191.86U
total_return: +91.86%
max_drawdown: -33.26%
worst_trade: -9.60%
worst_loss_streak: -26.77%
```

固定 C 方案全样本验证命令：

```powershell
python scripts/backtest_pivot_sr_model.py --data data/processed/ethusdt_1h_features_2020_2026.csv --model models/ethusdt_pivot_sr_factor_lgbm.joblib --out reports/ethusdt_100u_5x_no_btc_account_risk_C_2024_2026.csv --mode long_short --start 2024-01-01 --end "2026-04-30 23:59:59" --yearly-start 2024-01-01 --yearly-end "2026-04-30 23:59:59" --fixed-long-threshold 0.52 --fixed-short-threshold 0.40 --initial-capital 100 --leverage 5 --fee-bps 5 --slippage-bps 2 --include-funding --min-hold-hours 4 --max-consecutive-losses 3 --cooldown-hours 24 --atr-stop-mult 2.5 --max-trade-loss 0.045 --max-position-loss 0.09 --short-rebound-stop 0.04 --drawdown-reduce-threshold 0.20 --drawdown-reduce-factor 0.7 --drawdown-pause-threshold 0.35 --drawdown-cooldown-hours 72 --loss-reduce-after 2 --loss-reduced-size 0.25 --loss-streak-loss-limit 0.20 --loss-streak-cooldown-hours 96 --tighten-after-losses 3 --tightened-long-threshold 0.55 --tightened-short-threshold 0.38 --tighten-reset-hours 168 --daily-loss-limit 0.025 --rolling-loss-window-hours 168 --rolling-loss-limit 0.03 --rolling-loss-cooldown-hours 96 --vol-pause-quantile 0.95 --vol-pause-hours 24 --position-mode probability --base-size 0.5 --mid-size 0.72 --extreme-size 0.8 --max-size 0.8 --long-extreme-threshold 0.58 --short-extreme-threshold 0.42 --long-max-threshold 0.62 --short-max-threshold 0.38
```

固定 C 方案只验证 2026 最新样本：

```powershell
python scripts/backtest_pivot_sr_model.py --data data/processed/ethusdt_1h_features_2020_2026.csv --model models/ethusdt_pivot_sr_factor_lgbm.joblib --out reports/ethusdt_100u_5x_no_btc_account_risk_C_2026_only.csv --mode long_short --start 2026-01-01 --end "2026-04-30 23:59:59" --yearly-start 2026-01-01 --yearly-end "2026-04-30 23:59:59" --fixed-long-threshold 0.52 --fixed-short-threshold 0.40 --initial-capital 100 --leverage 5 --fee-bps 5 --slippage-bps 2 --include-funding --min-hold-hours 4 --max-consecutive-losses 3 --cooldown-hours 24 --atr-stop-mult 2.5 --max-trade-loss 0.045 --max-position-loss 0.09 --short-rebound-stop 0.04 --drawdown-reduce-threshold 0.20 --drawdown-reduce-factor 0.7 --drawdown-pause-threshold 0.35 --drawdown-cooldown-hours 72 --loss-reduce-after 2 --loss-reduced-size 0.25 --loss-streak-loss-limit 0.20 --loss-streak-cooldown-hours 96 --tighten-after-losses 3 --tightened-long-threshold 0.55 --tightened-short-threshold 0.38 --tighten-reset-hours 168 --daily-loss-limit 0.025 --rolling-loss-window-hours 168 --rolling-loss-limit 0.03 --rolling-loss-cooldown-hours 96 --vol-pause-quantile 0.95 --vol-pause-hours 24 --position-mode probability --base-size 0.5 --mid-size 0.72 --extreme-size 0.8 --max-size 0.8 --long-extreme-threshold 0.58 --short-extreme-threshold 0.42 --long-max-threshold 0.62 --short-max-threshold 0.38
```

注意：这个方案是高风险小资金策略。回测里最新样本最大回撤约 33%，连续亏损段可达 25%-30%。实盘前应先用模拟盘或极小资金验证执行稳定性。

## 历史实验命令

BTC 固定使用验证集选出的阈值，不再使用 2025 最优阈值：

```text
long_threshold = 0.52
short_threshold = 0.40
```

概率仓位规则：

```text
普通信号：50% 仓位
极端信号：80% 仓位
最大仓位：80%
多头极端：pred > 0.58
空头极端：pred < 0.42
ATR 止损：2.5 * ATR
单笔最大亏损：4.5%
空头强反弹止损：4%
连续亏损 2 笔后降仓到 25%
连续亏损 3 笔后冷却 24 小时
连续亏损 3 笔后临时提高信号阈值：做多从 0.52 提到 0.55，做空从 0.40 降到 0.38
严格阈值模式最长持续 168 小时，如果期间没有继续亏损则恢复正常阈值
单日已实现亏损超过 2.5% 后，当天停止开新仓
最近 168 小时已实现净亏损超过 5% 后，暂停开新仓 48 小时
24H 波动率超过历史 95% 分位时暂停开新仓 24 小时
```

BTC 固定阈值 + 概率仓位 + 成本 + 盘中止损 + 动态阈值 + 单日亏损限制，验证 2024-2026：

```powershell
python scripts/backtest_pivot_sr_model.py --data data/processed/btcusdt_1h_features_2020_2026.csv --model models/btcusdt_pivot_sr_factor_lgbm.joblib --out reports/btcusdt_pivot_sr_model_fixed_052_040_risk_dynamic_2024_2026.csv --mode long_short --start 2024-01-01 --end "2026-04-30 23:59:59" --yearly-start 2024-01-01 --yearly-end "2026-04-30 23:59:59" --fixed-long-threshold 0.52 --fixed-short-threshold 0.40 --fee-bps 5 --slippage-bps 2 --include-funding --min-hold-hours 4 --max-consecutive-losses 3 --cooldown-hours 24 --atr-stop-mult 2.5 --max-trade-loss 0.045 --short-rebound-stop 0.04 --loss-reduce-after 2 --loss-reduced-size 0.25 --tighten-after-losses 3 --tightened-long-threshold 0.55 --tightened-short-threshold 0.38 --tighten-reset-hours 168 --daily-loss-limit 0.025 --vol-pause-quantile 0.95 --vol-pause-hours 24 --position-mode probability --base-size 0.5 --extreme-size 0.8 --max-size 0.8 --long-extreme-threshold 0.58 --short-extreme-threshold 0.42
```

ETH 固定阈值 + 概率仓位 + 成本 + 盘中止损 + 动态阈值 + 单日亏损限制 + 组合级风控，验证 2024-2026：

```powershell
python scripts/backtest_pivot_sr_model.py --data data/processed/ethusdt_1h_features_2020_2026.csv --model models/ethusdt_pivot_sr_factor_lgbm.joblib --out reports/ethusdt_pivot_sr_model_fixed_052_040_risk_portfolio_2024_2026.csv --mode long_short --start 2024-01-01 --end "2026-04-30 23:59:59" --yearly-start 2024-01-01 --yearly-end "2026-04-30 23:59:59" --fixed-long-threshold 0.52 --fixed-short-threshold 0.40 --fee-bps 5 --slippage-bps 2 --include-funding --min-hold-hours 4 --max-consecutive-losses 3 --cooldown-hours 24 --atr-stop-mult 2.5 --max-trade-loss 0.045 --short-rebound-stop 0.04 --loss-reduce-after 2 --loss-reduced-size 0.25 --tighten-after-losses 3 --tightened-long-threshold 0.55 --tightened-short-threshold 0.38 --tighten-reset-hours 168 --daily-loss-limit 0.025 --rolling-loss-window-hours 168 --rolling-loss-limit 0.035 --rolling-loss-cooldown-hours 72 --vol-pause-quantile 0.95 --vol-pause-hours 24 --position-mode probability --base-size 0.5 --extreme-size 0.8 --max-size 0.8 --long-extreme-threshold 0.58 --short-extreme-threshold 0.42
```

ETH 100U 初始资金、5 倍名义杠杆、BTC 1D 趋势确认、更强连续亏损保护：

```powershell
python scripts/backtest_pivot_sr_model.py --data data/processed/ethusdt_1h_features_2020_2026.csv --model models/ethusdt_pivot_sr_factor_lgbm.joblib --out reports/ethusdt_pivot_sr_model_100u_5x_btc_trend_rolling_2024_2026.csv --mode long_short --start 2024-01-01 --end "2026-04-30 23:59:59" --yearly-start 2024-01-01 --yearly-end "2026-04-30 23:59:59" --fixed-long-threshold 0.52 --fixed-short-threshold 0.40 --initial-capital 100 --leverage 5 --btc-trend-data data/processed/btcusdt_1h_features_2020_2026.csv --btc-trend-column trend_1d_ema_gap --btc-trend-filter confirm --fee-bps 5 --slippage-bps 2 --include-funding --min-hold-hours 4 --max-consecutive-losses 3 --cooldown-hours 24 --atr-stop-mult 2.5 --max-trade-loss 0.045 --short-rebound-stop 0.04 --loss-reduce-after 2 --loss-reduced-size 0.25 --tighten-after-losses 3 --tightened-long-threshold 0.55 --tightened-short-threshold 0.38 --tighten-reset-hours 168 --daily-loss-limit 0.025 --rolling-loss-window-hours 168 --rolling-loss-limit 0.03 --rolling-loss-cooldown-hours 96 --vol-pause-quantile 0.95 --vol-pause-hours 24 --position-mode probability --base-size 0.5 --extreme-size 0.8 --max-size 0.8 --long-extreme-threshold 0.58 --short-extreme-threshold 0.42
```

ETH 测试 2025：

```powershell
python scripts/backtest_pivot_sr_model.py --data data/processed/ethusdt_1h_features_2020_2025.csv --model models/ethusdt_pivot_sr_factor_lgbm.joblib --out reports/ethusdt_pivot_sr_model_test_2025_risk.csv --mode long_short --start 2025-01-01 --end "2025-12-31 23:59:59" --fee-bps 5 --slippage-bps 2 --include-funding --min-hold-hours 4 --max-consecutive-losses 3 --cooldown-hours 24 --atr-stop-mult 2.5 --max-trade-loss 0.045 --short-rebound-stop 0.04 --loss-reduce-after 2 --loss-reduced-size 0.25 --vol-pause-quantile 0.95 --vol-pause-hours 24 --position-mode probability --base-size 0.5 --extreme-size 0.8 --max-size 0.8 --long-extreme-threshold 0.58 --short-extreme-threshold 0.42
```

用 2023 验证集选择阈值，固定阈值测试 2024，并输出 2020-2024 分年表现：

```powershell
python scripts/backtest_pivot_sr_model.py --data data/processed/btcusdt_1h_features_2020_2024.csv --model models/btcusdt_pivot_sr_factor_lgbm.joblib --out reports/btcusdt_pivot_sr_model_validate_2023_test_2024_costs.csv --mode long_short --select-start 2023-01-01 --select-end "2023-12-31 23:59:59" --test-start 2024-01-01 --test-end "2024-12-31 23:59:59" --yearly-start 2020-01-01 --yearly-end "2024-12-31 23:59:59" --fee-bps 5 --slippage-bps 2 --include-funding --min-hold-hours 4 --max-consecutive-losses 3 --cooldown-hours 24

python scripts/backtest_pivot_sr_model.py --data data/processed/ethusdt_1h_features_2020_2024.csv --model models/ethusdt_pivot_sr_factor_lgbm.joblib --out reports/ethusdt_pivot_sr_model_validate_2023_test_2024_costs.csv --mode long_short --select-start 2023-01-01 --select-end "2023-12-31 23:59:59" --test-start 2024-01-01 --test-end "2024-12-31 23:59:59" --yearly-start 2020-01-01 --yearly-end "2024-12-31 23:59:59" --fee-bps 5 --slippage-bps 2 --include-funding --min-hold-hours 4 --max-consecutive-losses 3 --cooldown-hours 24
```

用训练好的 2020-2024 模型测试新下载的 2025 数据：

```powershell
python scripts/backtest_pivot_sr_model.py --data data/processed/btcusdt_1h_features_2020_2025.csv --model models/btcusdt_pivot_sr_factor_lgbm.joblib --out reports/btcusdt_pivot_sr_model_test_2025_costs.csv --mode long_short --start 2025-01-01 --end "2025-12-31 23:59:59" --fee-bps 5 --slippage-bps 2 --include-funding --min-hold-hours 4 --max-consecutive-losses 3 --cooldown-hours 24

python scripts/backtest_pivot_sr_model.py --data data/processed/ethusdt_1h_features_2020_2025.csv --model models/ethusdt_pivot_sr_factor_lgbm.joblib --out reports/ethusdt_pivot_sr_model_test_2025_costs.csv --mode long_short --start 2025-01-01 --end "2025-12-31 23:59:59" --fee-bps 5 --slippage-bps 2 --include-funding --min-hold-hours 4 --max-consecutive-losses 3 --cooldown-hours 24
```

用模型概率做 2024 阈值回测：

```powershell
python scripts/backtest_pivot_sr_model.py --data data/processed/btcusdt_1h_features_2020_2024.csv --model models/btcusdt_pivot_sr_factor_lgbm.joblib --out reports/btcusdt_pivot_sr_model_backtest_2024.csv --mode long_short

python scripts/backtest_pivot_sr_model.py --data data/processed/ethusdt_1h_features_2020_2024.csv --model models/ethusdt_pivot_sr_factor_lgbm.joblib --out reports/ethusdt_pivot_sr_model_backtest_2024.csv --mode long_short
```

只做多版本：

```powershell
python scripts/backtest_pivot_sr_model.py --data data/processed/btcusdt_1h_features_2020_2024.csv --model models/btcusdt_pivot_sr_factor_lgbm.joblib --out reports/btcusdt_pivot_sr_model_backtest_2024_long_only.csv --mode long_only
```

报告会输出：

```text
不同 long_threshold / short_threshold 的收益
最大回撤
Sharpe
交易次数
多头收益
空头收益
```
