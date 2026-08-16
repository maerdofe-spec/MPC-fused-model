# SMC / MPC 200 cm 对比数据归档

本目录保存 2026-08-16 组实验的对比图、分析脚本和实际使用的原始输入数据。

## 图像

- `original_smc_mpc_comparison_20260816.png`：上一轮原始综合图。
- `original_smc_mpc_200cm_low_speed_20260816.png`：上一轮原始低速图。
- `original_smc_mpc_200cm_high_speed_20260816.png`：上一轮原始高速图。
- `corrected_smc_mpc_200cm_20s_20260816.png`：修正版 200 cm / 20 s 图。
- `corrected_smc_mpc_200cm_10s_20260816.png`：修正版 200 cm / 10 s 图。
- `corrected_smc_mpc_200cm_5s_20260816.png`：修正版 200 cm / 5 s 图。

修正版 Forward 误差使用误差幅值 `|Forward error|`，避免往返运动的正负号抵消；粗线为各段均值，阴影为 25–75% 区间。停车段由池顶相机运动速度门限剔除，Tag17 速度用于速度分组。

## 原始输入

- `inputs/mpc/`：MPC 控制 JSONL 和池顶相机 rosbag SQLite 数据。
- `inputs/smc/`：SMC 控制 JSONL 和同步池顶相机 CSV 数据。

20 s / 10 s 修正版使用 `175230`、`180128`、`180343`、`180538`、`180744`、`181255` 等 MPC 记录，以及 `175419`、`175715`、`181743` 等 SMC 记录；5 s 修正版使用 MPC `172732` 和 SMC `174133` 对应记录。

## 分析文件

- `analysis/plot_dynamic_200cm_comparison_batch.py`：批量分段、分组和绘图脚本。
- `analysis/smc_mpc_200cm_comparison_summary_20260816.json`：分段、来源和指标汇总。
