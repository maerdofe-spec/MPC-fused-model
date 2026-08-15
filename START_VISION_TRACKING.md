# FineSUB 一键启动视觉追踪

更新时间：2026-08-14

下面是一条可整段复制到一个 Bash 终端执行的命令。它会依次：

1. 启动双目红鱼视觉程序并打开实时相机窗口；
2. 固定使用 `--process-every 3`，把视觉结果写入本次独立目录；
3. 打开 FineSUB MPC 实时位置误差图；
4. 以前台方式启动当前 `dual-yaw` 实机实验 AUTO，并把它绑定到本次视觉 JSONL 和误差图 trace；
5. 不设置实验时长，直到操作者按 `Ctrl+C` 或控制器因故障退出；退出时自动清理本命令启动的视觉和误差图进程。

该命令只运行现有视觉程序，不修改 `tracking_depth` 的代码或参数文件。

```bash
reloadreloadreloadreload
```

## 运行结果位置

- 相机窗口：视觉程序的实时 GUI；
- 实时误差窗口：`FineSUB MPC Position Error — Live`；
- 视觉录像：`tracking_depth/depth-estimation-dev/output/finesub_mpc_dual_yaw_<时间>/monitor.mp4`；
- 视觉日志：同目录下的 `vision.log`；
- 误差图日志：同目录下的 `error_plot.log`；
- MPC trace：`MPC/calibration_logs/experimental_auto_<时间>.jsonl`。

如果相机窗口没有出现，先查看 `vision.log`；如果 30 秒内没有 JSONL，命令会自动停止，通常表示 UDP 5600 视频流没有到达。
