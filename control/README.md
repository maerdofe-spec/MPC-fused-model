# 控制实现

本目录是 Python 控制代码的共同根目录。包名保持不变，未修改控制算法或实机配置参数。

- `MPC_dual_model/`：双模型融合、状态估计、实机通信及标定工具。
- `MPC_dual_model_yaw/`：带偏航控制的双模型实现。
- `MPC_model1/`、`MPC_model2/`：两套单模型实现。
- `PID_controller/`：PID 控制实现。
- `滑模控制/`：SMC 控制实现。
- `finesub_smc_control.py`：SMC 实机启动入口。
- `plot_position_error.py`、`plot_smc_position_error.py`：实机实时误差监视。
- `rov_track_control3.py`：旧手动/CSRT 诊断入口，不允许进入 AUTO。
- `zero_test_model.py`：模型检查工具。

正式和实验 AUTO 的两个启动入口仍保留在仓库根目录，脚本名称及直接调用方式不变。
使用 uv 时，项目路径改为 `control/MPC_dual_model`；完整命令见根目录 `README.md`。
其他脚本从仓库根目录运行时，需加 `control/` 前缀，例如：

```powershell
python control/finesub_smc_control.py --help
python control/plot_position_error.py --help
```

使用 `python -m MPC_dual_model...` 或直接导入控制包时，请先进入 `control` 目录，
或将本目录加入 `PYTHONPATH`。各控制器的测试、依赖配置及实现说明保留在对应包内。

`calibration_logs` 是指向 `../raw_data/calibration_logs` 的兼容连接，不是另一份数据。
该连接仅在本机保留，不上传到 GitHub；控制配置和默认日志路径直接指向 `raw_data`，
克隆后不依赖兼容连接。
下位机固件仍位于仓库根目录的 `V4pro1_MPC`，Git 子模块位置未改动。
