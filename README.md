# FineSUB MPC 控制

本仓库包含单模型、双模型融合以及带偏航控制的 MPC。正式实机运行采用独立的
无手柄 AUTO-only 入口 `finesub_auto_control.py`，模型固定为 `dual-yaw`；旧
`rov_track_control3.py` 只保留手动/CSRT 查看用途，不能进入 AUTO。

## 克隆完整工程

`V4pro1_MPC` 固件以 Git 子模块接入本仓库，固件内部的 CMSIS-DSP 和 ETL
也保持为子模块。首次克隆时请递归拉取：

```powershell
git clone --recurse-submodules https://github.com/maerdofe-spec/MPC-fused-model.git
```

已有工作目录可执行：

```powershell
git submodule update --init --recursive
```

固件也可在独立仓库查看：
[V4pro1_MPC](https://github.com/maerdofe-spec/V4pro1_MPC)。

## 实机控制链路

控制坐标统一为机体系 FRD：x 向前、y 向右、z 向下，正偏航力矩使艇首
向右。MPC 输出物理量 `[Fx, Fy, Fz, N]`，
`MPC_dual_model/finesub_protocol.py` 按标定值转换为固件混控器的归一化
`[forward, right, down, yaw]` 通道。

v5 控制帧包含版本、显式解锁位、yaw 控制来源、随机会话号、16 位序号、发送
时间和 CRC-16/MODBUS，新会话必须先完成停桨握手。下位机以 20 Hz 回传完整
IMU、深度/压力、命令接受或拒绝原因、混控后真正下发的 8 路电机油门以及
8 路 DSHOT RPM。活动 AUTO 目前用最终实际下发的 8 路电机油门反算上一拍
平移力/偏航力矩；RPM 与同艇推力曲线并行留作诊断，尚不冒充真实测力。
该执行量估计供状态估计和模型预测使用。只有遥测、执行反馈和命令确认均新鲜时才能
进入自动 MPC；固件连续 250 ms 没有接受合法 MPC 帧时会清零并停桨。

完整帧格式、拒绝原因和安全状态机见协议文档与
`MPC_dual_model/finesub_protocol.py`。

以下 `--model` 表只适用于仿真、库示例和旧诊断程序，不适用于正式 AUTO：

| `--model` | 平移控制 | yaw 控制 |
|---|---|---|
| `model1` | 模型一 MPC | 下位机本地航向保持 |
| `model2` | 模型二 MPC | 下位机本地航向保持 |
| `dual` | 双模型融合 MPC | 下位机本地航向保持 |
| `dual-yaw` | 双模型融合 MPC | 上位机 yaw 状态机与双环控制 |

## 正式 AUTO 运行

在本目录由 uv 管理依赖。默认命令只预检，不连接硬件：

```powershell
uv sync --project MPC_dual_model --group dev
uv run --project MPC_dual_model python finesub_auto_control.py
```

只有预检输出 `AUTO READY` 后才运行：

```powershell
uv run --project MPC_dual_model python finesub_auto_control.py --execute
```

当前实机配置仍会输出 `AUTO BLOCKED`，原因是相机刚性外参与完整三轴实机动力学没有通过
接受门。阻断发生在创建 UDP/串口 transport 之前。正式入口没有 joystick、MANUAL、ROI/CSRT
或 `--model` 参数，也不会用仿真默认值补齐实机参数。

### 风险接受的首次实机追踪实验

操作者在 2026-08-13 明确接受当前外参与 sway/heave 候选误差，并要求停止重复测量、尽快进行
MPC+红鱼视觉实物追踪。为避免把实验结论伪装成正式批准，独立预检入口为：

```powershell
uv run --project MPC_dual_model python finesub_experimental_auto.py
```

预检输出 `EXPERIMENTAL AUTO READY` 后，实验执行命令为：

```powershell
uv run --project MPC_dual_model python finesub_experimental_auto.py --execute
```

若视觉系统每次在新的时间戳目录写结果，可由 MPC 只读指定该文件：

```powershell
uv run --project MPC_dual_model python finesub_experimental_auto.py --execute --vision-jsonl /absolute/path/to/pipeline_results.jsonl
```

实验固定使用 `dual-yaw`、目标相机光轴距离 `0.60 m`（当前候选外参对应 body FRD
`[0.857634,-0.055545,-0.120815] m`），当前 forward/right/down 最大通道均为
`+/-0.20`，默认不计时，由操作员 `Ctrl+C` 或安全故障结束。heave 模型使用实测净上浮外力
`-0.807290 N`，对应 Fossen 方程中的下向悬浮平衡力 `+0.807290 N`。实验 AUTO 不使用检测器置信度下限；通过深度/创新/范围/运动门的视觉坐标更新 MPC。视觉间隙时平移受变化率限制退回 `tau_h`，直接 yaw 力矩退回武装基线，下位机继续稳定 roll/pitch。每次执行自动保存逐帧 JSONL，包含相机/机体位置、估计状态、平移力、直接 yaw 力矩、通道、RPM、模型权重和停止原因。遥测、命令确认、通信或 QP 故障仍会解除武装并停止。该入口不会修改正式 `auto_runtime.enabled=false`。

通信方式、地址、安全时间和硬件换算参数位于：

```text
MPC_dual_model/finesub_v4pro1_mpc.json
```

也可以指定其他配置：

```powershell
uv run --project MPC_dual_model python finesub_auto_control.py --config path/to/finesub.json
```

配置中的 `transport.type` 支持 `tcp`、`udp`、`serial` 和 `dry_run`。

### 实际执行反馈的定义

MPC 使用 8 路真实 RPM 与逐电机正反向二次曲线重构已实现控制量。
当前同艇历史测力曲线与本艇 `+/-5/7.5/10%` RPM 支持低限幅力换算，
但垂直推进器曲线仍是按同推进单元旋向组迁移，这不等于完整三轴动力学已经获批。

水平四桨或垂直四桨的 RPM 有效位不完整时，闭环只对该组回退到下位机经过混控、
方向修正和限幅后的最终电机设定值反算；trace 会记录每轴来源。不得直接套用其他艇
或其他推进器的占位曲线。

### 烧录新固件

“烧录新固件”指把本次修改后的 FineSUB 下位机 C++ 源码编译成
`Robomaster_A.hex`，再通过下载器写入实机的 STM32 控制板。它就是更新下位机
程序，不是只把源码复制到 `V4pro1_MPC` 文件夹。

协议 v3 的上下行帧格式都已改变，因此新上位机代码和新下位机固件必须配套
使用。只更新电脑上的 Python 而不重新烧录 STM32，实际执行回显不会生效。

视频输入还要求操作系统已安装 GStreamer、H.264 解码插件和 PyGObject (`gi`)；
这些组件应使用目标机的系统包管理器安装。

旧 `rov_track_control3.py` 仍可用于手动/CSRT 诊断：`S` 框选、`Z` 停止、`Q` 退出，
按钮 7 手动解锁/停桨；按钮 3 只会打印 `AUTO BLOCKED`，不会切换模式。

程序退出时会发送显式停桨帧。没有视频、遥测过期、命令确认过期、网络中断或
上位机停止发帧时，下位机看门狗仍会停桨；通信恢复后不会自动重新解锁。

## 验证

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD="1"
uv run --project MPC_dual_model pytest -q MPC_dual_model/tests
```

## 下水前必须标定

示例中的质量、附加质量、阻尼、最大三轴力、最大偏航力矩、推力符号、相机
视场角以及“目标框宽度到距离”的比例仍是占位/初始值。首次联调应卸桨验证
CRC、通道顺序、符号和 250 ms 失联停机，再系留低限幅测试；未经这些步骤不
应直接自动下水。
