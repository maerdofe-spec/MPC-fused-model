# 滑模控制

这是一个面向复现和教学的最小控制器，只控制：

- 深度 `z`（FRD 坐标系，向下为正）；
- 航向 `psi`（绕 z 轴向右转为正）。

项目把两篇参考论文的核心做法合并成一个简化双环结构，并按本工作区的 `MPC_dual_model` 补上视觉输入和 FineSUB 下位机接口：

1. 参考 OpenAUV 论文的“外环位置/姿态 + 内环速度/力”思路；
2. 参考 ROV 滑模论文的指数趋近律，并用饱和函数代替 `sign(s)`，减小抖振；
3. 离线仿真去掉六自由度耦合、在线优化和 CFD 软件依赖；实时入口沿用融合平移模型的 CSRT 目标框、单目框宽测距和 FineSUB v3 安全通信。

> 这是控制算法原型，不是可直接下水的固件。实机使用前必须确认坐标符号、推进器分配矩阵、单推进器正反转推力曲线和安全限幅。

## 1. 动力学模型

深度和航向分别使用同一个单自由度模型：

```text
q_dot = v
m_eff * v_dot + d_l * v + d_q * |v| * v = tau + d_ext
```

其中：

- `q`：深度 `z` 或航向 `psi`；
- `v`：垂向速度 `w` 或偏航角速度 `r`；
- `m_eff`：包含附加质量/附加转动惯量的等效惯量；
- `d_l, d_q`：线性和二次水动力阻尼；
- `tau`：控制力 `Fz` 或偏航力矩 `N`；
- `d_ext`：海流、线缆和未建模耦合等合扰动。

OpenAUV 论文给出的参数直接用于深度轴：

| 参数 | 数值 | 说明 |
|---|---:|---|
| 刚体质量 | 18.5 kg | OpenAUV 论文 Table I |
| z 轴附加质量 | 16.4712 kg | OpenAUV 论文第 IV-C 节 |
| z 轴线性阻尼 | 6.2491 N/(m/s) | OpenAUV 论文第 IV-C 节 |
| z 轴二次阻尼 | 51.0402 N/(m/s)^2 | OpenAUV 论文第 IV-C 节 |
| 偏航附加转动惯量 | 0.8198 kg m^2 | OpenAUV 论文第 IV-C 节 |
| 偏航线性阻尼 | 0.0698 N m/(rad/s) | OpenAUV 论文第 IV-C 节 |
| 偏航二次阻尼 | 1.1343 N m/(rad/s)^2 | OpenAUV 论文第 IV-C 节 |

论文没有公布刚体偏航转动惯量。本项目仅为了仿真，按长方体近似：

```text
I_z = m * (length^2 + width^2) / 12
```

再加上偏航附加转动惯量。该近似值不能直接用于实机。

模型假设载体已完成近中性浮力配平；剩余恒定浮力偏差可视为 `d_ext`。默认的 `80 N` 垂向合力限幅和 `12 N m` 偏航力矩限幅也是仿真用工程假设，并非论文公布的实测执行器边界。

## 2. 简化双环控制律

### 外环：位置/航向误差变成期望速度

```text
e_q = q_ref - q
v_cmd = clip(k_q * e_q, -v_max, v_max)
```

航向误差会先折返到 `[-pi, pi]`，因此跨越正负 180 度时不会绕远路。`v_cmd` 再经过一阶平滑，得到 `v_ref` 和 `v_ref_dot`，避免阶跃参考直接产生很大的控制尖峰。

### 内环：带饱和边界层的滑模速度控制

```text
s = v_ref - v
sat(x) = clip(x, -1, 1)

tau = d_l*v + d_q*|v|*v
      + m_eff * (v_ref_dot + k_s*s + eta*sat(s/phi))
```

忽略输入限幅并把模型代回，可得：

```text
s_dot = -k_s*s - eta*sat(s/phi) - d_ext/m_eff
```

- `k_s*s` 是指数趋近项；
- `eta*sat(s/phi)` 是鲁棒切换项；
- `phi` 是边界层宽度，越小越接近 `sign(s)`，但越容易抖振；
- 最外层 `clip` 表示执行器力/力矩限幅。

这比完整六自由度 SMC 更容易复现，也保留了两篇论文最重要的控制结构。

## 3. 目录结构

```text
滑模控制/
├── openauv_smc/
│   ├── controller.py   # 双环饱和滑模控制器
│   ├── model.py        # 两自由度 OpenAUV 简化模型
│   ├── vision.py       # 目标框 -> 相机位置/深度/航向参考
│   └── finesub_interface.py # 遥测状态与控制命令适配
├── tests/
│   ├── test_controller.py
│   └── test_live_interfaces.py
├── simulate.py         # 一键仿真、CSV 和结果图
├── live_visual_control.py # GStreamer + CSRT + FineSUB 实时入口
├── finesub_live.json   # 相机、通信、标定和安全参数
└── requirements.txt
```

## 4. 一键复现

在本目录运行：

```powershell
python -m pip install -r requirements.txt
python simulate.py
python -m unittest discover -s tests -v
```

仿真会在 `results/` 下生成：

- `tracking_results.png`：深度、航向、控制量和扰动曲线；
- `simulation_history.csv`：完整时序数据；
- `metrics.json`：RMSE、最大误差和最终误差。

不需要显示图形窗口，脚本可直接在服务器或 CI 环境运行。也可以指定输出目录：

```powershell
python simulate.py --output-dir my_results
```

本项目已用默认扰动场景实际运行。最后 5 秒平均绝对误差为：深度约 `0.0036 m`，航向约 `0.082 deg`；最大控制量约为 `38.82 N` 和 `7.26 N m`，均未触及所设仿真限幅。阶跃发生瞬间的误差会计入整体 RMSE，因此最终稳态误差比全程 RMSE 更适合判断定点保持效果。

## 5. 视觉识别接口

视觉入口参考 `rov_track_control3.py` 和融合平移模型，仍采用以下数据约定：

```text
CSRT bbox = [x, y, width, height]
相机位置 = [right, down, forward]  (m)
机体系 FRD = [forward, right, down]
```

目标框宽度用于单目距离估计：

```text
forward = range_reference * target_width_at_reference / bbox_width
right   = (cx-image_cx) * forward / focal_x
down    = (cy-image_cy) * forward / focal_y
```

之后生成三个简单控制目标：

- 距离误差经过带死区的比例控制，得到前向力 `Fx`；
- 垂直像素误差生成深度参考 `z_ref = z + clip(down)`；
- 水平像素误差生成航向参考 `psi_ref = psi + atan2(right,forward)`。

当前控制命令是 `[Fx, 0, Fz, N]`：目标横向偏离主要通过转艏消除，暂不同时施加横移力，避免横移和偏航两个自由度争抢同一个视觉误差。`range_reference_m` 和 `target_width_at_reference_px` 必须针对实际目标重新测量。

这里的“视觉识别”是与现有融合模型一致的人工框选 ROI 后进行 CSRT 跟踪，并不是自动类别检测或深度神经网络识别。更换目标时需要按 `S` 重新框选。

## 6. FineSUB 下位机通信

`live_visual_control.py` 直接复用父目录 `MPC_dual_model` 中已经验证的：

- `FineSUBControlCommand` 和 `FineSUBHardwareAdapter`；
- TCP、UDP、串口和 `dry_run` 传输；
- FineSUB v3 会话号、16 位序号、发送时间和 CRC-16/MODBUS；
- 遥测解析、命令接受/拒绝确认、执行后 8 电机反馈；
- 遥测超时、确认超时、下位机 failsafe 和拒绝命令后的强制重新解锁。

因此本项目没有复制协议实现。协议升级或 CRC 修正只需要修改 `MPC_dual_model` 的唯一版本。

FineSUB 遥测直接提供 `z、psi、r`，但没有垂向速度 `w`。实时适配层用深度差分、速度限幅和一阶低通估计 `w`，构造：

```text
state = [z, w_filtered, psi, r]
```

控制器输出先变成物理量：

```text
force_body = [Fx, 0, Fz]
yaw_moment = N
```

再由融合平移模型同一个 `FineSUBHardwareAdapter` 换算为有界的：

```text
[forward, right, down, yaw]
```

Python 只发送高层通道，由 MCU 混控成 8 路推进器；不能再在 Python 中做一次 8 推进器分配。

## 7. 实时运行

先安装实时依赖：

```powershell
python -m pip install -r requirements-live.txt
python live_visual_control.py --config finesub_live.json
```

GStreamer、H.264 解码器和 PyGObject `gi` 仍需使用目标机的系统安装方式提供。默认接收 UDP `5600` 端口的 RTP/H.264 视频。

若只想检查通信程序而不连接真实下位机，可把 `finesub_live.json` 中的 `transport.type` 临时改为 `dry_run`。由于 `dry_run` 没有真实 v3 遥测和命令确认，安全门不会允许进入已确认解锁状态。

键盘操作：

| 按键 | 功能 |
|---|---|
| `S` | 框选目标；若已解锁会先停桨并要求重新解锁 |
| `Z` | 停止跟踪并清零自动控制 |
| `A` | 解锁/停桨 |
| `M` | AUTO/零输出保持切换 |
| `Q` | 退出并发送停桨帧 |

进入 AUTO 必须同时满足：新鲜遥测、v3 执行反馈、当前会话的合法命令确认和新鲜视觉观测。任一条件失效都会清零或停桨；通信故障、下位机拒绝或 failsafe 后必须人工重新解锁。

## 8. 接入其他 OpenAUV 下位机时的接口

每个控制周期向控制器提供：

```text
state = [z, w, psi, r]
reference = [z_ref, psi_ref]
```

控制器输出广义力：

```text
tau_2dof = [Fz, N]
tau_6dof = [0, 0, Fz, 0, 0, N]
```

然后必须使用实际推进器位置和方向构造 OpenAUV 论文中的分配关系：

```text
tau_6dof = T * tau_motor
```

若是冗余 8 推进器，可从 `T` 的加权伪逆和单推进器限幅开始。当前公开 GitHub 仓库没有提供可直接复用的控制代码或完整推力分配参数，因此本项目没有虚构 8 路电机输出。

## 9. 实机前必须替换的量

1. 用摆锤、CAD 或系统辨识替换偏航刚体转动惯量近似；
2. 用水池实验辨识深度/偏航阻尼；
3. 标定 T200 正反转“指令-推力”曲线；
4. 按真实安装位置建立 `6 x 8` 推力分配矩阵；
5. 从较小的 `force_limit`、`moment_limit` 和 `eta` 开始调参；
6. 增加失联停桨、深度上限、姿态异常和传感器超时保护。

当前 FineSUB 接口已经包含失联/遥测/确认超时停桨，但深度绝对上限、横滚俯仰异常和漏水检测仍应由下位机做最终保护。

## 10. 参考资料

- Z. Sha et al., “A Portable Autonomous Underwater Vehicle With Multi-Thruster Propulsion: Design, Development, and Vision-Based Tracking Control,” IEEE RA-L, 2025, DOI: [10.1109/LRA.2025.3540380](https://doi.org/10.1109/LRA.2025.3540380).
- F. Ren and Q. Hu, “ROV Sliding Mode Controller Design and Simulation,” Processes, 2023, DOI: [10.3390/pr11102359](https://doi.org/10.3390/pr11102359).
- OpenAUV 公开仓库：[schahzy/OpenAUV](https://github.com/schahzy/OpenAUV).
