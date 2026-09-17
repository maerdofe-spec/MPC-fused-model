# 实机标定与调参清单

以下量必须在水池、拖曳台、固定架或回放数据实验中确定。代码中的默认值仅用于软件仿真，不能视为 FineSUB 的实机参数。

## 1. 坐标系与视觉

| 参数 | 代码位置 | 单位 | 如何确定 |
|---|---|---:|---|
| `rotation_body_from_camera` | `camera_transform.py` | 无 | 采集多个已知相机点和机体系点，拟合刚体旋转；验证相机前/右/下分别映射到机体前/右/下。 |
| `camera_origin_in_body` | `camera_transform.py` | m | 测量左相机光心相对机体控制原点的前/右/下位移。 |
| 双目尺度、左右相机外参 | 视觉系统上游 | m | 用已知距离标定板或标尺验证三维输出尺度。 |
| `position_std` | `KalmanConfig` | m | ROV 和目标固定时采集位置序列，取各轴标准差。 |
| 相机时间戳与总延迟 | 集成层 | s | 记录曝光、传输、推理到控制命令的时间；用于实际 `dt` 和后续延时滤波。 |

## 2. ROV 平移动力学

| 参数 | 代码位置 | 单位 | 如何确定 |
|---|---|---:|---|
| `M_t` | `FixedLinearDampingRelativeModel` | kg（含附加质量） | 三轴阶跃推力/自由衰减实验拟合；允许为 3x3 非对角矩阵。 |
| `D_L` | `FixedLinearDampingRelativeModel` | N/(m/s) | 三轴匀速或自由衰减数据拟合固定线性阻尼。 |
| `dt` | `FixedLinearDampingRelativeModel` | s | 使用相邻有效三维量测的真实时间间隔；不要只假设 0.05。 |
| `acceleration_std` | `KalmanConfig` | m/s² | 从鱼突转、水流和模型残差统计得到；它描述模型无法解释的加速度。 |

## 3. 推进器和低层控制链

| 参数 | 代码位置 | 单位 | 如何确定 |
|---|---|---:|---|
| `force_min/max` | `MPCConfig` | N | 以连续可用推力，不是瞬时额定峰值，作为上限。 |
| `delta_force_min/max` | `MPCConfig` | N/控制周期 | 阶跃试验中由安全推力变化率和执行器响应确定。 |
| `positive_force_at_limit` | `ForceCommandAdapter`、分配器 | N | 分别标定 forward/right/down 的满命令实际综合力。 |
| `signs` | `ForceCommandAdapter` | ±1 | 固定 ROV、低推力逐轴测试正负方向。 |
| `command_limits` | `ForceCommandAdapter` | 设备命令 | 从 MCU 协议和安全限幅确认。 |
| `translation_channel_limits`、`attitude_channel_limits` | `FineSUBThrusterAllocator` | 归一化通道 | 在不发生单电机持续饱和的条件下调定。 |
| `deadband` | `FineSUBThrusterAllocator` | 归一化通道 | 测量推进器开始稳定产生推力的最小命令。 |
| 每台推进器油门-推力曲线、响应延迟 | 当前代码未实现 | N、s | 用推力计、RPM/电流或水池实验建立；用于估计 `tau_achieved`。 |

## 4. MPC 任务与安全参数

| 参数 | 代码位置 | 单位 | 如何确定 |
|---|---|---:|---|
| `reference_position` | `MPCConfig` | m | 任务要求的鱼相对位置，例如 `[0.6,0,0]`。 |
| `horizon` | `MPCConfig` | 控制步 | 先用 8-12；与 `dt` 一起决定预见时间。 |
| `position_weights` | `MPCConfig` | 代价权重 | 由允许位置误差归一化后，在仿真中调节。 |
| `velocity_weights` | `MPCConfig` | 代价权重 | 由允许相对速度和超调容忍度调节。 |
| `force_weights` | `MPCConfig` | 代价权重 | 模型一中惩罚 tau[k]-tau[k-1]；平衡跟踪动作与推力变化。 |
| `delta_force_weights` | `MPCConfig` | 代价权重 | 由允许的力变化和平滑性确定。 |
| `forward_distance_min/max` | `MPCConfig` | m | 由安全距离、双目有效工作距离确定。 |
| `horizontal_half_fov_deg`、`vertical_half_fov_deg` | `MPCConfig` | deg | 用实际镜头有效半视场，而不是宣传全视场。 |
| `fov_margin_deg` | `MPCConfig` | deg | 留给检测框误差、相机畸变和控制延迟的安全余量。 |
| 松弛权重与 `slack_max` | `MPCConfig` | 代价、m | 先在仿真中确认突转时可行，再限制可接受的违规程度。 |

## 5. 模型一基础力管理

| 参数/动作 | 位置 | 单位 | 标定方法 |
|---|---|---:|---|
| `tau_base` | `FixedLinearDampingRelativeModel` | N | AUTO 启用时锁存当前实际保持力，只用于安全回退。 |
| `tau_achieved_previous` | `MPCTracker.update` | N | 优先使用力观测或推进器模型估计；没有反馈时用上一拍饱和命令近似。 |
| `baseline_adaptation.enabled` | `BaselineAdaptationConfig` | bool | 默认关闭；只有已验证稳态保持力估计后才启用。 |
| `adaptation_rate` | `BaselineAdaptationConfig` | 0-1 | 启用时从小值开始，防止把目标机动或视觉偏差吸收到基础力。 |

## 6. 求解器与运行时安全

| 参数 | 代码位置 | 单位 | 如何确定 |
|---|---|---:|---|
| `backend` | `QPSolverSettings` | 名称 | 实机优先 `osqp`；仿真可 `auto`。 |
| `time_limit_seconds` | `QPSolverSettings` | s | 小于控制周期并留给视觉、通信和安全逻辑时间。 |
| `max_iterations`、`absolute_tolerance`、`relative_tolerance` | `QPSolverSettings` | 无 | 通过最坏工况仿真平衡精度与时间。 |
| `BaselineAdaptationConfig` | `mpc_tracker.py` | 无、m、m/s | 仅在基础力可靠、匹配状态稳定时启用。 |

## 上机顺序

1. 固定 ROV、低推力确认相机坐标和三轴正负号。
2. 标定相机外参和双目尺度。
3. 标定推进器命令到实际综合力，并记录响应延迟。
4. 用固定目标和低推力辨识 `M_t`、`D_L`。
5. 在仿真中调 MPC 权重、限幅和基础力回退。
6. 水池低限幅闭环，仅启用前向轴。
7. 依次加入右向、下向和目标机动，检查模型一假设的适用范围。

