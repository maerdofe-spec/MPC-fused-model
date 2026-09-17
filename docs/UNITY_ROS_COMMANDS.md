# Unity ROS MPC 指令

本文档记录 `UnderwaterVision` 场景连接 MPC 时常用的终端指令。

## 1. 编译 ROS 包

修改 ROS 节点、launch 文件或配置后执行：

```bash
cd /home/fins/Zhouyuheng_workspace/FinsSim/ros2_ws
.venv/bin/colcon build \
  --packages-select finssim_motion_control \
  --symlink-install
```

只修改 Unity 场景或 MPC Python 源码时，通常不需要重新编译 ROS 包。

## 2. 启动平移 MPC

平移模型不主动控制 yaw，Unity 姿态 PID 保持 yaw、roll 和 pitch。

```bash
cd /home/fins/Zhouyuheng_workspace/FinsSim/ros2_ws
source install/setup.bash

ros2 launch finssim_motion_control unity_mpc_translation.launch.py \
  model_backend:=dual_model \
  baseline_update_mode:=gated_ema \
  show_plot:=true \
  plot_output:=/home/fins/Zhouyuheng_workspace/FinsSim/ros2_ws/mpc_translation_live.png
```

## 3. 启动平移加旋转 MPC

旋转模型会共同优化平移力和 yaw 力矩，并执行八推进器联合可达域约束。

```bash
cd /home/fins/Zhouyuheng_workspace/FinsSim/ros2_ws
source install/setup.bash

ros2 launch finssim_motion_control unity_mpc_translation.launch.py \
  model_backend:=dual_model_yaw \
  baseline_update_mode:=gated_ema \
  show_plot:=true \
  plot_output:=/home/fins/Zhouyuheng_workspace/FinsSim/ros2_ws/mpc_rotation_live.png
```

## 4. Unity 操作顺序

1. 先运行上面的 ROS 启动命令。
2. 在 Unity 中打开 `Assets/Scenes/UnderwaterVision.unity`。
3. 点击 Unity 的 Play。
4. 收到 MPC 诊断数据后，实时误差图会自动出现。
5. 停止时先点击 Unity 的 Stop，再在 ROS 终端按 `Ctrl+C`。

不要同时启动平移和旋转两套 ROS 进程，否则它们会争用端口 `30052` 和同一个控制话题。

## 5. 检查当前模型

另开一个终端：

```bash
cd /home/fins/Zhouyuheng_workspace/FinsSim/ros2_ws
source install/setup.bash
ros2 param get /mpc_translation_bridge model_backend
```

预期输出：

```text
dual_model          # 平移 MPC
dual_model_yaw      # 平移加旋转 MPC
```

## 6. 查看误差图

平移模型：

```bash
xdg-open /home/fins/Zhouyuheng_workspace/FinsSim/ros2_ws/mpc_translation_live.png
```

旋转模型：

```bash
xdg-open /home/fins/Zhouyuheng_workspace/FinsSim/ros2_ws/mpc_rotation_live.png
```

如果实时图形窗口没有弹出，在启动 ROS 前执行：

```bash
export DISPLAY=:0
export XAUTHORITY=/run/user/1000/gdm/Xauthority
```

## 7. 常用诊断

查看 MPC 诊断话题：

```bash
ros2 topic echo /sim/finsrov/mpc/tracking_diagnostics
```

查看诊断发布频率：

```bash
ros2 topic hz /sim/finsrov/mpc/tracking_diagnostics
```

查看 Unity 相对目标输入：

```bash
ros2 topic echo /sim/finsrov/mpc/target_relative_body
```
