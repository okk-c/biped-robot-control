import time
import sys
import termios
import tty
import select
import threading

import mujoco
import mujoco.viewer
import numpy as np
import torch
import yaml
from legged_gym import LEGGED_GYM_ROOT_DIR

# =====================================
# 全局 key_state：后台线程实时更新
# =====================================
key_state = {
    "w": False,
    "s": False,
    "a": False,
    "d": False,
    "q": False,
    "e": False,
    " ": False
}

# =====================================
# 终端非阻塞按键监听（后台线程）
# =====================================
def keyboard_listener():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    tty.setcbreak(fd)

    try:
        while True:
            rlist, _, _ = select.select([fd], [], [], 0.01)
            if rlist:
                c = sys.stdin.read(1)
                if c == "w": key_state["w"] = True
                if c == "s": key_state["s"] = True
                if c == "a": key_state["a"] = True
                if c == "d": key_state["d"] = True
                if c == "q": key_state["q"] = True
                if c == "e": key_state["e"] = True
                if c == " ": key_state[" "] = True

                if c == "\x1b":  # ESC
                    print("Exit")
                    sys.exit(0)

            # 松键逻辑必须在 main loop 处理
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

# =====================================
# 姿态计算
# =====================================
def get_gravity_orientation(quaternion):
    qw, qx, qy, qz = quaternion
    return np.array([
        2 * (-qz * qx + qw * qy),
        -2 * (qz * qy + qw * qx),
        1 - 2 * (qw * qw + qz * qz)
    ])

def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd

# =====================================
# 主程序
# =====================================
if __name__ == "__main__":

    # ------------- 读取 YAML -------------
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str)
    args = parser.parse_args()

    with open(f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/{args.config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
    xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

    simulation_duration = config["simulation_duration"]
    simulation_dt = config["simulation_dt"]
    control_decimation = config["control_decimation"]

    kps = np.array(config["kps"], dtype=np.float32)
    kds = np.array(config["kds"], dtype=np.float32)
    default_angles = np.array(config["default_angles"], dtype=np.float32)

    ang_vel_scale = config["ang_vel_scale"]
    dof_pos_scale = config["dof_pos_scale"]
    dof_vel_scale = config["dof_vel_scale"]
    action_scale = config["action_scale"]
    cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

    num_actions = config["num_actions"]
    num_obs = config["num_obs"]

    # ------------- 初始化 -------------
    cmd = np.zeros(3, dtype=np.float32)
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)
    counter = 0

    # ------------- 加载模型 -------------
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    policy = torch.jit.load(policy_path)

    print("\n---- Keyboard Control Enabled ----")
    print("Hold W/S: forward/back")
    print("Hold A/D: left/right")
    print("Hold Q/E: turn left/right")
    print("SPACE: stop\n")
    print("⚠️ 注意：请点击终端窗口（不是 MuJoCo GUI）再操作键盘！")

    # =====================================
    # 启动后台键盘监听线程
    # =====================================
    t = threading.Thread(target=keyboard_listener, daemon=True)
    t.start()

    # =====================================
    # 启动 Viewer（兼容旧版本）
    # =====================================
    with mujoco.viewer.launch_passive(m, d) as viewer:

        start = time.time()

        while viewer.is_running() and time.time() - start < simulation_duration:

            step_start = time.time()

            # PD control
            tau = pd_control(target_dof_pos, d.qpos[7:], kps,
                             np.zeros_like(kds), d.qvel[6:], kds)
            d.ctrl[:] = tau
            mujoco.mj_step(m, d)

            counter += 1

            # 控制更新
            if counter % control_decimation == 0:

                # ------------- 读 key_state → 速度 -------------
                vx = vy = yaw = 0.0

                if key_state["w"]: vx = 1.0
                if key_state["s"]: vx = -1.0
                if key_state["a"]: vy = 0.5
                if key_state["d"]: vy = -0.5
                if key_state["q"]: yaw = 1.0
                if key_state["e"]: yaw = -1.0
                if key_state[" "]: vx = vy = yaw = 0.0

                # 重置 key_state（以便下一循环继续捕获）
                for k in key_state:
                    key_state[k] = False

                cmd = np.array([vx, vy, yaw], dtype=np.float32)

                # -------- 构建 obs --------
                qj = (d.qpos[7:] - default_angles) * dof_pos_scale
                dqj = d.qvel[6:] * dof_vel_scale

                quat = d.qpos[3:7]
                omega = d.qvel[3:6] * ang_vel_scale
                gravity_orientation = get_gravity_orientation(quat)

                period = 0.8
                phase = (counter * simulation_dt) % period / period
                sp = np.sin(2 * np.pi * phase)
                cp = np.cos(2 * np.pi * phase)

                obs[:3] = omega
                obs[3:6] = gravity_orientation
                obs[6:9] = cmd * cmd_scale
                obs[9:9 + num_actions] = qj
                obs[9 + num_actions:9 + 2 * num_actions] = dqj
                obs[9 + 2 * num_actions:9 + 3 * num_actions] = action
                obs[9 + 3 * num_actions:9 + 3 * num_actions + 2] = np.array([sp, cp])

                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                action = policy(obs_tensor).detach().numpy().squeeze()
                target_dof_pos = action * action_scale + default_angles

            viewer.sync()

            time.sleep(max(0, m.opt.timestep - (time.time() - step_start)))
