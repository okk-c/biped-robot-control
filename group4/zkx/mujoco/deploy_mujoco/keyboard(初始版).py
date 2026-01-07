import time
import sys
import termios
import tty
import select

import mujoco.viewer
import mujoco
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import yaml


# -----------------------------
# 非阻塞键盘输入
# -----------------------------
def get_key():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        rlist, _, _ = select.select([fd], [], [], 0.001)
        if rlist:
            return sys.stdin.read(1)
        return None
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


# -----------------------------
# 键盘控制类
# -----------------------------
class KeyboardController:
    def __init__(self):
        self.vx = 0.0
        self.vy = 0.0
        self.yaw = 0.0
        self.speed_scale = 1.0

    def update(self, key):
        if key is None:
            return

        if key == "w": self.vx += 0.1 * self.speed_scale
        if key == "s": self.vx -= 0.1 * self.speed_scale
        if key == "a": self.vy += 0.1 * self.speed_scale
        if key == "d": self.vy -= 0.1 * self.speed_scale
        if key == "q": self.yaw += 0.1 * self.speed_scale
        if key == "e": self.yaw -= 0.1 * self.speed_scale

        if key == " ":
            self.vx = self.vy = self.yaw = 0

        if key == "\x1b":  # ESC
            sys.exit(0)

        self.vx = np.clip(self.vx, -1.0, 1.0)
        self.vy = np.clip(self.vy, -0.5, 0.5)
        self.yaw = np.clip(self.yaw, -1.5, 1.5)

    def get_command(self):
        return np.array([self.vx, self.vy, self.yaw], dtype=np.float32)


# -----------------------------
# 姿态处理
# -----------------------------
def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd


# -----------------------------
# 主程序
# -----------------------------
if __name__ == "__main__":
    # get config file name from command line
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    args = parser.parse_args()

    config_file = args.config_file
    with open(f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/{config_file}", "r") as f:
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

    # 替代 YAML 的 cmd_init，用键盘控制
    controller = KeyboardController()
    cmd = np.zeros(3, dtype=np.float32)

    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)
    counter = 0

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # load policy
    policy = torch.jit.load(policy_path)

    print("---- Keyboard Teleop Enabled ----")
    print("W/S: forward/back")
    print("A/D: left/right")
    print("Q/E: turn left/right")
    print("SPACE: stop")
    print("ESC: exit")

    with mujoco.viewer.launch_passive(m, d) as viewer:
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()

            # PD control
            tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
            d.ctrl[:] = tau
            mujoco.mj_step(m, d)

            counter += 1
            if counter % control_decimation == 0:

                # ---- 从键盘读取控制指令 ----
                key = get_key()
                controller.update(key)
                cmd = controller.get_command()

                # ---- 构建观测 ----
                qj = d.qpos[7:]
                dqj = d.qvel[6:]
                quat = d.qpos[3:7]
                omega = d.qvel[3:6]

                qj = (qj - default_angles) * dof_pos_scale
                dqj = dqj * dof_vel_scale
                gravity_orientation = get_gravity_orientation(quat)
                omega = omega * ang_vel_scale

                period = 0.8
                count = counter * simulation_dt
                phase = count % period / period

                sin_phase = np.sin(2 * np.pi * phase)
                cos_phase = np.cos(2 * np.pi * phase)

                obs[:3] = omega
                obs[3:6] = gravity_orientation
                obs[6:9] = cmd * cmd_scale
                obs[9:9 + num_actions] = qj
                obs[9 + num_actions:9 + 2 * num_actions] = dqj
                obs[9 + 2 * num_actions:9 + 3 * num_actions] = action
                obs[9 + 3 * num_actions:9 + 3 * num_actions + 2] = np.array([sin_phase, cos_phase])

                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                action = policy(obs_tensor).detach().numpy().squeeze()

                target_dof_pos = action * action_scale + default_angles

            viewer.sync()

            # 时间同步
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
