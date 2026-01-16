import sys
import os
import termios
import tty
import select
import numpy as np

from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch


# ================================================================
# 非阻塞键盘输入
# ================================================================
def get_key():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        rlist, _, _ = select.select([fd], [], [], 0.01)
        if rlist:
            return sys.stdin.read(1)
        return None
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


# ================================================================
# 键盘控制
# ================================================================
class KeyboardController:
    def __init__(self):
        self.vx = 0.0
        self.vy = 0.0
        self.yaw = 0.0
        self.speed_scale = 1.0
        self.heading = 0.0


    def update(self, key):
        if key is None:
            return

        # 速度控制
        if key == "w": self.vx += 0.1 * self.speed_scale
        if key == "s": self.vx -= 0.1 * self.speed_scale
        if key == "a": self.vy += 0.1 * self.speed_scale
        if key == "d": self.vy -= 0.1 * self.speed_scale

        # 旋转控制
        if key == "q": self.heading += 0.5 * self.speed_scale
        if key == "e": self.heading -= 0.5 * self.speed_scale

        # 调整速度倍率
        if key == "z": self.speed_scale = max(0.2, self.speed_scale - 0.1)
        if key == "x": self.speed_scale = min(3.0, self.speed_scale + 0.1)

        # 停止
        if key == " ":
            self.vx = self.vy = self.heading =self.yaw = 0

        # ESC 退出
        if key == "\x1b":
            sys.exit(0)

        # 限幅
        self.vx = np.clip(self.vx, -1.0, 1.0)
        self.vy = np.clip(self.vy, -0.5, 0.5)
        self.yaw = np.clip(self.yaw, -1.5, 1.5)

    def get_commands(self):
        return np.array([self.vx, self.vy, 0.0, self.heading], dtype=np.float32)


# ================================================================
# 主函数
# ================================================================
def play(args):
    print("=== ARGS DUMP ===")
    print("task:", args.task)
    print("load_run:", args.load_run)
    print("checkpoint:", args.checkpoint)
    print("num_envs:", args.num_envs)
    print("=================")
    print("\n---- Keyboard Teleop for G1 (Complex Terrain Enabled) ----")
    print("W/S: forward/back")
    print("A/D: left/right")
    print("Q/E: yaw left/right")
    print("Z/X: lower/raise speed scale")
    print("SPACE: stop")
    print("ESC: exit\n")

    # 获取默认配置
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # 单环境
    env_cfg.env.num_envs = 1
    env_cfg.env.test = False

    # *** 复杂地形 ***
    terrain = env_cfg.terrain

    terrain.mesh_type = "trimesh"
    terrain.curriculum = True        # 禁用课程，否则回到平地
    terrain.selected = False          # 禁用静态地形选择

    terrain.horizontal_scale = 0.05
    terrain.vertical_scale = 0.005

    terrain.num_rows = 5
    terrain.num_cols = 5

    # 地形比例（你的 terrain.py 支持所有这些）
    terrain.terrain_proportions = [
        0.0,  # slope                   #yes
        0.0,  # rough slope             #yes
        0.0,  # stairs                  #yes
        0.0,  # obstacles               #yes
        0.0,  # discrete                #yes
        0.0,  # stepping stones         #yes
        0.0,  # gap                     #yes
        0.0,  # pit                     #yes
    ]

    # ================================================================

    # 禁用 domain randomization
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    # 创建环境（用 terrain.py 构建地形）
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    # 加载策略
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    policy = ppo_runner.get_inference_policy(device=env.device)

    obs = env.get_observations()
    controller = KeyboardController()

    # 运行循环
    while True:
        key = get_key()
        controller.update(key)

        cmd = controller.get_commands()
        env.commands[:] = torch.tensor(cmd, device=env.device).unsqueeze(0)

        actions = policy(obs)
        obs, _, _, _, _ = env.step(actions)


if __name__ == "__main__":
    args = get_args()
    play(args)
