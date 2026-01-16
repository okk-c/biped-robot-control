import sys
import os
import termios
import tty
import select
import numpy as np

from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch

# -----------------------------
# 非阻塞键盘输入
# -----------------------------
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


# -----------------------------
# 键盘控制逻辑
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

        if key == "w":
            self.vx += 0.1 * self.speed_scale
        if key == "s":
            self.vx -= 0.1 * self.speed_scale

        if key == "a":
            self.vy += 0.1 * self.speed_scale
        if key == "d":
            self.vy -= 0.1 * self.speed_scale

        if key == "q":
            self.yaw += 0.1 * self.speed_scale
        if key == "e":
            self.yaw -= 0.1 * self.speed_scale

        if key == "z":
            self.speed_scale = max(0.2, self.speed_scale - 0.1)
        if key == "x":
            self.speed_scale = min(3.0, self.speed_scale + 0.1)

        if key == " ":
            self.vx = self.vy = self.yaw = 0

        if key == "\x1b":  # ESC
            sys.exit(0)

        self.vx = np.clip(self.vx, -1.0, 1.0)
        self.vy = np.clip(self.vy, -0.5, 0.5)
        self.yaw = np.clip(self.yaw, -1.5, 1.5)

    def get_commands(self):
        return np.array([self.vx, self.vy, self.yaw, 0.0], dtype=np.float32)


# -----------------------------
# 主程序
# -----------------------------
def play(args):
    print("---- Keyboard Teleop for GO2 ----")
    print("W/S: forward/back")
    print("A/D: left/right")
    print("Q/E: turn left/right")
    print("Z/X: speed +/-")
    print("SPACE: stop")
    print("ESC: exit")

    # 加载配置
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # 单环境运行
    env_cfg.env.num_envs = 1
    env_cfg.env.test = False
    env_cfg.terrain.mesh_type = "trimesh"
    env_cfg.commands.curriculum = False
    env_cfg.commands.heading_command = False
    
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    # 创建环境
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    # 加载训练策略
    train_cfg.runner.resume = True
    train_cfg.runner.load_run = args.load_run
    train_cfg.runner.checkpoint = args.checkpoint
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env,
        name=args.task,
        args=args,
        train_cfg=train_cfg
    )
    policy = ppo_runner.get_inference_policy(device=env.device)

    obs = env.get_observations()

    controller = KeyboardController()

    while True:
        key = get_key()
        controller.update(key)

        # 写入 command（速度指令）
        cmd = controller.get_commands()
        env.commands[:] = torch.tensor(cmd, device=env.device).unsqueeze(0)

        # 策略输出动作
        actions = policy(obs)

        # 执行
        obs, _, _, _, _ = env.step(actions)


if __name__ == "__main__":
    args = get_args()
    play(args)
