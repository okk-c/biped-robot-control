# legged_gym/utils/task_registry.py
import os
from datetime import datetime
from typing import Tuple
from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner
from legged_gym import LEGGED_GYM_ROOT_DIR
from .helpers import get_args, update_cfg_from_args, class_to_dict, get_load_path, set_seed, parse_sim_params
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class TaskRegistry():
    def __init__(self):
        # 延迟导入 GO2 相关环境和配置，避免循环导入
        GO2Env, GO2Jump, GO2JumpCfg, GO2JumpCfgPPO = self._import_go2_envs()

        # 注册任务
        self.task_classes = {
            "go2": GO2Env,
            "go2_jump": GO2Jump,
        }
        self.env_cfgs = {
            "go2": GO2JumpCfg,
            "go2_jump": GO2JumpCfg,
        }
        self.train_cfgs = {
            "go2": GO2JumpCfgPPO,
            "go2_jump": GO2JumpCfgPPO,
        }

    def _import_go2_envs(self):
        from legged_gym.envs.go2.go2_env import GO2Env
        from legged_gym.envs.go2.go2_jump_env import GO2Jump
        from legged_gym.envs.go2.go2_jump_config import GO2JumpCfg, GO2JumpCfgPPO
        return GO2Env, GO2Jump, GO2JumpCfg, GO2JumpCfgPPO

    def register(self, name: str, task_class: VecEnv, env_cfg: LeggedRobotCfg, train_cfg: LeggedRobotCfgPPO):
        self.task_classes[name] = task_class
        self.env_cfgs[name] = env_cfg
        self.train_cfgs[name] = train_cfg

    def get_task_class(self, name: str) -> VecEnv:
        return self.task_classes[name]

    def get_cfgs(self, name) -> Tuple[LeggedRobotCfg, LeggedRobotCfgPPO]:
        train_cfg = self.train_cfgs[name]
        env_cfg = self.env_cfgs[name]
        env_cfg.seed = train_cfg.seed
        return env_cfg, train_cfg

    def make_env(self, name, args=None, env_cfg=None) -> Tuple[VecEnv, LeggedRobotCfg]:
        if args is None:
            args = get_args()
        if name in self.task_classes:
            task_class = self.get_task_class(name)
        else:
            raise ValueError(f"Task with name: {name} was not registered")
        if env_cfg is None:
            env_cfg, _ = self.get_cfgs(name)
        env_cfg, _ = update_cfg_from_args(env_cfg, None, args)
        set_seed(env_cfg.seed)
        sim_params = {"sim": class_to_dict(env_cfg.sim)}
        sim_params = parse_sim_params(args, sim_params)
        env = task_class(cfg=env_cfg,
                         sim_params=sim_params,
                         physics_engine=args.physics_engine,
                         sim_device=args.sim_device,
                         headless=args.headless)
        return env, env_cfg

    def make_alg_runner(self, env, name=None, args=None, train_cfg=None, log_root="default"):
        if args is None:
            args = get_args()
        if train_cfg is None:
            if name is None:
                raise ValueError("Either 'name' or 'train_cfg' must be not None")
            _, train_cfg = self.get_cfgs(name)
        else:
            if name is not None:
                print(f"'train_cfg' provided -> Ignoring 'name={name}'")
        _, train_cfg = update_cfg_from_args(None, train_cfg, args)
        if log_root=="default":
            log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
            log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + train_cfg.runner.run_name)
        elif log_root is None:
            log_dir = None
        else:
            log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + train_cfg.runner.run_name)
        train_cfg_dict = class_to_dict(train_cfg)
        runner = OnPolicyRunner(env, train_cfg_dict, log_dir, device=args.rl_device)
        resume = train_cfg.runner.resume
        if resume:
            resume_path = get_load_path(log_root, load_run=train_cfg.runner.load_run, checkpoint=train_cfg.runner.checkpoint)
            print(f"Loading model from: {resume_path}")
            runner.load(resume_path)
        return runner, train_cfg

# 全局 task_registry 实例
task_registry = TaskRegistry()
