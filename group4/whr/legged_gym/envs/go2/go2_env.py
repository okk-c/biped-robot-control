from legged_gym.envs.base.legged_robot import LeggedRobot
from .go2_config import GO2RoughCfg, GO2RoughCfgPPO


class Go2Env(LeggedRobot):
    """
    GO2 环境类（继承 LeggedRobot）
    """

    def __init__(
        self,
        cfg: GO2RoughCfg = None,
        sim_params=None,
        physics_engine="physx",
        sim_device="cuda:0",
        headless=False,
    ):

        if cfg is None:
            cfg = GO2RoughCfg()

        super().__init__(
            cfg=cfg,
            sim_params=sim_params,
            physics_engine=physics_engine,
            sim_device=sim_device,
            headless=headless,
        )

    @staticmethod
    def task_name():
        return "go2"