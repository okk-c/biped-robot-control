# legged_gym/envs/go2/go2_jump_env.py
import torch
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.envs.go2.go2_config import GO2RoughCfg

class GO2Jump(LeggedRobot):
    """GO2 跳跃任务，冲量式跳跃"""
    def __init__(self, cfg: GO2RoughCfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    def _resample_commands(self, env_ids):
        super()._resample_commands(env_ids)
        if len(env_ids) == 0:
            return
        p_jump = 0.05
        rand = torch.rand(len(env_ids), device=self.device)
        jumps = (rand < p_jump).float()
        self.commands[env_ids, 4] = jumps

    def compute_observations(self):
        super().compute_observations()
        contact_thresh = 1.0
        foot_contacts = (torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) > contact_thresh).float()
        n_feet = self.feet_indices.shape[0]
        if foot_contacts.shape[1] != n_feet:
            foot_contacts = torch.zeros(self.num_envs, n_feet, device=self.device)
        jump_cmd = self.commands[:, 4].unsqueeze(-1)
        self.obs_buf = torch.cat([self.obs_buf, foot_contacts, jump_cmd], dim=-1)
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)

    def _reward_jump_takeoff(self):
        v_z = self.base_lin_vel[:, 2]
        jump_cmd = self.commands[:, 4]
        return jump_cmd * torch.clip(v_z, min=0.0, max=5.0)

    def _reward_jump_air_time(self):
        jump_cmd = self.commands[:, 4]
        mean_air_time = torch.mean(self.feet_air_time, dim=1)
        return jump_cmd * torch.clip(mean_air_time, min=0.0, max=2.0)

    def _reward_jump_landing(self):
        contact = (self.contact_forces[:, self.feet_indices, 2] > 1.0)
        any_contact = torch.any(contact, dim=1)
        first_contact = (any_contact & (torch.mean(self.feet_air_time, dim=1) > 0.0)).float()
        ang_vel_norm = torch.sum(self.base_ang_vel[:, :2] ** 2, dim=1)
        stability = torch.exp(-ang_vel_norm)
        jump_cmd = self.commands[:, 4]
        return jump_cmd * first_contact * stability

    def _reward_jump_height(self):
        base_height = self.root_states[:, 2]
        baseline = self.cfg.rewards.base_height_target
        jump_cmd = self.commands[:, 4]
        return jump_cmd * torch.clip(base_height - baseline, min=0.0, max=1.0)
