from legged_gym import LEGGED_GYM_ROOT_DIR, envs  # 导入legged_gym库的根目录和环境模块
import time  # 导入时间模块，用于模拟时间控制
from warnings import WarningMessage  # 导入警告消息模块（虽然导入但未使用）
import numpy as np  # 导入numpy库，用于数值计算
import os  # 导入操作系统模块，用于文件路径操作

from isaacgym.torch_utils import *  # 导入isaacgym的torch工具函数
from isaacgym import gymtorch, gymapi, gymutil  # 导入isaacgym的核心API

import torch  # 导入PyTorch深度学习框架
from torch import Tensor  # 导入PyTorch的Tensor类型
from typing import Tuple, Dict  # 导入类型提示，用于函数签名

from legged_gym import LEGGED_GYM_ROOT_DIR  # 再次导入根目录（重复导入，可以删除）
from legged_gym.envs.base.base_task import BaseTask  # 导入基础任务类
from legged_gym.utils.math import wrap_to_pi  # 导入角度包装函数，将角度限制在[-π, π]
from legged_gym.utils.isaacgym_utils import get_euler_xyz as get_euler_xyz_in_tensor  # 导入欧拉角转换函数
from legged_gym.utils.helpers import class_to_dict  # 导入类转字典的辅助函数
from .legged_robot_config import LeggedRobotCfg  # 导入当前目录下的机器人配置类

class LeggedRobot(BaseTask):  # 定义四足机器人任务类，继承自BaseTask
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        """ 解析提供的配置文件，
            调用create_sim()（创建模拟和环境），
            初始化训练期间使用的PyTorch缓冲区

        参数:
            cfg (Dict): 环境配置文件
            sim_params (gymapi.SimParams): 模拟参数
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX（必须是PhysX）
            device_type (string): 'cuda' 或 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): 如果为True则以无头模式运行（无渲染）
        """
        self.cfg = cfg  # 存储配置对象
        self.sim_params = sim_params  # 存储模拟参数
        self.height_samples = None  # 初始化高度采样为None
        self.debug_viz = False  # 调试可视化标志，默认为False
        self.init_done = False  # 初始化完成标志，默认为False
        self._parse_cfg(self.cfg)  # 解析配置文件
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)  # 调用父类初始化

        if not self.headless:  # 如果不是无头模式
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)  # 设置相机位置和观察点
        self._init_buffers()  # 初始化PyTorch缓冲区
        self._prepare_reward_function()  # 准备奖励函数
        self.init_done = True  # 设置初始化完成标志为True

    def step(self, actions):
        """ 应用动作，模拟，调用self.post_physics_step()

        参数:
            actions (torch.Tensor): 形状为(num_envs, num_actions_per_env)的张量
        """

        clip_actions = self.cfg.normalization.clip_actions  # 获取动作裁剪范围
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)  # 裁剪动作并转移到设备
        # 逐步进行物理模拟和渲染每一帧
        self.render()  # 渲染当前帧
        for _ in range(self.cfg.control.decimation):  # 根据控制降采样次数循环
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)  # 计算扭矩
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))  # 设置关节驱动力
            self.gym.simulate(self.sim)  # 执行物理模拟
            if self.cfg.env.test:  # 如果是测试模式
                elapsed_time = self.gym.get_elapsed_time(self.sim)  # 获取已用时间
                sim_time = self.gym.get_sim_time(self.sim)  # 获取模拟时间
                if sim_time-elapsed_time>0:  # 如果模拟时间快于实际时间
                    time.sleep(sim_time-elapsed_time)  # 休眠以同步实时
            
            if self.device == 'cpu':  # 如果设备是CPU
                self.gym.fetch_results(self.sim, True)  # 获取模拟结果
            self.gym.refresh_dof_state_tensor(self.sim)  # 刷新关节状态张量
        self.post_physics_step()  # 调用物理后处理步骤

        # 返回裁剪后的观测、裁剪后的状态（None）、奖励、完成标志和额外信息
        clip_obs = self.cfg.normalization.clip_observations  # 获取观测裁剪范围
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)  # 裁剪观测
        if self.privileged_obs_buf is not None:  # 如果有特权观测
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)  # 裁剪特权观测
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras  # 返回所有输出

    def post_physics_step(self):
        """ 检查终止条件，计算观测和奖励
            调用self._post_physics_step_callback()进行通用计算
            如果需要则调用self._draw_debug_vis()
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)  # 刷新演员根状态张量
        self.gym.refresh_net_contact_force_tensor(self.sim)  # 刷新净接触力张量

        self.episode_length_buf += 1  # 增加当前回合长度计数器
        self.common_step_counter += 1  # 增加通用步数计数器

        # 准备各种量
        self.base_pos[:] = self.root_states[:, 0:3]  # 获取基础位置（x, y, z）
        self.base_quat[:] = self.root_states[:, 3:7]  # 获取基础四元数
        self.rpy[:] = get_euler_xyz_in_tensor(self.base_quat[:])  # 将四元数转换为欧拉角（滚转、俯仰、偏航）
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])  # 获取基础线速度（基坐标系下）
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])  # 获取基础角速度（基坐标系下）
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)  # 获取投影重力（基坐标系下）

        self._post_physics_step_callback()  # 调用后处理回调函数

        # 计算观测、奖励、重置等...
        self.check_termination()  # 检查终止条件
        self.compute_reward()  # 计算奖励
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()  # 获取需要重置的环境ID
        self.reset_idx(env_ids)  # 重置这些环境
        
        if self.cfg.domain_rand.push_robots:  # 如果启用了随机推动机器人
            self._push_robots()  # 随机推动机器人

        self.compute_observations()  # 计算观测（某些情况下可能需要模拟步骤来刷新观测，例如身体位置）

        self.last_actions[:] = self.actions[:]  # 存储上一时刻的动作
        self.last_dof_vel[:] = self.dof_vel[:]  # 存储上一时刻的关节速度
        self.last_root_vel[:] = self.root_states[:, 7:13]  # 存储上一时刻的根速度

    def check_termination(self):
        """ 检查环境是否需要重置
        """
        # 检查终止接触点的接触力是否大于1（表示碰撞）
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        # 检查俯仰角或滚转角是否超过阈值（机器人翻倒）
        self.reset_buf |= torch.logical_or(torch.abs(self.rpy[:,1])>1.0, torch.abs(self.rpy[:,0])>0.8)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length  # 检查是否超时
        self.reset_buf |= self.time_out_buf  # 超时也触发重置

    def reset_idx(self, env_ids):
        """ 重置某些环境。
            调用self._reset_dofs(env_ids), self._reset_root_states(env_ids), 和self._resample_commands(env_ids)
            [可选] 调用self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) 和
            记录回合信息
            重置一些缓冲区

        参数:
            env_ids (list[int]): 需要重置的环境ID列表
        """
        if len(env_ids) == 0:  # 如果没有需要重置的环境
            return
        
        # 重置机器人状态
        self._reset_dofs(env_ids)  # 重置关节状态
        self._reset_root_states(env_ids)  # 重置根状态

        self._resample_commands(env_ids)  # 重新采样命令

        # 重置缓冲区
        self.actions[env_ids] = 0.  # 重置动作为零
        self.last_actions[env_ids] = 0.  # 重置上一动作为零
        self.last_dof_vel[env_ids] = 0.  # 重置上一关节速度为零
        self.feet_air_time[env_ids] = 0.  # 重置脚部空中时间为零
        self.episode_length_buf[env_ids] = 0  # 重置回合长度为零
        self.reset_buf[env_ids] = 1  # 设置重置缓冲区标志
        # 填充额外信息
        self.extras["episode"] = {}  # 初始化回合额外信息字典
        for key in self.episode_sums.keys():  # 遍历所有奖励项
            # 计算平均奖励并存储到额外信息中
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.  # 重置该奖励项的回合总和
        if self.cfg.commands.curriculum:  # 如果启用了命令课程学习
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]  # 记录当前最大x方向线速度命令
        # 将超时信息发送给算法
        if self.cfg.env.send_timeouts:  # 如果配置要求发送超时信息
            self.extras["time_outs"] = self.time_out_buf  # 存储超时缓冲区
    
    def compute_reward(self):
        """ 计算奖励
            调用每个非零奖励尺度的奖励函数（在self._prepare_reward_function()中处理）
            将每项奖励加到回合总和和总奖励中
        """
        self.rew_buf[:] = 0.  # 重置奖励缓冲区为零
        for i in range(len(self.reward_functions)):  # 遍历所有奖励函数
            name = self.reward_names[i]  # 获取奖励名称
            rew = self.reward_functions[i]() * self.reward_scales[name]  # 计算奖励值乘以尺度
            self.rew_buf += rew  # 累加到总奖励
            self.episode_sums[name] += rew  # 累加到回合奖励总和
        if self.cfg.rewards.only_positive_rewards:  # 如果只允许正奖励
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)  # 将负奖励裁剪为零
        # 裁剪后添加终止奖励
        if "termination" in self.reward_scales:  # 如果配置中有终止奖励
            rew = self._reward_termination() * self.reward_scales["termination"]  # 计算终止奖励
            self.rew_buf += rew  # 累加到总奖励
            self.episode_sums["termination"] += rew  # 累加到终止奖励总和
    
    def compute_observations(self):
        """ 计算观测
        """
        # 拼接各种观测分量形成最终观测向量
        self.obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,  # 基础线速度（缩放后）
                                    self.base_ang_vel  * self.obs_scales.ang_vel,  # 基础角速度（缩放后）
                                    self.projected_gravity,  # 投影重力
                                    self.commands[:, :3] * self.commands_scale,  # 命令（缩放后）
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,  # 关节位置偏移（缩放后）
                                    self.dof_vel * self.obs_scales.dof_vel,  # 关节速度（缩放后）
                                    self.actions  # 当前动作
                                    ),dim=-1)  # 沿最后一维拼接
        # 如果不是盲模式，添加感知输入（注释掉的代码）
        # 如果需要，添加噪声
        if self.add_noise:  # 如果启用了噪声
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec  # 添加均匀分布噪声

    def create_sim(self):
        """ 创建模拟、地形和环境
        """
        self.up_axis_idx = 2  # 2表示z轴向上，1表示y轴向上 -> 相应地调整重力
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)  # 创建模拟实例
        self._create_ground_plane()  # 创建地平面
        self._create_envs()  # 创建环境

    def set_camera(self, position, lookat):
        """ 设置相机位置和方向
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])  # 创建相机位置向量
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])  # 创建相机目标点向量
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)  # 设置相机视角

    #------------- 回调函数 --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ 回调函数，允许存储/更改/随机化每个环境的刚体形状属性。
            在环境创建期间调用。
            基础行为：随机化每个环境的摩擦系数

        参数:
            props (List[gymapi.RigidShapeProperties]): 资产每个形状的属性
            env_id (int): 环境ID

        返回:
            [List[gymapi.RigidShapeProperties]]: 修改后的刚体形状属性
        """
        if self.cfg.domain_rand.randomize_friction:  # 如果启用了摩擦系数随机化
            if env_id==0:  # 只在第一个环境初始化时准备随机化
                # 准备摩擦系数随机化
                friction_range = self.cfg.domain_rand.friction_range  # 获取摩擦系数范围
                num_buckets = 64  # 设置桶的数量
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))  # 为每个环境随机分配桶ID
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')  # 创建摩擦系数桶
                self.friction_coeffs = friction_buckets[bucket_ids]  # 为每个环境分配摩擦系数

            for s in range(len(props)):  # 遍历所有形状属性
                props[s].friction = self.friction_coeffs[env_id]  # 设置摩擦系数
        return props

    def _process_dof_props(self, props, env_id):
        """ 回调函数，允许存储/更改/随机化每个环境的自由度属性。
            在环境创建期间调用。
            基础行为：存储URDF中定义的位置、速度和扭矩限制

        参数:
            props (numpy.array): 资产每个自由度的属性
            env_id (int): 环境ID

        返回:
            [numpy.array]: 修改后的自由度属性
        """
        if env_id==0:  # 只在第一个环境初始化时处理
            # 初始化各种限制张量
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):  # 遍历所有自由度属性
                self.dof_pos_limits[i, 0] = props["lower"][i].item()  # 存储位置下限
                self.dof_pos_limits[i, 1] = props["upper"][i].item()  # 存储位置上限
                self.dof_vel_limits[i] = props["velocity"][i].item()  # 存储速度限制
                self.torque_limits[i] = props["effort"][i].item()  # 存储扭矩限制
                # 软限制处理
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2  # 计算位置限制中点
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]  # 计算位置限制范围
                # 根据配置调整软限制范围
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props

    def _process_rigid_body_props(self, props, env_id):
        # 注释掉的代码：用于调试和质量随机化之前的打印
        # if env_id==0:
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")
        # 随机化基础质量
        if self.cfg.domain_rand.randomize_base_mass:  # 如果启用了基础质量随机化
            rng = self.cfg.domain_rand.added_mass_range  # 获取质量添加范围
            props[0].mass += np.random.uniform(rng[0], rng[1])  # 为基础质量添加随机值
        return props
    
    def _post_physics_step_callback(self):
        """ 在计算终止条件、奖励和观测之前调用的回调函数
            默认行为：基于目标和航向计算角速度命令，计算测量的地形高度并随机推动机器人
        """
        # 
        # 根据命令重新采样时间间隔，选择需要重新采样命令的环境
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)  # 重新采样这些环境的命令
        if self.cfg.commands.heading_command:  # 如果是航向命令模式
            forward = quat_apply(self.base_quat, self.forward_vec)  # 计算前进方向
            heading = torch.atan2(forward[:, 1], forward[:, 0])  # 计算机器人当前航向
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)  # 计算并限制角速度命令

    def _resample_commands(self, env_ids):
        """ 随机选择某些环境的命令

        参数:
            env_ids (List[int]): 需要新命令的环境ID
        """
        # 为指定环境重新采样x方向线速度命令
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        # 为指定环境重新采样y方向线速度命令
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:  # 如果是航向命令模式
            # 重新采样航向命令
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:  # 如果是角速度命令模式
            # 重新采样偏航角速度命令
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        # 将小命令设置为零（避免微小命令导致的不稳定）
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def _compute_torques(self, actions):
        """ 从动作计算扭矩。
            动作可以解释为位置或速度目标，传递给PD控制器，或直接作为缩放后的扭矩。
            [注意]：扭矩必须与自由度数具有相同的维度，即使某些自由度未被驱动。

        参数:
            actions (torch.Tensor): 动作

        返回:
            [torch.Tensor]: 发送到模拟的扭矩
        """
        # PD控制器
        actions_scaled = actions * self.cfg.control.action_scale  # 缩放动作
        control_type = self.cfg.control.control_type  # 获取控制类型
        if control_type=="P":  # 位置控制
            torques = self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel
        elif control_type=="V":  # 速度控制
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":  # 扭矩控制
            torques = actions_scaled
        else:  # 未知控制类型
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)  # 裁剪扭矩到限制范围内

    def _reset_dofs(self, env_ids):
        """ 重置选定环境的自由度位置和速度
            位置在默认位置的0.5到1.5倍范围内随机选择。
            速度设置为零。

        参数:
            env_ids (List[int]): 环境ID
        """
        # 随机初始化关节位置
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        self.dof_vel[env_ids] = 0.  # 关节速度设为零

        env_ids_int32 = env_ids.to(dtype=torch.int32)  # 转换为int32类型
        # 设置选定环境的关节状态
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    def _reset_root_states(self, env_ids):
        """ 重置选定环境的根状态位置和速度
            基于课程设置基础位置
            在-0.5:0.5 [m/s, rad/s]范围内选择随机化的基础速度
        参数:
            env_ids (List[int]): 环境ID
        """
        # 基础位置
        if self.custom_origins:  # 如果使用自定义原点
            self.root_states[env_ids] = self.base_init_state  # 初始化为基础初始状态
            self.root_states[env_ids, :3] += self.env_origins[env_ids]  # 添加环境原点
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device)  # 在中心1米范围内随机化xy位置
        else:  # 如果使用标准原点
            self.root_states[env_ids] = self.base_init_state  # 初始化为基础初始状态
            self.root_states[env_ids, :3] += self.env_origins[env_ids]  # 添加环境原点
        # 基础速度
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device)  # [7:10]: 线速度, [10:13]: 角速度
        env_ids_int32 = env_ids.to(dtype=torch.int32)  # 转换为int32类型
        # 设置选定环境的根状态
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _push_robots(self):
        """ 随机推动机器人。通过设置随机化的基础速度来模拟冲量。
        """
        env_ids = torch.arange(self.num_envs, device=self.device)  # 所有环境ID
        # 根据推动间隔选择需要推动的环境
        push_env_ids = env_ids[self.episode_length_buf[env_ids] % int(self.cfg.domain_rand.push_interval) == 0]
        if len(push_env_ids) == 0:  # 如果没有环境需要推动
            return
        max_vel = self.cfg.domain_rand.max_push_vel_xy  # 获取最大推动速度
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device)  # 设置xy方向线速度
        
        env_ids_int32 = push_env_ids.to(dtype=torch.int32)  # 转换为int32类型
        # 更新选定环境的根状态
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                    gymtorch.unwrap_tensor(self.root_states),
                                                    gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

   
    
    def update_command_curriculum(self, env_ids):
        """ 实现增加命令的课程学习

        参数:
            env_ids (List[int]): 正在重置的环境ID
        """
        # 如果跟踪奖励超过最大值的80%，则增加命令范围
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
            # 降低x方向线速度命令下限
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
            # 增加x方向线速度命令上限
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)


    def _get_noise_scale_vec(self, cfg):
        """ 设置用于缩放添加到观测中的噪声的向量。
            [注意]：当更改观测结构时必须调整

        参数:
            cfg (Dict): 环境配置文件

        返回:
            [torch.Tensor]: 用于乘以[-1, 1]均匀分布的缩放向量
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])  # 创建与观测相同形状的零向量
        self.add_noise = self.cfg.noise.add_noise  # 获取是否添加噪声的标志
        noise_scales = self.cfg.noise.noise_scales  # 获取噪声缩放参数
        noise_level = self.cfg.noise.noise_level  # 获取噪声水平
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel  # 线速度噪声
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel  # 角速度噪声
        noise_vec[6:9] = noise_scales.gravity * noise_level  # 重力噪声
        noise_vec[9:12] = 0.  # 命令噪声（当前为零）
        noise_vec[12:12+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos  # 关节位置噪声
        noise_vec[12+self.num_actions:12+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel  # 关节速度噪声
        noise_vec[12+2*self.num_actions:12+3*self.num_actions] = 0.  # 先前动作噪声（当前为零）

        return noise_vec

    #----------------------------------------
    def _init_buffers(self):
        """ 初始化将包含模拟状态和处理量的PyTorch张量
        """
        # 获取gym GPU状态张量
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)  # 获取演员根状态张量
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)  # 获取自由度状态张量
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)  # 获取净接触力张量
        self.gym.refresh_dof_state_tensor(self.sim)  # 刷新自由度状态张量
        self.gym.refresh_actor_root_state_tensor(self.sim)  # 刷新演员根状态张量
        self.gym.refresh_net_contact_force_tensor(self.sim)  # 刷新净接触力张量

        # 为不同切片创建包装张量
        self.root_states = gymtorch.wrap_tensor(actor_root_state)  # 包装根状态张量
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)  # 包装自由度状态张量
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]  # 提取关节位置
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]  # 提取关节速度
        self.base_quat = self.root_states[:, 3:7]  # 提取基础四元数
        self.rpy = get_euler_xyz_in_tensor(self.base_quat)  # 计算欧拉角
        self.base_pos = self.root_states[:self.num_envs, 0:3]  # 提取基础位置
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)  # 形状：num_envs, num_bodies, xyz轴

        # 初始化稍后使用的数据
        self.common_step_counter = 0  # 通用步数计数器
        self.extras = {}  # 额外信息字典
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)  # 获取噪声缩放向量
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))  # 重力向量
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))  # 前进方向向量
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)  # 扭矩缓冲区
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)  # P增益
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)  # D增益
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)  # 动作缓冲区
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)  # 上一动作缓冲区
        self.last_dof_vel = torch.zeros_like(self.dof_vel)  # 上一关节速度缓冲区
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])  # 上一根速度缓冲区
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False)  # 命令缓冲区
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,)  # 命令缩放
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)  # 脚部空中时间
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)  # 上一接触状态
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])  # 基础线速度（基坐标系下）
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])  # 基础角速度（基坐标系下）
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)  # 投影重力
      

        # 关节位置偏移和PD增益
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)  # 默认关节位置
        for i in range(self.num_dofs):  # 遍历所有自由度
            name = self.dof_names[i]  # 获取关节名称
            angle = self.cfg.init_state.default_joint_angles[name]  # 获取默认关节角度
            self.default_dof_pos[i] = angle  # 设置默认关节位置
            found = False  # 查找标志
            for dof_name in self.cfg.control.stiffness.keys():  # 遍历配置中的刚度键
                if dof_name in name:  # 如果关节名称包含配置键
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]  # 设置P增益
                    self.d_gains[i] = self.cfg.control.damping[dof_name]  # 设置D增益
                    found = True  # 设置找到标志
            if not found:  # 如果未找到对应配置
                self.p_gains[i] = 0.  # 设置P增益为零
                self.d_gains[i] = 0.  # 设置D增益为零
                if self.cfg.control.control_type in ["P", "V"]:  # 如果是P或V控制类型
                    print(f"PD gain of joint {name} were not defined, setting them to zero")  # 打印警告
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)  # 增加批次维度

    def _prepare_reward_function(self):
        """ 准备奖励函数列表，这些函数将被调用来计算总奖励。
            查找self._reward_<REWARD_NAME>，其中<REWARD_NAME>是配置中所有非零奖励尺度的名称。
        """
        # 移除零尺度 + 将非零尺度乘以dt
        for key in list(self.reward_scales.keys()):  # 遍历所有奖励尺度
            scale = self.reward_scales[key]  # 获取尺度值
            if scale==0:  # 如果尺度为零
                self.reward_scales.pop(key)  # 从字典中移除
            else:  # 如果尺度非零
                self.reward_scales[key] *= self.dt  # 乘以时间步长
        # 准备函数列表
        self.reward_functions = []  # 奖励函数列表
        self.reward_names = []  # 奖励名称列表
        for name, scale in self.reward_scales.items():  # 遍历奖励尺度字典
            if name=="termination":  # 跳过终止奖励
                continue
            self.reward_names.append(name)  # 添加奖励名称
            name = '_reward_' + name  # 构造奖励函数名
            self.reward_functions.append(getattr(self, name))  # 获取奖励函数并添加到列表

        # 奖励回合总和
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}  # 为每个奖励项初始化回合总和

    def _create_ground_plane(self):
        """ 向模拟中添加地平面，基于配置设置摩擦和恢复系数。
        """
        plane_params = gymapi.PlaneParams()  # 创建地平面参数
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)  # 设置法线方向为z轴向上
        plane_params.static_friction = self.cfg.terrain.static_friction  # 设置静摩擦系数
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction  # 设置动摩擦系数
        plane_params.restitution = self.cfg.terrain.restitution  # 设置恢复系数
        self.gym.add_ground(self.sim, plane_params)  # 向模拟中添加地平面

    def _create_envs(self):
        """ 创建环境：
             1. 加载机器人URDF/MJCF资产，
             2. 对于每个环境
                2.1 创建环境， 
                2.2 调用自由度和刚体形状属性回调，
                2.3 使用这些属性创建演员并添加到环境
             3. 存储机器人不同身体的索引
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)  # 格式化资产路径
        asset_root = os.path.dirname(asset_path)  # 获取资产根目录
        asset_file = os.path.basename(asset_path)  # 获取资产文件名

        asset_options = gymapi.AssetOptions()  # 创建资产选项
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode  # 设置默认自由度驱动模式
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints  # 设置是否折叠固定关节
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule  # 设置是否用胶囊体替换圆柱体
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments  # 设置是否翻转视觉附件
        asset_options.fix_base_link = self.cfg.asset.fix_base_link  # 设置是否固定基础连杆
        asset_options.density = self.cfg.asset.density  # 设置密度
        asset_options.angular_damping = self.cfg.asset.angular_damping  # 设置角阻尼
        asset_options.linear_damping = self.cfg.asset.linear_damping  # 设置线阻尼
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity  # 设置最大角速度
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity  # 设置最大线速度
        asset_options.armature = self.cfg.asset.armature  # 设置电枢
        asset_options.thickness = self.cfg.asset.thickness  # 设置厚度
        asset_options.disable_gravity = self.cfg.asset.disable_gravity  # 设置是否禁用重力

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)  # 加载机器人资产
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)  # 获取资产自由度数量
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)  # 获取资产刚体数量
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)  # 获取资产自由度属性
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)  # 获取资产刚体形状属性

        # 从资产保存身体名称
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)  # 获取刚体名称列表
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)  # 获取自由度名称列表
        self.num_bodies = len(body_names)  # 计算刚体数量
        self.num_dofs = len(self.dof_names)  # 计算自由度数量
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]  # 筛选脚部刚体名称
        penalized_contact_names = []  # 惩罚接触名称列表
        for name in self.cfg.asset.penalize_contacts_on:  # 遍历配置中的惩罚接触名称
            penalized_contact_names.extend([s for s in body_names if name in s])  # 筛选并添加到列表
        termination_contact_names = []  # 终止接触名称列表
        for name in self.cfg.asset.terminate_after_contacts_on:  # 遍历配置中的终止接触名称
            termination_contact_names.extend([s for s in body_names if name in s])  # 筛选并添加到列表

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel  # 构造基础初始状态列表
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)  # 转换为张量
        start_pose = gymapi.Transform()  # 创建起始位姿
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])  # 设置起始位置

        self._get_env_origins()  # 获取环境原点
        env_lower = gymapi.Vec3(0., 0., 0.)  # 环境下界
        env_upper = gymapi.Vec3(0., 0., 0.)  # 环境上界
        self.actor_handles = []  # 演员句柄列表
        self.envs = []  # 环境句柄列表
        for i in range(self.num_envs):  # 遍历所有环境
            # 创建环境实例
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))  # 创建环境
            pos = self.env_origins[i].clone()  # 克隆环境原点
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)  # 在原点附近随机化xy位置
            start_pose.p = gymapi.Vec3(*pos)  # 设置起始位姿
                
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)  # 处理刚体形状属性
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)  # 设置资产刚体形状属性
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)  # 创建演员
            dof_props = self._process_dof_props(dof_props_asset, i)  # 处理自由度属性
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)  # 设置演员自由度属性
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)  # 获取演员刚体属性
            body_props = self._process_rigid_body_props(body_props, i)  # 处理刚体属性
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)  # 设置演员刚体属性
            self.envs.append(env_handle)  # 添加环境句柄到列表
            self.actor_handles.append(actor_handle)  # 添加演员句柄到列表

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)  # 初始化脚部索引
        for i in range(len(feet_names)):  # 遍历脚部名称
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])  # 查找脚部刚体句柄

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)  # 初始化惩罚接触索引
        for i in range(len(penalized_contact_names)):  # 遍历惩罚接触名称
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])  # 查找惩罚接触刚体句柄

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)  # 初始化终止接触索引
        for i in range(len(termination_contact_names)):  # 遍历终止接触名称
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])  # 查找终止接触刚体句柄

    def _get_env_origins(self):
        """ 设置环境原点。在粗糙地形上，原点由地形平台定义。
            否则创建网格。
        """
      
        self.custom_origins = False  # 自定义原点标志，默认为False
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)  # 初始化环境原点
        # 创建机器人网格
        num_cols = np.floor(np.sqrt(self.num_envs))  # 计算列数
        num_rows = np.ceil(self.num_envs / num_cols)  # 计算行数
        xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))  # 创建网格坐标
        spacing = self.cfg.env.env_spacing  # 获取环境间距
        self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]  # 设置x坐标
        self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]  # 设置y坐标
        self.env_origins[:, 2] = 0.  # 设置z坐标为零

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt  # 计算控制时间步长
        self.obs_scales = self.cfg.normalization.obs_scales  # 获取观测缩放参数
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)  # 将奖励尺度类转换为字典
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)  # 将命令范围类转换为字典
     

        self.max_episode_length_s = self.cfg.env.episode_length_s  # 获取最大回合长度（秒）
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)  # 计算最大回合步数

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)  # 计算推动间隔步数


    #------------ 奖励函数 ----------------
    def _reward_lin_vel_z(self):
        # 惩罚z轴基础线速度
        return torch.square(self.base_lin_vel[:, 2])  # 返回z方向线速度的平方
    
    def _reward_ang_vel_xy(self):
        # 惩罚xy轴基础角速度
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)  # 返回xy方向角速度平方和
    
    def _reward_orientation(self):
        # 惩罚非平坦的基础方向
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)  # 返回投影重力xy分量的平方和

    def _reward_base_height(self):
        # 惩罚基础高度偏离目标
        base_height = self.root_states[:, 2]  # 获取基础高度
        return torch.square(base_height - self.cfg.rewards.base_height_target)  # 返回高度差的平方
    
    def _reward_torques(self):
        # 惩罚扭矩
        return torch.sum(torch.square(self.torques), dim=1)  # 返回扭矩平方和

    def _reward_dof_vel(self):
        # 惩罚关节速度
        return torch.sum(torch.square(self.dof_vel), dim=1)  # 返回关节速度平方和
    
    def _reward_dof_acc(self):
        # 惩罚关节加速度
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)  # 返回关节加速度平方和
    
    def _reward_action_rate(self):
        # 惩罚动作变化
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)  # 返回动作变化平方和
    
    def _reward_collision(self):
        # 惩罚选定身体上的碰撞
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)  # 返回接触力大于阈值的计数
    
    def _reward_termination(self):
        # 终止奖励/惩罚
        return self.reset_buf * ~self.time_out_buf  # 仅对非超时的终止返回奖励
    
    def _reward_dof_pos_limits(self):
        # 惩罚关节位置太接近限制
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.)  # 低于下限的部分
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)  # 高于上限的部分
        return torch.sum(out_of_limits, dim=1)  # 返回超出限制的总和

    def _reward_dof_vel_limits(self):
        # 惩罚关节速度太接近限制
        # 裁剪到最大误差 = 每个关节1 rad/s，避免巨大惩罚
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)  # 返回超出软限制的部分

    def _reward_torque_limits(self):
        # 惩罚扭矩太接近限制
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)  # 返回超出软限制的部分

    def _reward_tracking_lin_vel(self):
        # 线速度命令跟踪（xy轴）
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)  # 计算线速度误差
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)  # 返回指数衰减的跟踪奖励
    
    def _reward_tracking_ang_vel(self):
        # 角速度命令跟踪（偏航）
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])  # 计算角速度误差
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)  # 返回指数衰减的跟踪奖励

    def _reward_feet_air_time(self):
        # 奖励长步态
        # 需要过滤接触，因为PhysX在网格上的接触报告不可靠
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.  # 检测脚部是否有垂直接触力
        contact_filt = torch.logical_or(contact, self.last_contacts)  # 与上一时刻接触状态进行或滤波
        self.last_contacts = contact  # 更新上一时刻接触状态
        first_contact = (self.feet_air_time > 0.) * contact_filt  # 检测是否首次接触地面
        self.feet_air_time += self.dt  # 增加脚部空中时间
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1)  # 仅对首次接触地面进行奖励
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1  # 零命令时无奖励
        self.feet_air_time *= ~contact_filt  # 如果接触则重置空中时间
        return rew_airTime  # 返回空中时间奖励
    
    def _reward_stumble(self):
        # 惩罚脚部撞击垂直表面
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)  # 返回水平接触力是否大于5倍垂直接触力
        
    def _reward_stand_still(self):
        # 惩罚零命令时的运动
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)  # 返回关节位置偏移与零命令的乘积

    def _reward_feet_contact_forces(self):
        # 惩罚高接触力
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)  # 返回超出最大接触力的部分