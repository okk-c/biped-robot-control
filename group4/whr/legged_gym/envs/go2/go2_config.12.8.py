from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class GO2RoughCfg(LeggedRobotCfg):
    # 在 GO2RoughCfg 中添加/修改:
    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'trimesh'       # 改为 trimesh
        horizontal_scale = 0.1
        vertical_scale = 0.005
        border_size = 25
        curriculum = True
        selected = False
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 10
        num_cols = 20
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        slope_treshold = 0.75
#*******************the sentence above is me myself added , cancellable***********************
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_actions = 12
        num_observations = 48
        episode_length_s = 20
        test = False

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.42]
        default_joint_angles = {
            'FL_hip_joint': 0.1,
            'RL_hip_joint': 0.1,
            'FR_hip_joint': -0.1,
            'RR_hip_joint': -0.1,
            'FL_thigh_joint': 0.8,
            'RL_thigh_joint': 1.0,
            'FR_thigh_joint': 0.8,
            'RR_thigh_joint': 1.0,
            'FL_calf_joint': -1.5,
            'RL_calf_joint': -1.5,
            'FR_calf_joint': -1.5,
            'RR_calf_joint': -1.5,
        }

    class commands(LeggedRobotCfg.commands):
        curriculum = True              # ← 必须打开
        heading_command = False        # ← 我们用 yaw，而不是 heading
        resampling_time = 10.0
        num_commands = 4
        class ranges:
            lin_vel_x = [-1.0, 1.0]    # ← 走路前进速度
            lin_vel_y = [-0.5, 0.5]    # ← 左右横向速度
            ang_vel_yaw = [-1.5, 1.5]  # ← 旋转速度
            heading = [-3.14, 3.14]

    class control(LeggedRobotCfg.control):
        control_type = 'P'
        stiffness = {'joint': 20.}
        damping = {'joint': 0.5}
        action_scale = 0.25
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
        name = "go2"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25
        class scales(LeggedRobotCfg.rewards.scales):
            torques = -0.0002
            dof_pos_limits = -10.0

class GO2RoughCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
    class runner(LeggedRobotCfgPPO.runner):
        run_name = ""
        experiment_name = "rough_go2"
