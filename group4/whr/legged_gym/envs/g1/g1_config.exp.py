# 从基础配置文件模块中导入基础类和PPO配置类
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

# 定义G1机器人地形适应配置类，继承自基础机器人配置类
class G1RoughCfg( LeggedRobotCfg ):
    # 初始化状态配置类，继承自基础配置的初始化状态类
    class init_state( LeggedRobotCfg.init_state ):
        # 初始位置坐标 [x, y, z]，单位：米
        pos = [3.0, 3.0, 0.8] # x,y,z [m]
        # 默认关节角度（当动作为0.0时的目标角度），单位：弧度
        default_joint_angles = { 
           'left_hip_yaw_joint' : 0. ,      # 左髋关节偏航角
           'left_hip_roll_joint' : 0,       # 左髋关节滚转角              
           'left_hip_pitch_joint' : -0.1,   # 左髋关节俯仰角       
           'left_knee_joint' : 0.3,         # 左膝关节角度
           'left_ankle_pitch_joint' : -0.2, # 左踝关节俯仰角   
           'left_ankle_roll_joint' : 0,     # 左踝关节滚转角
           'right_hip_yaw_joint' : 0.,      # 右髋关节偏航角
           'right_hip_roll_joint' : 0,      # 右髋关节滚转角
           'right_hip_pitch_joint' : -0.1,  # 右髋关节俯仰角                                   
           'right_knee_joint' : 0.3,        # 右膝关节角度                                           
           'right_ankle_pitch_joint': -0.2, # 右踝关节俯仰角                          
           'right_ankle_roll_joint' : 0,    # 右踝关节滚转角
           'torso_joint' : 0.               # 躯干关节角度
        }
    
    # 环境配置类，继承自基础环境配置
    class env(LeggedRobotCfg.env):
        num_observations = 47       # 观测空间的维度
        num_privileged_obs = 50     # 特权观测的维度（用于教师网络）
        num_actions = 12            # 动作空间的维度

    # 域随机化配置类，继承自基础域随机化配置
    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True          # 是否随机化地面摩擦系数
        friction_range = [0.1, 1.25]       # 摩擦系数随机范围
        randomize_base_mass = True         # 是否随机化基座质量
        added_mass_range = [-1., 3.]       # 质量添加的随机范围
        push_robots = True                 # 是否对机器人施加随机推力
        push_interval_s = 5                # 推力施加的时间间隔（秒）
        max_push_vel_xy = 1.5              # 最大推力速度（xy平面）

    # 控制配置类，继承自基础控制配置
    class control( LeggedRobotCfg.control ):
        control_type = 'P'  # 控制类型：P表示位置控制（PD控制器）
        # PD控制器的刚度参数，单位：N*m/rad
        stiffness = {
            'hip_yaw': 100,     # 髋关节偏航刚度
            'hip_roll': 100,    # 髋关节滚转刚度
            'hip_pitch': 100,   # 髋关节俯仰刚度
            'knee': 150,        # 膝关节刚度
            'ankle': 40,        # 踝关节刚度
        }
        # PD控制器的阻尼参数，单位：N*m*s/rad
        damping = {
            'hip_yaw': 2,       # 髋关节偏航阻尼
            'hip_roll': 2,      # 髋关节滚转阻尼
            'hip_pitch': 2,     # 髋关节俯仰阻尼
            'knee': 4,          # 膝关节阻尼
            'ankle': 2,         # 踝关节阻尼
        }
        action_scale = 0.25     # 动作缩放因子：目标角度 = actionScale * action + defaultAngle
        decimation = 4          # 降采样因子：每个策略时间步长内的控制动作更新次数

    # 资产（机器人模型）配置类，继承自基础资产配置
    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1_description/g1_12dof.urdf'  # URDF模型文件路径
        name = "g1"                         # 机器人名称
        foot_name = "ankle_roll"            # 足端关节名称
        penalize_contacts_on = ["hip", "knee"]  # 需要惩罚接触的部件列表
        terminate_after_contacts_on = ["pelvis"] # 接触后终止episode的部件列表
        self_collisions = 0                 # 自碰撞设置：0启用，1禁用
        flip_visual_attachments = False     # 是否翻转视觉附件

    # 奖励函数配置类，继承自基础奖励配置
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9        # 关节位置软限制阈值
        base_height_target = 0.78       # 基座目标高度
        
        # 奖励缩放系数配置类，继承自基础奖励缩放系数
        class scales( LeggedRobotCfg.rewards.scales ):
            tracking_lin_vel = 1.0      # 线速度跟踪奖励系数
            tracking_ang_vel = 0.5      # 角速度跟踪奖励系数
            lin_vel_z = -2.0            # Z轴线速度惩罚系数（负值表示惩罚）
            ang_vel_xy = -0.05          # XY平面角速度惩罚系数
            orientation = -1.0          # 姿态惩罚系数
            base_height = -10.0         # 基座高度惩罚系数
            dof_acc = -2.5e-7           # 关节加速度惩罚系数
            dof_vel = -1e-3             # 关节速度惩罚系数
            feet_air_time = 0.0         # 足端空中时间奖励系数（设为0表示不使用）
            collision = 0.0             # 碰撞惩罚系数（设为0表示不使用）
            action_rate = -0.01         # 动作变化率惩罚系数
            dof_pos_limits = -5.0       # 关节位置限制惩罚系数
            alive = 0.15                # 存活奖励系数
            hip_pos = -1.0              # 髋关节位置惩罚系数
            contact_no_vel = -0.2       # 无速度接触惩罚系数
            feet_swing_height = -20.0   # 足端摆动高度惩罚系数
            contact = 0.18              # 接触奖励系数

# PPO算法配置类，继承自基础PPO配置
class G1RoughCfgPPO( LeggedRobotCfgPPO ):
    # 策略网络配置类
    class policy:
        init_noise_std = 0.8               # 初始动作噪声标准差
        actor_hidden_dims = [32]           # 演员网络隐藏层维度
        critic_hidden_dims = [32]          # 评论家网络隐藏层维度
        activation = 'elu'                 # 激活函数类型：elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # 以下仅用于循环神经网络策略：
        rnn_type = 'lstm'                  # RNN类型：lstm
        rnn_hidden_size = 64               # RNN隐藏层大小
        rnn_num_layers = 1                 # RNN层数
        
    # 算法参数配置类，继承自基础PPO算法配置
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01                # 熵系数，用于鼓励探索
    
    # 训练运行器配置类，继承自基础PPO运行器配置
    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = "ActorCriticRecurrent"  # 策略类名称（循环神经网络版本）
        max_iterations = 10000                      # 最大训练迭代次数
        run_name = ''                               # 运行名称（通常为空，自动生成）
        experiment_name = 'g1'                      # 实验名称