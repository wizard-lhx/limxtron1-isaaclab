import math
from dataclasses import MISSING

from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import DomeLightCfg, MdlFileCfg, RigidBodyMaterialCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as GaussianNoise
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as UniformNoise
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import CommandsCfg as BaseCommandsCfg

from bipedal_locomotion.tasks.locomotion import mdp

##################
# 场景定义 / Scene Definition
##################


@configclass
class PFSceneCfg(InteractiveSceneCfg):
    """测试场景配置类 / Configuration for the test scene"""

    # 地形配置 / Terrain configuration
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",      # 地形在场景中的路径 / Terrain path in scene
        terrain_type="plane",           # 地形类型：平面 / Terrain type: plane
        terrain_generator=None,         # 不使用地形生成器 / No terrain generator used
        max_init_terrain_level=0,       # 最大初始地形难度等级 / Maximum initial terrain difficulty level
        collision_group=-1,             # 碰撞组ID / Collision group ID

        # 物理材质属性 / Physics material properties
        physics_material=RigidBodyMaterialCfg(
            friction_combine_mode="multiply",    # 摩擦力结合模式：乘法 / Friction combine mode: multiply
            restitution_combine_mode="multiply", # 恢复系数结合模式：乘法 / Restitution combine mode: multiply
            static_friction=1.0,                # 静摩擦系数 / Static friction coefficient
            dynamic_friction=1.0,               # 动摩擦系数 / Dynamic friction coefficient
            restitution=1.0,                    # 恢复系数 / Restitution coefficient
        ),

        # 视觉材质配置 / Visual material configuration
        visual_material=MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/"
            + "TilesMarbleSpiderWhiteBrickBondHoned.mdl",  # 大理石纹理材质路径 / Marble texture material path
            project_uvw=True,              # 启用UV投影 / Enable UV projection
            texture_scale=(0.25, 0.25),    # 纹理缩放比例 / Texture scaling factor
        ),
        debug_vis=False,   # 不显示调试可视化 / Don't show debug visualization
    )

    # 天空光照配置 / Sky lighting configuration
    light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=DomeLightCfg(
            intensity=750.0,
            color=(0.9, 0.9, 0.9),
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # pointfoot robot
    robot: ArticulationCfg = MISSING

    # 高度扫描传感器 (将在子类中定义) / Height scanner sensor (to be defined in subclasses)
    height_scanner: RayCasterCfg = MISSING

    # 接触力传感器配置 / Contact force sensor configuration
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",  # 传感器安装路径 / Sensor attachment path
        history_length=4,                     # 历史数据长度 / History data length
        track_air_time=True,                  # 跟踪空中时间 / Track air time
        update_period=0.0,                    # 更新周期 (0表示每帧更新) / Update period (0 means every frame)
    )


##############
# MDP设置 / MDP Settings
##############


@configclass
class CommandCfg(BaseCommandsCfg):
    # 步态命令配置 / Gait command configuration
    gait_command = mdp.UniformGaitCommandCfg(
        resampling_time_range=(5.0, 5.0),  # 命令重采样时间范围 (固定5秒) / Command resampling time range (fixed 5s)
        debug_vis=False,                    # 不显示调试可视化 / No debug visualization
        ranges=mdp.UniformGaitCommandCfg.Ranges(
            frequencies=(1.5, 2.5),     # 步态频率范围 [Hz] / Gait frequency range [Hz]
            offsets=(0.5, 0.5),         # 相位偏移范围 [0-1] / Phase offset range [0-1]
            durations=(0.5, 0.5),       # 接触持续时间范围 [0-1] / Contact duration range [0-1]
            swing_height=(0.1, 0.3)     # 摆动高度范围 [m] / Swing height range [m]
        ),
    )

    """后初始化配置 / Post-initialization configuration"""
    def __post_init__(self):
        self.base_velocity.asset_name = "robot"          # 关联的机器人资产名称 / Associated robot asset name
        self.base_velocity.heading_command = True        # 启用航向命令 / Enable heading commands
        self.base_velocity.debug_vis = True              # 启用调试可视化 / Enable debug visualization
        self.base_velocity.heading_control_stiffness = 1.0  # 航向控制刚度 / Heading control stiffness
        self.base_velocity.resampling_time_range = (0.0, 5.0)  # 速度命令重采样时间 / Velocity command resampling time
        self.base_velocity.rel_standing_envs = 0.2       # 站立环境比例 / Standing environments ratio
        self.base_velocity.rel_heading_envs = 0.0        # 航向环境比例 / Heading environments ratio
        # 速度命令范围设置 / Velocity command ranges
        self.base_velocity.ranges = mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.5, 1.5),      # 前进速度范围 [m/s] / Forward velocity range [m/s]
            lin_vel_y=(-1.0, 1.0),      # 横向速度范围 [m/s] / Lateral velocity range [m/s]
            ang_vel_z=(-0.5, 0.5),      # 转向角速度范围 [rad/s] / Turning angular velocity range [rad/s]
            heading=(-math.pi, math.pi)  # 航向角范围 [rad] / Heading angle range [rad]
        )


@configclass
class ActionsCfg:
    """动作规范配置类 / Action specifications configuration class"""

    # 关节位置动作配置 / Joint position action configuration
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",                  # 目标资产名称 / Target asset name
        # 控制的关节名称列表 / List of controlled joint names
        joint_names=["abad_L_Joint", "abad_R_Joint", "hip_L_Joint", 
                    "hip_R_Joint", "knee_L_Joint", "knee_R_Joint"],
        scale=0.25,              # 动作缩放因子 / Action scaling factor
        use_default_offset=True, # 使用默认偏移量 / Use default offset
    )


@configclass
class ObservarionsCfg:
    """观测规范配置类 / Observation specifications configuration class"""

    @configclass
    class PolicyCfg(ObsGroup):
        """策略网络观测组配置 / Policy network observation group configuration"""

        # 机器人基座测量 / Robot base measurements
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,              # 基座角速度函数 / Base angular velocity function
            noise=GaussianNoise(mean=0.0, std=0.05),  # 高斯噪声 / Gaussian noise
            clip=(-100.0, 100.0),               # 数值裁剪范围 / Value clipping range
            scale=0.25,                         # 缩放因子 / Scaling factor
        )
        proj_gravity = ObsTerm(
            func=mdp.projected_gravity,         # 投影重力函数 / Projected gravity function
            noise=GaussianNoise(mean=0.0, std=0.025),  # 噪声配置 / Noise configuration
            clip=(-100.0, 100.0),               # 裁剪范围 / Clipping range
            scale=1.0,                          # 缩放因子 / Scaling factor
        )

        # 机器人关节测量 / Robot joint measurements
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,            # 关节位置函数 / Joint position function
            noise=GaussianNoise(mean=0.0, std=0.01),  # 噪声配置 / Noise configuration
            clip=(-100.0, 100.0),               # 裁剪范围 / Clipping range
            scale=1.0,                          # 缩放因子 / Scaling factor
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel,                 # 关节速度函数 / Joint velocity function
            noise=GaussianNoise(mean=0.0, std=0.01),  # 噪声配置 / Noise configuration
            clip=(-100.0, 100.0),               # 裁剪范围 / Clipping range
            scale=0.05,                         # 缩放因子 / Scaling factor
        )

        # 上一步动作 / Last action
        last_action = ObsTerm(func=mdp.last_action)

        # 步态相关观测 / Gait-related observations
        gait_phase = ObsTerm(func=mdp.get_gait_phase)  # 步态相位 / Gait phase
        gait_command = ObsTerm(
            func=mdp.get_gait_command, 
            params={"command_name": "gait_command"}  # 步态命令 / Gait command
        )
        
        def __post_init__(self):
            self.enable_corruption = True      # 启用观测损坏 / Enable observation corruption
            self.concatenate_terms = True      # 连接所有观测项 / Concatenate all observation terms
    
    @configclass
    class HistoryObsCfg(ObsGroup):
        """历史观测组配置 - 用于存储观测历史 / History observation group - for storing observation history"""

        # robot base measurements
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=GaussianNoise(mean=0.0, std=0.05),clip=(-100.0, 100.0),scale=0.25,)
        proj_gravity = ObsTerm(func=mdp.projected_gravity, noise=GaussianNoise(mean=0.0, std=0.025),clip=(-100.0, 100.0),scale=1.0,)

        # robot joint measurements
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=GaussianNoise(mean=0.0, std=0.01),clip=(-100.0, 100.0),scale=1.0,)
        joint_vel = ObsTerm(func=mdp.joint_vel, noise=GaussianNoise(mean=0.0, std=0.01),clip=(-100.0, 100.0),scale=0.05,)

        # last action
        last_action = ObsTerm(func=mdp.last_action)

        # gaits
        gait_phase = ObsTerm(func=mdp.get_gait_phase)
        gait_command = ObsTerm(func=mdp.get_gait_command, params={"command_name": "gait_command"})
        
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 10          # 历史长度为10步 / History length of 10 steps
            self.flatten_history_dim = False  # 不展平历史维度 / Don't flatten history dimension

    @configclass
    class CriticCfg(ObsGroup):
        """评价网络观测组配置 - 包含特权信息 / Critic network observation group - includes privileged information"""

        # 策略观测 (与智能体相同) / Policy observations (same as agent)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        proj_gravity = ObsTerm(func=mdp.projected_gravity)

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel)

        last_action = ObsTerm(func=mdp.last_action)

        gait_phase = ObsTerm(func=mdp.get_gait_phase)
        gait_command = ObsTerm(func=mdp.get_gait_command, params={"command_name": "gait_command"})

        heights = ObsTerm(func=mdp.height_scan,params={"sensor_cfg": SceneEntityCfg("height_scanner")})
        
        # 特权观测 (仅评价网络可见) / Privileged observations (only visible to critic)
        robot_joint_torque = ObsTerm(func=mdp.robot_joint_torque)    # 关节力矩 / Joint torques
        robot_joint_acc = ObsTerm(func=mdp.robot_joint_acc)          # 关节加速度 / Joint accelerations
        robot_feet_contact_force = ObsTerm(                          # 足部接触力 / Foot contact forces
            func=mdp.robot_feet_contact_force,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot_[LR]_Link"),
            },
        )
        robot_mass = ObsTerm(func=mdp.robot_mass)                    # 机器人质量 / Robot mass
        robot_inertia = ObsTerm(func=mdp.robot_inertia)              # 机器人惯量 / Robot inertia
        robot_joint_stiffness = ObsTerm(func=mdp.robot_joint_stiffness)  # 关节刚度 / Joint stiffness
        robot_joint_damping = ObsTerm(func=mdp.robot_joint_damping)      # 关节阻尼 / Joint damping
        robot_pos = ObsTerm(func=mdp.robot_pos)                      # 机器人位置 / Robot position
        robot_vel = ObsTerm(func=mdp.robot_vel)                      # 机器人速度 / Robot velocity
        robot_material_propertirs = ObsTerm(func=mdp.robot_material_properties)  # 材质属性 / Material properties
        robot_base_pose = ObsTerm(func=mdp.robot_base_pose)          # 基座姿态 / Base pose

        def __post_init__(self):
            self.enable_corruption = False     # 不对特权信息添加噪声 / No noise for privileged information
            self.concatenate_terms = True      # 连接所有观测项 / Concatenate all terms

    @configclass
    class CommandsObsCfg(ObsGroup):
        """命令观测配置 / Commands observation configuration"""
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, 
            params={"command_name": "base_velocity"}  # 速度命令 / Velocity commands
        )
    
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    commands: CommandsObsCfg = CommandsObsCfg()
    obsHistory: HistoryObsCfg = HistoryObsCfg()


@configclass
class EventsCfg:
    """事件配置类 - 定义训练过程中的随机化事件 / Events configuration class - defines randomization events during training"""
    # 即域随机化 / i.e. domain randomization

    # 启动时事件 / Startup events
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,     # 随机化刚体质量函数 / Randomize rigid body mass function
        mode="startup",                         # 启动模式 / Startup mode
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_Link"),  # 目标：机器人基座 / Target: robot base
            "mass_distribution_params": (-1.0, 3.0),  # 质量分布参数 [kg] / Mass distribution parameters [kg]
            "operation": "add",                 # 操作类型：添加 / Operation type: add
        },
        is_global_time=False,                   # 不使用全局时间 / Don't use global time
        min_step_count_between_reset=0,         # 重置间最小步数 / Min steps between resets
    )

    add_link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,     # 随机化连杆质量 / Randomize link mass
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_[LR]_Link"),  # 所有左右连杆 / All left-right links
            "mass_distribution_params": (0.8, 1.2),  # 质量缩放范围 / Mass scaling range
            "operation": "scale",               # 操作类型：缩放 / Operation type: scale
        },
        is_global_time=False,
        min_step_count_between_reset=0,
    )
    
    radomize_rigid_body_mass_inertia = EventTerm(
        func=mdp.randomize_rigid_body_mass_inertia,  # 随机化质量和惯量 / Randomize mass and inertia
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mass_inertia_distribution_params": (0.8, 1.2),  # 质量惯量分布 / Mass inertia distribution
            "operation": "scale",
        },
    )
    
    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,     # 随机化物理材质 / Randomize physics material
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.4, 1.2),   # 静摩擦系数范围 / Static friction range
            "dynamic_friction_range": (0.7, 0.9),  # 动摩擦系数范围 / Dynamic friction range
            "restitution_range": (0.0, 1.0),       # 恢复系数范围 / Restitution range
            "num_buckets": 48,                      # 离散化桶数 / Discretization buckets
        },
        is_global_time=False,
        min_step_count_between_reset=0,
    )

    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,          # 随机化执行器增益 / Randomize actuator gains
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (32, 48),   # 刚度分布 / Stiffness distribution
            "damping_distribution_params": (2.0, 3.0),   # 阻尼分布 / Damping distribution
            "operation": "abs",                     # 取绝对值操作 / Absolute value operation
            "distribution": "uniform",              # 均匀分布 / Uniform distribution
        },
        is_global_time=False,
        min_step_count_between_reset=0,
    )

    robot_center_of_mass = EventTerm(
        func=mdp.randomize_rigid_body_coms,         # 随机化重心位置 / Randomize center of mass
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            # 重心偏移范围 (x, y, z) [m] / Center of mass offset range (x, y, z) [m]
            "com_distribution_params": ((-0.075, 0.075), (-0.05, 0.06), (-0.05, 0.05)),
            "operation": "add",
            "distribution": "uniform",
        },
    )

    # 重置时事件 / Reset events
    reset_robot_base = EventTerm(
        func=mdp.reset_root_state_uniform,          # 均匀重置根状态 / Uniform reset root state
        mode="reset",                               # 重置模式 / Reset mode
        params={
            # 姿态范围 / Pose range
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            # 速度范围 / Velocity range
            "velocity_range": {
                "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5), "pitch": (-0.5, 0.5), "yaw": (-0.5, 0.5),
            },
        },
        is_global_time=False,
        min_step_count_between_reset=0,
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,             # 按比例重置关节 / Reset joints by scale
        mode="reset",
        params={
            "position_range": (-0.5, 0.5),         # 位置扰动范围 / Position perturbation range
            "velocity_range": (0.0, 0.0),          # 速度范围 (重置为0) / Velocity range (reset to 0)
        },
        is_global_time=False,
        min_step_count_between_reset=0,
    )

    # 间隔事件 / Interval events
    push_robot = EventTerm(
        func=mdp.apply_external_force_torque_stochastic,  # 随机外力扰动 / Stochastic external force disturbance
        mode="interval",                            # 间隔模式 / Interval mode
        interval_range_s=(0.0, 0.0),               # 间隔时间范围 / Interval time range
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_Link"),
            # 力的范围 [N] / Force range [N]
            "force_range": {
                "x": (-500.0, 500.0), "y": (-500.0, 500.0), "z": (-0.0, 0.0),
            },
            # 力矩范围 [N⋅m] / Torque range [N⋅m]
            "torque_range": {"x": (-50.0, 50.0), "y": (-50.0, 50.0), "z": (-0.0, 0.0)},
            "probability": 0.002,                   # 发生概率 / Occurrence probability
        },
        is_global_time=False,
        min_step_count_between_reset=0,
    )


@configclass
class RewardsCfg:
    """奖励项配置类 - 定义强化学习的奖励函数 / Reward terms configuration class - defines RL reward functions"""
    #
    # 终止相关奖励 / Termination-related rewards
    keep_balance = RewTerm(
        func=mdp.stay_alive,    # 保持存活奖励 / Stay alive reward
        weight=1.0              # 奖励权重 / Reward weight
    )

    # tracking related rewards
    rew_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=3.0, params={"command_name": "base_velocity", "std": math.sqrt(0.2)}
    )
    rew_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=1.5, params={"command_name": "base_velocity", "std": math.sqrt(0.2)}
    )
    rew_no_fly = RewTerm(
        func=mdp.no_fly,
        weight=1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot_[LR]_Link"),
            "threshold": 5.0,
        },
    )

    # 调节相关奖励 / Regulation-related rewards
    pen_base_height = RewTerm(
        func=mdp.base_com_height,                   # 基座高度惩罚 / Base height penalty
        params={"target_height": 0.78,
                # "sensor_cfg": SceneEntityCfg("height_scanner")
                },
        weight=-1.0,                               # 负权重表示惩罚 / Negative weight indicates penalty
    )
    
    # 关节相关惩罚 / Joint-related penalties
    pen_lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.5)
    pen_ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    pen_joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-2.0e-05)
    pen_joint_accel = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-07)
    pen_action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    pen_joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)
    pen_joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-5.0e-05)
    pen_joint_powers = RewTerm(func=mdp.joint_powers_l1, weight=-2e-05)
    
    pen_undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,                # 不期望接触惩罚 / Undesired contacts penalty
        weight=-0.5,
        params={
            # 监控非足部的接触 / Monitor non-foot contacts
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["abad_.*", "hip_.*", "knee_.*", "base_Link"]),
            "threshold": 10.0,                      # 接触力阈值 / Contact force threshold
        },
    )

    pen_action_smoothness = RewTerm(
        func=mdp.ActionSmoothnessPenalty,           # 动作平滑性惩罚 / Action smoothness penalty
        weight=-0.01
    )
    pen_flat_orientation = RewTerm(
        func=mdp.flat_orientation_l2,               # 平坦朝向L2惩罚 / Flat orientation L2 penalty
        weight=-1.0
    )
    #
    pen_feet_distance = RewTerm(
        func=mdp.feet_distance,                     # 足部距离惩罚 / Foot distance penalty
        weight=-100,
        params={
            "min_feet_distance": 0.115,            # 最小足部距离 / Minimum foot distance
            "feet_links_name": ["foot_[RL]_Link"]  # 足部连杆名称 / Foot link names
        }
    )
    #
    pen_feet_regulation = RewTerm(
        func=mdp.feet_regulation,                   # 足部调节惩罚 / Foot regulation penalty
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["foot_[RL]_Link"]),
            "base_height_target": 0.65,            # 基座目标高度 / Base target height
            "foot_radius": 0.03,                    # 足部半径 / Foot radius
            # "sensor_cfg": SceneEntityCfg("height_scanner"),
        },
    )
    #
    foot_landing_vel = RewTerm(
        func=mdp.foot_landing_vel,                  # 足部着陆速度惩罚 / Foot landing velocity penalty
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["foot_[RL]_Link"]),
            "contact_sensor_cfg": SceneEntityCfg("contact_forces", body_names=["foot_[RL]_Link"]),
            "foot_radius": 0.03,
            "about_landing_threshold": 0.08,         # 即将着陆阈值 / About to land threshold
            # "height_sensor_cfg": SceneEntityCfg("height_scanner"),
        },
    )
    
    
    # 步态奖励 / Gait reward
    test_gait_reward = RewTerm(
        func=mdp.GaitReward,                        # 步态奖励函数 / Gait reward function
        weight=1.0,
        params={
            "tracking_contacts_shaped_force": -1.0,    # 接触力跟踪形状参数 / Contact force tracking shaping
            "tracking_contacts_shaped_vel": -1.0,      # 接触速度跟踪形状参数 / Contact velocity tracking shaping
            "gait_force_sigma": 25.0,                  # 步态力标准差 / Gait force sigma
            "gait_vel_sigma": 0.25,                    # 步态速度标准差 / Gait velocity sigma
            "kappa_gait_probs": 0.05,                  # 步态概率参数 / Gait probability parameter
            "command_name": "gait_command",            # 命令名称 / Command name
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="foot_.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names="foot_.*"),
        },
    )


@configclass
class TerminationsCfg:
    """终止条件配置类 / Termination conditions configuration class"""

    # 时间超时终止 / Time out termination
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    
    # 基座接触终止 (机器人倒下) / Base contact termination (robot falls down)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_Link"),
            "threshold": 1.0                        # 接触力阈值 / Contact force threshold
        },
    )


@configclass
class CurriculumCfg:
    """课程学习配置类 / Curriculum learning configuration class"""

    # 地形难度课程 / Terrain difficulty curriculum
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


########################
# 环境定义 / Environment Definition
########################


@configclass
class PFEnvCfg(ManagerBasedRLEnvCfg):
    """测试环境配置类 / Test environment configuration class"""

    # 场景设置 / Scene settings
    scene: PFSceneCfg = PFSceneCfg(num_envs=4096, env_spacing=2.5)
    # 基本设置 / Basic settings
    observations: ObservarionsCfg = ObservarionsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandCfg = CommandCfg()
    # MDP设置 / MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """后初始化配置 / Post-initialization configuration"""
        self.decimation = 4                         # 控制频率降采样 (50Hz -> 12.5Hz) / Control frequency downsampling
        self.episode_length_s = 20.0               # 每个episode长度20秒 / Episode length 20 seconds
        self.sim.render_interval = 2 * self.decimation  # 渲染间隔 / Rendering interval
        
        # 仿真设置 / Simulation settings
        self.sim.dt = 0.005                        # 仿真时间步 5ms / Simulation timestep 5ms
        self.seed = 42                             # 随机种子 / Random seed
        
        # 更新传感器更新周期 / Update sensor update periods
        # 基于最小更新周期(物理更新周期)来同步所有传感器 / Sync all sensors based on smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
