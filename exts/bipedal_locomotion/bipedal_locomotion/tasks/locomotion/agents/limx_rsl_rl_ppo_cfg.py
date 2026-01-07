from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from bipedal_locomotion.utils.wrappers.rsl_rl.rl_mlp_cfg import EncoderCfg, RslRlPpoAlgorithmMlpCfg

import os
robot_type = os.getenv("ROBOT_TYPE")  # 从环境变量获取机器人类型 / Get robot type from environment variable

# Isaac Lab original RSL-RL configuration
@configclass
class PFPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24        # 每个环境每次收集的步数 / Steps collected per environment per iteration
    max_iterations = 15000        # 最大训练迭代次数 / Maximum training iterations
    save_interval = 500           # 模型保存间隔 / Model saving interval
    experiment_name = "pf_flat"   # 实验名称 / Experiment name
    empirical_normalization = False  # 不使用经验归一化 / Don't use empirical normalization

    # Actor-Critic网络配置
    # Actor-Critic network configuration
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,              # 初始动作噪声标准差 / Initial action noise std
        actor_hidden_dims=[512, 256, 128],  # Actor网络隐藏层维度 / Actor network hidden dimensions
        critic_hidden_dims=[512, 256, 128], # Critic网络隐藏层维度 / Critic network hidden dimensions
        activation="elu",                # 激活函数类型 / Activation function type
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,          # 值函数损失系数 / Value function loss coefficient
        use_clipped_value_loss=True,  # 使用截断值函数损失 / Use clipped value function loss
        clip_param=0.2,               # PPO截断参数 / PPO clipping parameter
        entropy_coef=0.01,            # 熵正则化系数 / Entropy regularization coefficient
        num_learning_epochs=5,        # 每次迭代的学习轮数 / Learning epochs per iteration
        num_mini_batches=4,           # 小批次数量 / Number of mini-batches
        learning_rate=1.0e-3,         # 学习率 / Learning rate
        schedule="adaptive",          # 自适应学习率调度 / Adaptive learning rate schedule
        gamma=0.99,                   # 折扣因子 / Discount factor
        lam=0.95,                     # GAE lambda参数 / GAE lambda parameter
        desired_kl=0.01,              # 目标KL散度 / Target KL divergence
        max_grad_norm=1.0,            # 梯度裁剪阈值 / Gradient clipping threshold
    )


# PF_TRON1A平地训练配置 - 针对特定机器人型号的优化配置
# PF_TRON1A flat terrain training configuration - optimized for specific robot model
@configclass
class PF_TRON1AFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 3000         # 较短的训练周期，适合平地环境 / Shorter training for flat terrain
    save_interval = 200           # 更频繁的保存 / More frequent saving
    experiment_name = "pf_tron_1a_flat"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    # 使用MLP版本的PPO算法，支持历史观测
    # Use MLP version of PPO algorithm with history observation support
    algorithm = RslRlPpoAlgorithmMlpCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        obs_history_len=10,   # 观测历史长度 / Observation history length
    )

    # 编码器配置 - 用于处理历史观测信息
    # Encoder configuration - for processing history observation information
    encoder = EncoderCfg(
        output_detach=True,       # 输出分离，防止梯度回传 / Detach output to prevent gradient flow
        num_output_dim=3,         # 输出维度 / Output dimensions
        hidden_dims=[256, 128],   # 编码器隐藏层 / Encoder hidden dimensions
        activation="elu",         # 激活函数 / Activation function
        orthogonal_init=False,    # 不使用正交初始化 / Don't use orthogonal initialization
    )

# PF_TRON1A粗糙训练配置 - 针对特定机器人型号的优化配置
# PF_TRON1A rough terrain training configuration - optimized for specific robot model
@configclass
class PF_TRON1ARoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 15000         # 适合粗糙环境 / Suitable for rough terrain
    save_interval = 200           # 更频繁的保存 / More frequent saving
    experiment_name = "pf_tron_1a_rough"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    # 使用MLP版本的PPO算法，支持历史观测
    # Use MLP version of PPO algorithm with history observation support
    algorithm = RslRlPpoAlgorithmMlpCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        obs_history_len=10,   # 观测历史长度 / Observation history length
    )

    # 编码器配置 - 用于处理历史观测信息
    # Encoder configuration - for processing history observation information
    encoder = EncoderCfg(
        output_detach=True,       # 输出分离，防止梯度回传 / Detach output to prevent gradient flow
        num_output_dim=3,         # 输出维度 / Output dimensions
        hidden_dims=[256, 128],   # 编码器隐藏层 / Encoder hidden dimensions
        activation="elu",         # 激活函数 / Activation function
        orthogonal_init=False,    # 不使用正交初始化 / Don't use orthogonal initialization
    )

# PF_TRON1A楼梯训练配置 - 针对特定机器人型号的优化配置
# PF_TRON1A stair terrain training configuration - optimized for specific robot model
class PF_TRON1AStairPPORunnerCfg(PF_TRON1AFlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "pf_tron_1a_stair"
        self.max_iterations = 15000         # 适合楼梯环境 / Suitable for stair terrain
#-----------------------------------------------------------------
@configclass
class SF_TRON1AFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 15000
    save_interval = 500
    experiment_name = "sf_tron_1a_flat"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmMlpCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        obs_history_len=10,
    )
    encoder = EncoderCfg(
        output_detach = True,
        num_output_dim = 3,
        hidden_dims = [256, 128],
        activation = "elu",
        orthogonal_init = False,
    )


#-----------------------------------------------------------------
@configclass
class WF_TRON1AFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 500
    experiment_name = "wf_tron_1a_flat"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmMlpCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        obs_history_len=10,
    )
    encoder = EncoderCfg(
        output_detach = True,
        num_output_dim = 3,
        hidden_dims = [256, 128],
        activation = "elu",
        orthogonal_init = False,
    )
