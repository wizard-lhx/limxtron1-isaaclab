"""RSL-RL智能体检查点播放脚本 / Script to play a checkpoint of an RL agent from RSL-RL."""

"""首先启动Isaac Sim仿真器 / Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# 添加argparse参数 / Add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--checkpoint_path", type=str, default=None, help="Relative path to checkpoint file.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


import gymnasium as gym
import os
import torch
import numpy as np
import time
import matplotlib.pyplot as plt

from rsl_rl.runner import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg,DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
# Import extensions to set up environment tasks
import bipedal_locomotion  # noqa: F401
from bipedal_locomotion.utils.wrappers.rsl_rl import RslRlPpoAlgorithmMlpCfg, export_mlp_as_onnx, export_policy_as_jit

# # [新增 1] 初始化键盘控制器
# from isaaclab.devices.keyboard import Se3Keyboard, Se3KeyboardCfg
# keyboard_cfg = Se3KeyboardCfg()
# # keyboard_cfg.vx_scale = 1.0      # 前后速度缩放
# # keyboard_cfg.vy_scale = 1.0      # 左右速度缩放
# # keyboard_cfg.yaw_scale = 1.5     # 旋转速度更敏感
# keyboard = Se3Keyboard(cfg=keyboard_cfg)
# print("\n" + "=" * 50)
# print("键盘控制已激活 / Keyboard Control Active")
# print("W / S : 前进 / 后退 (Linear Velocity X)")
# print("A / D : 左移 / 右移 (Linear Velocity Y)")
# print("Q / E : 左转 / 右转 (Angular Velocity Z)")
# print("K     : 复位键盘输入 (Reset Input)")
# print("=" * 50 + "\n")

"""
改造说明：
- 移除键盘控制，改为自动随机指令。
- 持续 60s 仿真，每 5s 采样一次新的随机速度指令：vx, vy, wz ∈ U(-1.2, 1.2)。
- 记录期望与实际速度（基座坐标系）并在结束后绘图保存。
"""

class CameraController:
    """Camera controller to follow the robot (no external deps)."""

    def __init__(self, env, camera_distance=5.0, camera_height=2.0):
        self.env = env
        self.camera_distance = camera_distance
        self.camera_height = camera_height

    def update_camera_view(self):
        """Update camera to follow the robot."""
        # Get robot position and yaw heading from quaternion
        robot_pos = self.env.unwrapped.scene["robot"].data.root_pos_w[0].cpu().numpy()
        q = self.env.unwrapped.scene["robot"].data.root_quat_w[0].cpu().numpy()  # [w, x, y, z]
        w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        # Compute forward vector in world from yaw (approx):
        # forward in local = [1, 0, 0]
        # world forward xz using quaternion (ignore roll/pitch for camera)
        # yaw from quaternion
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        forward = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        # camera behind the robot along -forward
        camera_eye = robot_pos + (-forward * self.camera_distance)
        camera_eye[2] += self.camera_height
        # look at the robot center slightly above
        camera_target = robot_pos.copy()
        camera_target[2] += 0.5
        self.env.unwrapped.sim.set_camera_view(eye=camera_eye, target=camera_target, camera_prim_path="/OmniverseKit_Persp")

def main():
    """使用RSL-RL智能体进行测试 / Play with RSL-RL agent."""
    # 解析配置 / Parse configuration
    env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(
        task_name=args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs
    )
    agent_cfg: RslRlPpoAlgorithmMlpCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    env_cfg.seed = agent_cfg.seed

    # 指定日志实验目录 / Specify directory for logging experiments
    if args_cli.checkpoint_path is None:
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    else:
        resume_path = args_cli.checkpoint_path
    log_dir = os.path.dirname(resume_path)

    # 创建isaac环境 / Create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env,clip_actions=5.0)
    # load previously trained model
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    encoder = ppo_runner.get_inference_encoder(device=env.unwrapped.device)

     # 导出策略到onnx / Export policy to onnx
    if EXPORT_POLICY:
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        export_policy_as_jit(
            ppo_runner.alg.actor_critic, export_model_dir
        )
        print("Exported policy as jit script to: ", export_model_dir)
        export_mlp_as_onnx(
            ppo_runner.alg.actor_critic.actor, 
            export_model_dir, 
            "policy",
            ppo_runner.alg.actor_critic.num_actor_obs,
        )
        export_mlp_as_onnx(
            ppo_runner.alg.encoder,
            export_model_dir,
            "encoder",
            ppo_runner.alg.encoder.num_input_dim,
        )

    # Initialize camera controller
    camera_controller = CameraController(env)

    timestep = 0
    
    # Initialize velocity tracking for MSE calculation
    velocity_errors_squared = []
    # Time-series logging for plotting
    t_series = []
    des_vx_series, des_vy_series, des_wz_series = [], [], []
    act_vx_series, act_vy_series, act_wz_series = [], [], []
    
    # Initialize loop timing measurement
    loop_times = []
    last_loop_time = time.time()

    # reset environment
    obs, obs_dict = env.get_observations()
    obs_history = obs_dict["observations"].get("obsHistory")
    obs_history = obs_history.flatten(start_dim=1)
    commands = obs_dict["observations"].get("commands")

    # Command scheduling
    duration_s = 30.0
    change_period_s = 6.0
    start_time = time.time()
    next_change_time = start_time

    # Prepare command manager term (base_velocity)
    cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
    # Initialize first command immediately
    rng = np.random.default_rng()
    def sample_cmd():
        return np.array([
            rng.uniform(-1.2, 1.2),  # vx
            rng.uniform(-1.2, 1.2),  # vy
            rng.uniform(-1.2, 1.2),  # wz
        ], dtype=np.float32)

    current_cmd_np = sample_cmd()
    current_cmd = torch.tensor(current_cmd_np, device=env.unwrapped.device, dtype=torch.float32)
    # broadcast to all envs for command manager
    cmd_term.vel_command_b[:] = current_cmd.unsqueeze(0).repeat(env.num_envs, 1)
    # also update the commands tensor fed to policy
    commands = commands.clone()
    commands[:, :3] = current_cmd.unsqueeze(0)

    # simulate environment
    while simulation_app.is_running():
        # Measure loop timing
        current_time = time.time()
        loop_dt = current_time - last_loop_time
        loop_times.append(loop_dt)
        last_loop_time = current_time

        # Stop condition: run for duration_s seconds
        if (current_time - start_time) >= duration_s:
            print(f"[INFO] Reached {duration_s:.1f}s, stopping simulation loop.")
            break

        # Change command every change_period_s seconds
        if current_time >= next_change_time:
            current_cmd_np = sample_cmd()
            current_cmd = torch.tensor(current_cmd_np, device=env.unwrapped.device, dtype=torch.float32)
            # apply to env command manager and policy input
            cmd_term.vel_command_b[:] = current_cmd.unsqueeze(0).repeat(env.num_envs, 1)
            commands = commands.clone()
            commands[:, :3] = current_cmd.unsqueeze(0)
            next_change_time += change_period_s


        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            est = encoder(obs_history)
            actions = policy(torch.cat((est, obs, commands), dim=-1).detach())
            # env stepping
            obs, _, _, infos = env.step(actions)
            obs_history = infos["observations"].get("obsHistory")
            obs_history = obs_history.flatten(start_dim=1)
            commands = infos["observations"].get("commands")
            
        # Get robot actual velocity and desired velocity
        robot = env.unwrapped.scene["robot"]
        
        # Check if base frame velocities are available (first iteration only)
        if timestep == 0:
            print("\n=== Checking available velocity attributes ===")
            print(f"Has root_lin_vel_w: {hasattr(robot.data, 'root_lin_vel_w')}")
            print(f"Has root_ang_vel_w: {hasattr(robot.data, 'root_ang_vel_w')}")
            print(f"Has root_lin_vel_b: {hasattr(robot.data, 'root_lin_vel_b')}")
            print(f"Has root_ang_vel_b: {hasattr(robot.data, 'root_ang_vel_b')}")
            if hasattr(robot.data, 'root_lin_vel_b'):
                print("✓ Base-frame velocities are available directly!")
            else:
                print("✗ Base-frame velocities NOT available, using manual transformation")
            print("=" * 50 + "\n")
        
        # Try to use base-frame velocities if available, otherwise transform
        if hasattr(robot.data, 'root_lin_vel_b') and hasattr(robot.data, 'root_ang_vel_b'):
            # Direct access to base-frame velocities
            actual_lin_vel = robot.data.root_lin_vel_b[0, :2]  # [vx_b, vy_b] in base frame
            actual_ang_vel = robot.data.root_ang_vel_b[0, 2]   # wz in base frame

        
        # Desired velocity from commands (first env, already in base frame)
        desired_vel = commands[0, :3]  # [vx_des, vy_des, wz_des] in base frame
        desired_lin_vel = desired_vel[:2]
        desired_ang_vel = desired_vel[2]
        
        # Calculate velocity tracking error
        lin_vel_error = actual_lin_vel - desired_lin_vel
        ang_vel_error = actual_ang_vel - desired_ang_vel
        
        # Calculate MSE for this timestep
        lin_vel_mse = torch.sum(lin_vel_error ** 2).item()
        ang_vel_mse = (ang_vel_error ** 2).item()
        total_vel_mse = lin_vel_mse + ang_vel_mse
        
        velocity_errors_squared.append(total_vel_mse)
        
        # Log time-series for plotting
        t_series.append(current_time - start_time)
        des_vx_series.append(desired_lin_vel[0].item())
        des_vy_series.append(desired_lin_vel[1].item())
        des_wz_series.append(desired_ang_vel.item())
        act_vx_series.append(actual_lin_vel[0].item())
        act_vy_series.append(actual_lin_vel[1].item())
        act_wz_series.append(actual_ang_vel.item())

        # Print velocity tracking info every ~0.5s（根据步频估计）
        if timestep % 10 == 0:
            avg_mse = np.mean(velocity_errors_squared[-50:]) if len(velocity_errors_squared) >= 50 else np.mean(velocity_errors_squared)
            avg_loop_time = np.mean(loop_times[-100:]) if len(loop_times) >= 100 else np.mean(loop_times)
            avg_loop_freq = 1.0 / avg_loop_time if avg_loop_time > 0 else 0
            print(f"\n[Step {timestep}] Velocity Tracking:")
            print(f"  Desired: vx={desired_vel[0].item():.3f}, vy={desired_vel[1].item():.3f}, wz={desired_vel[2].item():.3f}")
            print(f"  Actual:  vx={actual_lin_vel[0].item():.3f}, vy={actual_lin_vel[1].item():.3f}, wz={actual_ang_vel.item():.3f}")
            print(f"  MSE (last 50 steps): {avg_mse:.6f}")
            print(f"  Loop timing: {avg_loop_time*1000:.2f}ms ({avg_loop_freq:.1f} Hz)")
        
        timestep += 1 

        # Update camera to follow robot
        camera_controller.update_camera_view()

    # Print final statistics
    if len(velocity_errors_squared) > 0:
        overall_mse = np.mean(velocity_errors_squared)
        avg_loop_time = np.mean(loop_times[1:]) if len(loop_times) > 1 else 0  # Skip first measurement
        avg_loop_freq = 1.0 / avg_loop_time if avg_loop_time > 0 else 0
        
        print(f"\n{'='*60}")
        print(f"[FINAL STATISTICS]")
        print(f"  Total timesteps: {timestep}")
        print(f"  Overall Velocity Tracking MSE: {overall_mse:.6f}")
        print(f"  Overall Velocity Tracking RMSE: {np.sqrt(overall_mse):.6f}")
        print(f"  Average loop time: {avg_loop_time*1000:.2f}ms ({avg_loop_freq:.1f} Hz)")
        print(f"  Physics dt: {env.unwrapped.physics_dt*1000:.2f}ms ({1.0/env.unwrapped.physics_dt:.0f} Hz)")
        print(f"  Control decimation: {env.unwrapped.cfg.decimation}")
        print(f"  Control dt: {env.unwrapped.step_dt*1000:.2f}ms ({1.0/env.unwrapped.step_dt:.1f} Hz)")
        print(f"{'='*60}\n")

    # Plot and save figure for desired vs actual velocities
    try:
        save_dir = os.path.join(os.path.dirname(log_dir), "velocitytest") if 'log_dir' in locals() else "./logs/velocitytest"
        os.makedirs(save_dir, exist_ok=True)
        fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        axs[0].plot(t_series, des_vx_series, label="vx_des")
        axs[0].plot(t_series, act_vx_series, label="vx_act", linestyle="--")
        axs[0].set_ylabel("vx [m/s]")
        axs[0].legend(loc="upper right")

        axs[1].plot(t_series, des_vy_series, label="vy_des")
        axs[1].plot(t_series, act_vy_series, label="vy_act", linestyle="--")
        axs[1].set_ylabel("vy [m/s]")
        axs[1].legend(loc="upper right")

        axs[2].plot(t_series, des_wz_series, label="wz_des")
        axs[2].plot(t_series, act_wz_series, label="wz_act", linestyle="--")
        axs[2].set_ylabel("wz [rad/s]")
        axs[2].set_xlabel("time [s]")
        axs[2].legend(loc="upper right")

        fig.suptitle("Velocity Tracking (cmd vs actual)")
        fig.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))
        save_path = os.path.join(save_dir, "velocity_tracking.png")
        plt.savefig(save_path, dpi=150)
        print(f"[INFO] Saved velocity tracking plot to: {os.path.abspath(save_path)}")
    except Exception as e:
        print(f"[WARN] Failed to save plot: {e}")

    # close the simulator
    env.close()


if __name__ == "__main__":
    EXPORT_POLICY = True
    # run the main execution
    main()
    # close sim app
    simulation_app.close()
