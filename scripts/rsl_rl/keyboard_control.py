"""
键盘控制测试脚本 - Keyboard Control Test Script
允许用户通过键盘实时控制机器人速度指令
Allows users to control robot velocity commands in real-time via keyboard
"""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# Parse command line arguments
parser = argparse.ArgumentParser(description="键盘控制机器人行走 / Keyboard control for robot walking")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Limx-PF-Blind-Rough-Play-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--checkpoint_path", type=str, default="/home/lhx/project/limxtron1lab-main/logs/rsl_rl/pf_tron_1a_flat/2025-12-24_10-14-58/model_3000.pt", help="Relative path to checkpoint file.")
parser.add_argument("--use_onnx", action="store_true", help="使用ONNX模型 / Use ONNX model")

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

from rsl_rl.runner import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg, DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

# Import extensions to set up environment tasks
import bipedal_locomotion  # noqa: F401
from bipedal_locomotion.utils.wrappers.rsl_rl import RslRlPpoAlgorithmMlpCfg


class ManualController:
    """Manual controller for robot using WASD keys."""
    
    def __init__(self, device="cuda"):
        self.device = device
        self.linear_velocity = torch.zeros(3, device=device)
        self.angular_velocity = torch.zeros(3, device=device)
        self.max_linear_vel = 1.0
        self.max_angular_vel = 1.0
        self.max_force = 200.0
        self.keys_pressed_vel = []
        self.keys_pressed_force = []
        # Get keyboard input handler
        from pynput import keyboard
        def on_press(key):
            try:
                self.keys_pressed_vel.clear()
                # Get current key presses
                if key.char=='w': self.keys_pressed_vel.append('w')
                if key.char=='a': self.keys_pressed_vel.append('a')
                if key.char=='s': self.keys_pressed_vel.append('s')
                if key.char=='d': self.keys_pressed_vel.append('d')
                if key.char=='q': self.keys_pressed_vel.append('q')
                if key.char=='e': self.keys_pressed_vel.append('e')
                if key.char=='p': self.keys_pressed_vel.append('p')

                if key.char=='i': self.keys_pressed_force.append('i')
                if key.char=='j': self.keys_pressed_force.append('j')
                if key.char=='k': self.keys_pressed_force.append('k')
                if key.char=='l': self.keys_pressed_force.append('l')
                if key.char=='u': self.keys_pressed_force.append('u')
                print('字母键： {} 被按下'.format(key.char))
            except AttributeError:
                print('特殊键： {} 被按下'.format(key))


        def on_release(key):
            # print('{} 释放了'.format(key))
            if key == keyboard.Key.esc:
                # 释放了esc 键，停止监听
                return False
        self.listener = keyboard.Listener(
            on_press=on_press,
            on_release=on_release)
        self.listener.start()
        
    def get_velocity_command(self):
        """Return SE2 command [vx, vy, wz] in robot base frame."""
        return torch.tensor(
            [self.linear_velocity[0].item(), self.linear_velocity[1].item(), self.angular_velocity[2].item()],
            device=self.device,
            dtype=torch.float32,
        )
    def get_force_command(self):
        """Return external force command [fx, fy, fz]."""
        return self.force
    
    def update_from_vel_keys(self, keys_pressed_vel):
        """Update velocity based on pressed keys."""
        # Reset velocities
        self.linear_velocity.zero_()
        self.angular_velocity.zero_()
        
        # WASD control
        if 'w' in keys_pressed_vel:
            self.linear_velocity[0] = self.max_linear_vel  # Forward
        if 's' in keys_pressed_vel:
            self.linear_velocity[0] = -self.max_linear_vel  # Backward
        if 'a' in keys_pressed_vel:
            self.linear_velocity[1] = self.max_linear_vel  # Left
        if 'd' in keys_pressed_vel:
            self.linear_velocity[1] = -self.max_linear_vel  # Right
            
        # QE for rotation
        if 'q' in keys_pressed_vel:
            self.angular_velocity[2] = self.max_angular_vel  # Turn left
        if 'e' in keys_pressed_vel:
            self.angular_velocity[2] = -self.max_angular_vel  # Turn right
        # P for stop
        if 'p' in keys_pressed_vel:
            self.linear_velocity.zero_()
            self.angular_velocity.zero_()
    
    def update_from_force_keys(self, keys_pressed_force):
        """Update external force based on pressed keys."""
        self.force = torch.zeros(3, device=self.device)
        if 'k' in keys_pressed_force:
            self.force[0] = self.max_force
        if 'i' in keys_pressed_force:
            self.force[0] = -self.max_force
        if 'j' in keys_pressed_force:
            self.force[1] = self.max_force
        if 'l' in keys_pressed_force:
            self.force[1] = -self.max_force
        if 'u' in keys_pressed_force:
            self.force.zero_()
def main():
    """主函数 / Main function"""
    
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
            "video_folder": os.path.join(log_dir, "videos", "keyboard_control"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during keyboard control.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    
    # Initialize manual controller and camera controller
    manual_controller = ManualController(device=env.unwrapped.device)
    
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)
    
    # 加载策略 / Load policy
    policy = None
    encoder = None
    
    if args_cli.use_onnx:
        # 使用ONNX模型 / Use ONNX model
        import onnxruntime as ort
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        policy_path = os.path.join(export_model_dir, "policy.onnx")
        encoder_path = os.path.join(export_model_dir, "encoder.onnx")
        
        if os.path.exists(policy_path) and os.path.exists(encoder_path):
            policy_session = ort.InferenceSession(policy_path)
            encoder_session = ort.InferenceSession(encoder_path)
            
            def encoder_fn(obs):
                obs_np = obs.cpu().numpy()
                encoder_out = encoder_session.run(None, {"obs": obs_np})[0]
                return torch.from_numpy(encoder_out).to(env.unwrapped.device)
            
            def policy_fn(obs):
                obs_np = obs.cpu().numpy()
                action = policy_session.run(None, {"obs": obs_np})[0]
                return torch.from_numpy(action).to(env.unwrapped.device)
            
            encoder = encoder_fn
            policy = policy_fn
            print(f"[INFO] 已加载ONNX模型 / Loaded ONNX model from: {export_model_dir}")
        else:
            print(f"[WARNING] ONNX模型文件不存在 / ONNX model files not found at: {export_model_dir}")
            print("[INFO] 将使用PyTorch模型 / Will use PyTorch model instead")
            args_cli.use_onnx = False
    
    if not args_cli.use_onnx:
        # 使用PyTorch模型 / Use PyTorch model
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        ppo_runner.load(resume_path)
        
        # obtain the trained policy for inference
        policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
        encoder = ppo_runner.get_inference_encoder(device=env.unwrapped.device)
        print(f"[INFO] 已加载PyTorch模型 / Loaded PyTorch model")

    # reset environment
    obs, obs_dict = env.get_observations()
    obs_history = obs_dict["observations"].get("obsHistory")
    obs_history = obs_history.flatten(start_dim=1)
    commands = obs_dict["observations"].get("commands")
    
    # 检查策略是否已加载 / Check if policy is loaded
    if policy is None or encoder is None:
        raise RuntimeError("策略未加载！请提供有效的检查点路径。/ Policy not loaded! Please provide a valid checkpoint path.")
    
    # 主循环 / Main loop
    print("\n开始键盘控制模式 / Starting keyboard control mode...\n")
    
    # initialize commands and step count
    step_count = 0
    commands = torch.zeros((env.num_envs, 3), device=env.unwrapped.device)
    cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
    cmd_term.vel_command_b[:] = commands
    
    # apply external force function
    robot = env.unwrapped.scene["robot"]
    def apply_external_force(force):
        torque = torch.zeros((1, 3), device=robot.device)
        robot.set_external_force_and_torque(
            force.unsqueeze(1),  # shape: [num_envs, num_bodies, 3]
            torque.unsqueeze(1), 
            env_ids=torch.tensor([0], device=robot.device),
            body_ids=[robot.find_bodies("base_Link")[0][0]]
        )
        manual_controller.keys_pressed_force.clear()

    # simulate environment
    while simulation_app.is_running():       
        # Update manual controller
        if manual_controller.keys_pressed_vel:
            manual_controller.update_from_vel_keys(manual_controller.keys_pressed_vel)
            # Override the command in the environment (expects [vx, vy, wz])
            velocity_cmd = manual_controller.get_velocity_command()
            cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
            # broadcast to all envs
            commands = velocity_cmd.unsqueeze(0).repeat(env.num_envs, 1)
            cmd_term.vel_command_b[:] = commands
            # ensure angular velocity mode (not heading) and not standing
            if hasattr(cmd_term, "is_heading_env"):
                cmd_term.is_heading_env[:] = False
            if hasattr(cmd_term, "is_standing_env"):
                cmd_term.is_standing_env[:] = False
        if manual_controller.keys_pressed_force:
            manual_controller.update_from_force_keys(manual_controller.keys_pressed_force)
            # Apply external force to the robot base
            force = manual_controller.get_force_command()
            apply_external_force(force)
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
            
            step_count += 1
            
            actual_vel = env.unwrapped.scene["robot"].data.root_lin_vel_w[0, :2]
            actual_ang_vel = env.unwrapped.scene["robot"].data.root_ang_vel_w[0, 2]
            mse_vel = torch.mean((torch.cat([actual_vel,actual_ang_vel.unsqueeze(0)])-commands[0]) ** 2).item()
            # 每100步打印一次状态 / Print status every 100 steps
            if step_count % 100 == 0:
                print(f"\n步数 / Steps: {step_count}")
                print(f"指令 / Command: vx={commands[0, 0].item():.2f}, vy={commands[0, 1].item():.2f}, wz={commands[0, 2].item():.2f}")
                print(f"实际 / Actual:  vx={actual_vel[0].item():.2f}, vy={actual_vel[1].item():.2f}, wz={actual_ang_vel.item():.2f}")
                print(f"实际速度与指令速度的均方误差: MSE={mse_vel:.2f}")

    # 关闭环境 / Close environment
    env.close()


if __name__ == "__main__":
    # Run the main function
    main()
    # Close sim app
    simulation_app.close()