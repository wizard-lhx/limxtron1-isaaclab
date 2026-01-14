# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ActorCritic, MLP_Encoder
from rsl_rl.storage import RolloutStorage


class PPO:
    """PPO算法实现类 - 近端策略优化算法的完整实现 / PPO algorithm implementation class - complete implementation of Proximal Policy Optimization"""
    actor_critic: ActorCritic  # Actor-Critic网络 / Actor-Critic network
    encoder: MLP_Encoder       # MLP编码器 / MLP encoder

    def __init__(
        self,
        num_group,                    # 环境组数 / Number of environment groups
        encoder,                      # 编码器实例 / Encoder instance
        actor_critic,                 # Actor-Critic网络实例 / Actor-Critic network instance
        num_learning_epochs=1,        # 学习轮数 / Number of learning epochs
        num_mini_batches=1,           # 小批次数量 / Number of mini-batches
        clip_param=0.2,               # PPO裁剪参数 / PPO clipping parameter
        gamma=0.998,                  # 折扣因子 / Discount factor
        lam=0.95,                     # GAE lambda参数 / GAE lambda parameter
        value_loss_coef=1.0,          # 值函数损失系数 / Value function loss coefficient
        entropy_coef=0.0,             # 熵正则化系数 / Entropy regularization coefficient
        learning_rate=1e-3,           # 学习率 / Learning rate
        max_grad_norm=1.0,            # 梯度裁剪阈值 / Gradient clipping threshold
        use_clipped_value_loss=True,  # 使用裁剪值函数损失 / Use clipped value function loss
        schedule="fixed",             # 学习率调度策略 / Learning rate scheduling strategy
        desired_kl=0.01,              # 目标KL散度 / Target KL divergence
        vae_beta=1.0,                 # VAE beta参数 / VAE beta parameter
        est_learning_rate=1.0e-3,     # 估计器学习率 / Estimator learning rate
        critic_take_latent=False,     # 评价器是否使用潜在表示 / Whether critic uses latent representation
        early_stop=False,             # 早停机制 / Early stopping mechanism
        anneal_lr=False,              # 学习率衰减 / Learning rate annealing
        device="cpu",                 # 计算设备 / Computing device
        **kwargs
    ):
        self.device = device
        self.num_group = num_group

        self.desired_kl = desired_kl
        self.early_stop = early_stop
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.anneal_lr = anneal_lr
        self.vae_beta = vae_beta
        self.critic_take_latent = critic_take_latent

        self.encoder = encoder

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None  # initialized later
        self.optimizer = optim.Adam([{"params": self.actor_critic.parameters()}], lr=learning_rate)

        if self.encoder.num_output_dim != 0:
            self.extra_optimizer = optim.Adam(
                self.encoder.parameters(), lr=est_learning_rate
            )
        else:
            self.extra_optimizer = None
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

    def init_storage(
        self,
        num_envs,                     # 环境数量 / Number of environments
        num_transitions_per_env,      # 每个环境的转换数 / Number of transitions per environment
        actor_obs_shape,              # Actor观测形状 / Actor observation shape
        critic_obs_shape,             # Critic观测形状 / Critic observation shape
        obs_history_shape,            # 观测历史形状 / Observation history shape
        commands_shape,               # 命令形状 / Commands shape
        action_shape,                 # 动作形状 / Action shape
    ):
        """初始化经验存储缓冲区 / Initialize experience storage buffer"""
        self.storage = RolloutStorage(
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            obs_history_shape,
            commands_shape,
            action_shape,
            self.device,
        )

    def test_mode(self):
        """设置为测试模式 / Set to test mode"""
        self.actor_critic.test()

    def train_mode(self):
        """设置为训练模式 / Set to training mode"""
        self.actor_critic.train()

    def act(self, obs, obs_history, commands, critic_obs):
        """执行动作选择和价值评估 / Perform action selection and value evaluation
        
        Args:
            obs: 当前观测 / Current observations
            obs_history: 观测历史 / Observation history
            commands: 命令输入 / Command inputs
            critic_obs: 评价器观测 / Critic observations
            
        Returns:
            选择的动作 / Selected actions
        """

        # 拼接评价器观测和命令 / Concatenate critic observations and commands
        critic_obs = torch.cat((critic_obs, commands), dim=-1)

        # 动作选择 / Action selection
        encoder_out = self.encoder.encode(obs_history)
        self.transition.actions = self.actor_critic.act(
            torch.cat((encoder_out, obs, commands), dim=-1)
        ).detach()

        # 价值评估 / Value evaluation
        if self.critic_take_latent:
            critic_obs = torch.cat((critic_obs, encoder_out), dim=-1)
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()

        # 存储转换信息 / Store transition information
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(
            self.transition.actions
        ).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()

        # 需要在env.step()之前记录观测和评价器观测 / Need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_obs = critic_obs
        self.transition.observation_history = obs_history
        self.transition.commands = commands
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos, next_obs=None):
        """处理环境步进结果 / Process environment step results
        
        Args:
            rewards: 奖励信号 / Reward signals
            dones: 结束标志 / Done flags
            infos: 额外信息 / Additional information
            next_obs: 下一步观测 / Next observations
        """
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values
                * infos["time_outs"].unsqueeze(1).to(self.device),
                1,
            )

        # Record the transition
        self.transition.next_observations = next_obs
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs):
        """计算回报值 / Compute returns"""
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        """执行PPO更新 / Perform PPO update
        
        Returns:
            平均值函数损失、额外损失、代理损失、KL散度 / Average value loss, extra loss, surrogate loss, KL divergence
        """
        num_updates = 0
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_kl = 0
        generator = self.storage.mini_batch_generator(
            self.num_group,
            self.num_mini_batches,
            self.num_learning_epochs,
        )
        for (
            obs_batch,
            critic_obs_batch,
            obs_history_batch, _,
            group_commands_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
        ) in generator:
            # 编码观测历史 / Encode observation history
            encoder_out_batch = self.encoder.encode(obs_history_batch)
            commands_batch = group_commands_batch

            # 前向传播获取新的策略分布 / Forward pass to get new policy distribution
            self.actor_critic.act(
                torch.cat(
                    (encoder_out_batch, obs_batch, commands_batch),
                    dim=-1,
                )
            )

            # 计算新的动作对数概率 / Calculate new action log probabilities
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(
                actions_batch
            )

            # 计算价值函数输出 / Calculate value function output
            value_batch = self.actor_critic.evaluate(critic_obs_batch)
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # 计算KL散度 / Calculate KL divergence
            kl_mean = torch.tensor(0, device=self.device, requires_grad=False)
            with torch.inference_mode():
                kl = torch.sum(
                    torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                    + (
                        torch.square(old_sigma_batch)
                        + torch.square(old_mu_batch - mu_batch)
                    )
                    / (2.0 * torch.square(sigma_batch))
                    - 0.5,
                    axis=-1,
                )
                kl_mean = torch.mean(kl)

            # 自适应学习率调整 / Adaptive learning rate adjustment
            if self.desired_kl != None and self.schedule == "adaptive":
                with torch.inference_mode():
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # 早停机制 / Early stopping mechanism
            if self.desired_kl != None and self.early_stop:
                if kl_mean > self.desired_kl * 1.5:
                    print("early stop, num_updates =", num_updates)
                    break

            # 计算代理损失 / Calculate surrogate loss
            ratio = torch.exp(
                actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch)
            )
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # 计算价值函数损失 / Calculate value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (
                    value_batch - target_values_batch
                ).clamp(-self.clip_param, self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            # 计算总损失 / Calculate total loss
            entropy_batch_mean = entropy_batch.mean()
            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch_mean
            )
            
            # lin_vel_est = encoder_out_batch[:, :3]
            # lin_vel_target = critic_obs_batch[:, :3]
            # lin_vel_est_loss = torch.nn.functional.mse_loss(
            #     lin_vel_est, lin_vel_target
            # )
            # loss += lin_vel_est_loss

            # 学习率退火 / Learning rate annealing
            if self.anneal_lr:
                frac = 1.0 - num_updates / (
                    self.num_learning_epochs * self.num_mini_batches
                )
                self.optimizer.param_groups[0]["lr"] = frac * self.learning_rate

            # 梯度更新 / Gradient update
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            num_updates += 1
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_kl += kl_mean.item()

        # 编码器额外更新 / Additional encoder updates
        num_updates_extra = 0
        mean_extra_loss = 0
        if self.extra_optimizer is not None:
            generator = self.storage.encoder_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
            for (
                next_obs_batch,
                critic_obs_batch,
                obs_history_batch,
            ) in generator:
                if self.encoder.is_mlp_encoder:
                    self.encoder.encode(obs_history_batch)
                    encode_batch = self.encoder.get_encoder_out()

                if self.encoder.is_mlp_encoder:
                    extra_loss = (
                        (encode_batch[:, 0:3] - critic_obs_batch[:, 0:3]).pow(2).mean()
                    )
                else:
                    extra_loss = torch.zeros_like(value_loss)

                self.extra_optimizer.zero_grad()
                extra_loss.backward()
                self.extra_optimizer.step()

                num_updates_extra += 1
                mean_extra_loss += extra_loss.item()

        # 计算平均损失 / Calculate average losses
        mean_value_loss /= num_updates
        if num_updates_extra > 0:
            mean_extra_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_kl /= num_updates
        self.storage.clear()

        return (mean_value_loss, mean_extra_loss, mean_surrogate_loss, mean_kl)
