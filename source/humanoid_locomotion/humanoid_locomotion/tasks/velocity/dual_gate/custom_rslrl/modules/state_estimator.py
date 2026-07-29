# SPDX-License-Identifier: BSD-3-Clause
# Ported from the official CMoE (Fudan University): rsl_rl/rsl_rl/modules/state_estimator.py
#
# 与官方逐行对齐, 仅做两处"必要的移植修改"(其余结构/损失/更新流程/重参数化全部照搬官方):
#   1) 维度: 官方把 prop 输入硬编码成 450、decoder 输出 45。本移植改为从构造参数读取
#      (prop_input_dim / num_one_step_obs), 由 runner 从观测形状运行期推断。
#      (2026-07-17 起 12-DoF 对齐后恰为 10*45=450 / 45, 与官方相同; 早期 29-DoF 时
#       曾是 50*96=4800 / 96 —— 本注释同步勘误, 代码无需改动。)
#   2) I/O: 官方在 update 内部对一整条 next_critic_obs 做切片取速度与重建目标
#      (next_critic_obs[:, 45:48] 与 [:, 3:48]);Isaac Lab 把观测拆成了 TensorDict 分组,
#      所以这里把"速度目标 vel_target"和"重建目标 recon_target"由外部显式传入(语义不变,
#      仍是"下一步"的目标 —— 由 ppo_cmoe 侧从 next_proprio / next_critic[:, :3] 取出)。
# reparameterize 照搬官方(torch.rand_like)。注:这其实是官方的一个 bug(VAE 重参数化应当用
# 正态 randn_like 而非均匀 rand_like);若想修正, 把下面那一行改成 torch.randn_like 即可。

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class StateEstimator(nn.Module):
    def __init__(self,
                 prop_input_dim,            # = temporal_steps * num_one_step_obs (12-DoF 对齐后: 10*45 = 450)
                 num_one_step_obs,          # = per-frame proprio dim / decoder 输出 (12-DoF 对齐后: 45)
                 prop_enc_hidden_dims=[128, 64, 32],
                 dec_hidden_dims=[32, 64, 128],
                 latent_dim=16,
                 explicit_dim=3,
                 activation='elu',
                 learning_rate=1e-3,
                 max_grad_norm=10.0,
                 kld_weight=0.005,
                 **kwargs):

        self.use_estimation_loss = kwargs.pop("use_estimation_loss", True)
        self.use_latent_loss = kwargs.pop("use_latent_loss", True)
        if kwargs:
            print("StateEstimator.__init__ got unexpected arguments, which will be ignored: "
                  + str([key for key in kwargs.keys()]))

        super(StateEstimator, self).__init__()
        activation = get_activation(activation)

        self.num_one_step_obs = num_one_step_obs
        self.num_latent = prop_enc_hidden_dims[-1]
        self.max_grad_norm = max_grad_norm
        self.kld_weight = kld_weight
        self.latent_dim = latent_dim

        # Proprioceptive MLP Encoder
        prop_enc_input_dim = prop_input_dim
        prop_enc_layers = []
        for l in range(len(prop_enc_hidden_dims)):
            prop_enc_layers += [nn.Linear(prop_enc_input_dim, prop_enc_hidden_dims[l]), activation]
            prop_enc_input_dim = prop_enc_hidden_dims[l]
        self.encoder = nn.Sequential(*prop_enc_layers)

        self.fc_mu = nn.Linear(prop_enc_input_dim, latent_dim)        # \mu: latent mean
        self.fc_var = nn.Linear(prop_enc_input_dim, latent_dim)       # 2*log(\sigma): latent log-var
        self.fc_explicit = nn.Linear(prop_enc_input_dim, explicit_dim)  # explicit (body velocity)

        # Decoder
        dec_input_dim = latent_dim + explicit_dim                    # 16 + 3 = 19
        dec_output_dim = self.num_one_step_obs                       # 45
        dec_layers = []
        for l in range(len(dec_hidden_dims)):
            dec_layers += [nn.Linear(dec_input_dim, dec_hidden_dims[l]), activation]
            dec_input_dim = dec_hidden_dims[l]
        dec_layers += [nn.Linear(dec_input_dim, dec_output_dim)]
        self.decoder = nn.Sequential(*dec_layers)

        # Optimizer (独立 Adam, lr 跟随 PPO)
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def get_latent(self, obs_history):
        explicit, z, mu, log_var = self.encode(obs_history)
        return explicit.detach(), z.detach()

    def forward(self, obs_history):
        result = self.encoder(obs_history.detach())
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        log_var = torch.clamp(log_var, -10.0, 10.0)
        explicit = self.fc_explicit(result)
        if self.training:
            z = self.reparameterize(mu, log_var)
        else:
            z = mu
        return explicit.detach(), z.detach()

    def encode(self, obs_history):
        result = self.encoder(obs_history.detach())
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        # 数值防线: log_var 无界时 exp(log_var) 会溢出 —— 本次训练 iter2218 KL 冲到
        # 1.19e6, 两个迭代后权重被污染, 动作输出 NaN -> 环境观测 NaN -> check_nan 崩溃。
        # ±10 覆盖 e^±10 的方差范围, 远超正常工作区, 只截断发散, 不影响正常学习。
        log_var = torch.clamp(log_var, -10.0, 10.0)
        explicit = self.fc_explicit(result)
        if self.training:
            z = self.reparameterize(mu, log_var)
        else:
            z = mu
        return explicit, z, mu, log_var

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.rand_like(std)   # 照搬官方(注:官方用均匀 rand_like, 实为 bug;改 randn_like 可修正)
        return eps * std + mu

    def update(self, obs_history, vel_target, recon_target, lr=None):
        # 对齐官方 StateEstimator.update:
        #   官方: explicit = next_critic_obs[:, 45:48];  next_obs = next_critic_obs[:, 3:48]
        #   这里: explicit = vel_target(下一步 base_lin_vel); next_obs = recon_target(下一步 proprio)
        if lr is not None:
            self.learning_rate = lr
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.learning_rate

        explicit = vel_target.detach()
        next_obs = recon_target.detach()

        pred_explicit, z, mu, log_var = self.encode(obs_history)
        z = torch.cat((z, pred_explicit), dim=1)
        pred_next_obs = self.decoder(z)

        recons_loss = F.mse_loss(pred_next_obs, next_obs)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        vae_loss = recons_loss + self.kld_weight * kld_loss

        estimation_loss = F.mse_loss(pred_explicit, explicit)
        losses = self.use_estimation_loss * estimation_loss + self.use_latent_loss * vae_loss

        self.optimizer.zero_grad()
        # 数值防线: 物理爆炸批次(|q̇| 极大)会给出 inf/NaN 损失; 反传会当场污染 Adam
        # 动量与权重, 之后每次前向都是 NaN。损失或梯度范数非有限时跳过本次更新。
        if not torch.isfinite(losses):
            return estimation_loss.item(), vae_loss.item(), recons_loss.item(), kld_loss.item()
        losses.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        if torch.isfinite(grad_norm):
            self.optimizer.step()

        return estimation_loss.item(), vae_loss.item(), recons_loss.item(), kld_loss.item()


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "silu":
        return nn.SiLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
