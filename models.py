import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical

class ChessNet(nn.Module):
    """中国象棋的神经网络模型"""
    def __init__(self, input_shape=(10, 9), n_actions=8100):
        super(ChessNet, self).__init__()
        
        # 检查是否有可用的GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 输入形状为(10, 9)的棋盘状态 + 1个玩家通道 = 2个输入通道
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        # 计算展平后的特征维度
        self.flatten_size = 256 * input_shape[0] * input_shape[1]
        
        # 共享特征提取层
        self.shared = nn.Sequential(
            nn.Linear(self.flatten_size, 1024),
            nn.Mish(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.Mish(),
            nn.Dropout(0.1)
        )
        
        # 策略头（actor）
        self.actor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )
        
        # 价值头（critic）
        self.critic = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # 将模型移动到指定设备
        self.to(self.device)
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x, action_mask=None, players=None):
        # 确保输入在正确的设备上
        x = x.to(self.device)
        
        # 确保输入形状正确 (batch_size, height, width) -> (batch_size, 1, height, width)
        if len(x.shape) == 3:
            x = x.unsqueeze(1)  # 添加通道维度
        
        # 创建玩家通道
        if players is not None:
            # 确保 players 是 tensor 并在正确设备上
            if not torch.is_tensor(players):
                players = torch.LongTensor(players).to(self.device)
            # 将玩家 ID (0 或 1) 扩展为与棋盘状态相同空间大小的通道
            # 创建一个全零或全一张量，形状为 (batch_size, 1, 10, 9)
            batch_size = x.size(0)
            player_channel = torch.zeros(batch_size, 1, 10, 9, device=self.device, dtype=torch.float32)
            # 根据 players 的值填充 player_channel
            # players 是 (batch_size,) 形状
            for i in range(batch_size):
                 player_channel[i, 0, :, :] = players[i].float() # 将玩家ID广播到整个10x9切片

            # 将玩家通道与棋盘状态通道拼接
            x = torch.cat((x, player_channel), dim=1)
        
        # 卷积层
        x = F.mish(self.bn1(self.conv1(x)))
        x = F.mish(self.bn2(self.conv2(x)))
        x = F.mish(self.bn3(self.conv3(x)))
        
        # 展平
        x = x.view(-1, self.flatten_size)
        
        # 共享特征
        x = self.shared(x)
        
        # 策略和价值
        action_logits = self.actor(x)
        state_value = self.critic(x)
        
        # 应用行动遮罩
        if action_mask is not None:
            action_mask = action_mask.to(self.device)
            if action_logits.shape != action_mask.shape:
                print(f"警告: ChessNet.forward 中 action_logits shape {action_logits.shape} 与 action_mask shape {action_mask.shape} 不匹配!")
                # 尝试调整action_mask，但这非常危险，应该修复环境
                if action_logits.shape[0] == action_mask.shape[0] and action_mask.shape[1] < self.n_actions:
                    print(f"警告: action_mask 列数过少，尝试补齐。")
                    padding = torch.zeros((action_mask.shape[0], self.n_actions - action_mask.shape[1]), device=self.device, dtype=action_mask.dtype)
                    action_mask = torch.cat((action_mask, padding), dim=1)
                elif action_logits.shape[0] == action_mask.shape[0] and action_mask.shape[1] > self.n_actions:
                    print(f"警告: action_mask 列数过多，尝试截断。")
                    action_mask = action_mask[:, :self.n_actions]
                # 如果仍然不匹配，则不使用mask，打印错误
                if action_logits.shape != action_mask.shape:
                    print(f"错误: ChessNet.forward 中 action_mask 形状调整失败，将不使用掩码。")
                    action_mask = None # Fallback to no mask
            
            if action_mask is not None:
                 action_logits = action_logits.masked_fill(action_mask == 0, -1e9) # 使用较大的负值以确保极小概率
        
        # 使用log_softmax而不是softmax，提高数值稳定性
        action_log_probs = F.log_softmax(action_logits, dim=-1)
        
        return action_log_probs, state_value

class ChessPPO:
    """PPO算法的实现"""
    def __init__(self, model, lr=1e-4, gamma=0.99, epsilon=0.1, value_coef=0.5, entropy_coef=0.01, max_grad_norm=0.5):
        self.model = model
        self.device = model.device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-5)
        self.gamma = gamma
        self.epsilon_initial = epsilon
        self.epsilon = epsilon
        self.epsilon_decay = 0.995
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.kl_target = 0.02 # 调整KL目标
    
    def compute_gae(self, rewards, values, next_value, dones):
        """计算广义优势估计（GAE）"""
        advantages = []
        gae = 0
        lambda_ = 0.95  # GAE参数
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value_t = next_value
            else:
                next_value_t = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value_t * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * lambda_ * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        advantages_tensor = torch.tensor(advantages, device=self.device, dtype=torch.float32)
        # 裁剪极端值，防止数值不稳定
        advantages_tensor = torch.clamp(advantages_tensor, -10.0, 15.0)
        return advantages_tensor
    
    def update(self, states, actions, old_log_probs_batch, batch_returns, batch_advantages, action_masks_batch=None, batch_players=None):
        """更新策略和价值网络"""
        # 转换为张量并移动到正确的设备
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        old_log_probs_batch = torch.FloatTensor(old_log_probs_batch).to(self.device)
        # 确保传入的 returns 和 advantages 是张量并在正确设备上
        batch_returns = torch.FloatTensor(batch_returns).to(self.device)
        batch_advantages = torch.FloatTensor(batch_advantages).to(self.device)
        # 确保传入的 players 是张量并在正确设备上
        if batch_players is not None:
            batch_players = torch.LongTensor(batch_players).to(self.device)

        # 修复：确保 action_masks_batch 是 tensor 并在正确的 device 上
        if action_masks_batch is not None and not torch.is_tensor(action_masks_batch):
            action_masks_batch = torch.FloatTensor(action_masks_batch).to(self.device)

        # 计算当前策略的动作概率和状态价值
        # Note: 如果 ChessNet forward 需要 batch_players，这里需要修改调用
        try:
            # 示例：如果模型 forward 改为 model(x, action_mask=None, players=None)
            # log_probs, values = self.model(states, action_masks_batch, batch_players)
            # 修复：调用模型时传递 batch_players
            log_probs, values = self.model(states, action_mask=action_masks_batch, players=batch_players)
        except Exception as model_e:
            print(f"错误: 模型前向传播失败: {model_e}")
            # 如果模型调用失败，无法继续，返回0损失
            return {
                'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0,
                'kl_div': 0.0, 'epsilon': self.epsilon
            }
        
        # 计算优势（使用传入的 batch_advantages）
        advantages = batch_advantages # 直接使用计算好的优势
        
        # 归一化优势 (在计算GAE后对整个 episode 序列归一化更标准)
        # 为了与原代码相似，保留这里的归一化，但应用于传入的 batch
        if len(advantages) > 1:
            # 使用传入的batch_advantages进行归一化
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            # 裁剪极端值
            advantages = torch.clamp(advantages, -5.0, 5.0)
        
        # 计算returns，用于价值损失 (使用传入的 batch_returns)
        returns = batch_returns # 直接使用计算好的回报
        
        # 使用对数概率计算比率（更稳定）
        values = values.squeeze(-1)  # 确保是1D张量
        new_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 限制新旧概率的差距，防止KL散度过大
        log_ratio = new_log_probs - old_log_probs_batch
        # log_ratio = torch.clamp(log_ratio, -2.0, 2.0) # 限制log_ratio防止极端值
        ratio = torch.exp(log_ratio)
        
        # 裁剪比率，防止过大更新
        ratio_clipped = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon)
        
        # 计算裁剪的策略损失，使用传入的 batch_advantages (现在是 advantages 变量)
        surr1 = ratio * advantages
        surr2 = ratio_clipped * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # 计算价值损失，使用传入的 batch_returns (现在是 returns 变量)
        value_loss = F.mse_loss(values, returns)
        
        # 安全计算熵
        entropy = torch.tensor(0.0, device=self.device)
        try:
            probs = torch.exp(log_probs) # log_probs shape: (batch_size, n_actions)
            if action_masks_batch is not None:
                masked_probs = probs * action_masks_batch.float()
            else:
                masked_probs = probs
            
            row_entropies_list = []
            for i in range(masked_probs.shape[0]):
                row_prob_sum = masked_probs[i].sum()
                if row_prob_sum > 1e-8:
                    normalized_row_probs = masked_probs[i] / row_prob_sum
                    # clamp to avoid log(0)
                    normalized_row_probs = torch.clamp(normalized_row_probs, min=1e-10)
                    # Re-normalize after clamp
                    normalized_row_probs = normalized_row_probs / normalized_row_probs.sum()
                    dist = Categorical(probs=normalized_row_probs)
                    row_entropies_list.append(dist.entropy())
                else:
                    row_entropies_list.append(torch.tensor(0.0, device=self.device))
            
            if len(row_entropies_list) > 0:
                entropy = torch.stack(row_entropies_list).mean()
            else:
                entropy = torch.tensor(0.0, device=self.device)

        except Exception as e_entropy:
            print(f"警告：熵计算出错 ({type(e_entropy).__name__}): {e_entropy}")
            entropy = torch.tensor(0.01, device=self.device) # Fallback
        
        # 计算KL散度，用于监控
        kl_div = torch.mean(old_log_probs_batch - new_log_probs).item()
        if abs(kl_div) > self.kl_target * 1.5: # 如果KL散度过大，减小epsilon
            self.epsilon = max(0.05, self.epsilon * 0.9)
        elif abs(kl_div) < self.kl_target * 0.5: # 如果KL散度过小，增大epsilon
            self.epsilon = min(self.epsilon_initial, self.epsilon * 1.1)
        
        # 总损失 - 结合策略损失、价值损失和熵正则化
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
        
        # 检查损失是否为NaN
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"警告：损失为NaN/inf。Policy: {policy_loss.item()}, Value: {value_loss.item()}, Entropy: {entropy.item()}")
            return {
                'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0,
                'kl_div': kl_div, 'epsilon': self.epsilon
            }
        
        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        
        # 检查梯度是否包含NaN/inf
        found_bad_grad = False
        for param in self.model.parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    print(f"警告：参数 {param.shape} 的梯度中包含NaN/inf，跳过此次更新")
                    found_bad_grad = True
                    break
        if not found_bad_grad:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm) # 裁剪有效梯度
            self.optimizer.step()
        else:
            self.optimizer.zero_grad() # 清除坏梯度
        
        # 衰减epsilon，逐渐减小裁剪范围
        self.epsilon = max(0.05, self.epsilon * self.epsilon_decay)
        
        return {
            'policy_loss': policy_loss.item(), 'value_loss': value_loss.item(), 'entropy': entropy.item(),
            'kl_div': kl_div, 'epsilon': self.epsilon
        }

    def select_action(self, state, action_mask, player):
        """根据当前状态、合法动作掩码和玩家选择动作（用于推理）"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_mask_tensor = torch.FloatTensor(action_mask).unsqueeze(0).to(self.device)
        player_tensor = torch.LongTensor([player]).to(self.device)

        with torch.no_grad():
            # 调用模型 forward，传递玩家信息和 action_mask
            action_log_probs, value = self.model(state_tensor, action_mask_tensor, players=player_tensor)

        # 从对数概率中获取概率分布
        action_probs = torch.exp(action_log_probs).squeeze(0) # 移除批次维度

        # 应用合法动作掩码
        action_probs = action_probs * action_mask_tensor.squeeze(0)

        # 归一化概率分布（确保合法动作的概率之和为1）
        action_probs = action_probs / (action_probs.sum() + 1e-8)

        # 使用 Categorical 分布进行采样
        dist = Categorical(action_probs)
        action = dist.sample().item()

        # 返回选择的动作
        return action