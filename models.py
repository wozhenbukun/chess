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
        
        # 输入形状为(10, 9)的棋盘状态
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
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
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
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
    
    def forward(self, x, action_mask=None):
        # 确保输入在正确的设备上
        x = x.to(self.device)
        
        # 确保输入形状正确
        if len(x.shape) == 3:
            x = x.unsqueeze(1)  # 添加通道维度
        
        # 卷积层
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
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
        self.kl_target = 0.015 # 调整KL目标
    
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
        advantages_tensor = torch.clamp(advantages_tensor, -10.0, 10.0)
        return advantages_tensor
    
    def update(self, states, actions, old_log_probs_batch, rewards, dones, action_masks_batch=None):
        """更新策略和价值网络"""
        # 转换为张量并移动到正确的设备
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        old_log_probs_batch = torch.FloatTensor(old_log_probs_batch).to(self.device)
        rewards = torch.clamp(torch.FloatTensor(rewards).to(self.device), -10.0, 10.0)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # 防止奖励值过大
        rewards = torch.clamp(rewards, -10.0, 10.0)
        
        # 计算当前策略的动作概率和状态价值
        if action_masks_batch is None:
            # 如果没有提供遮罩，创建一个只允许已选动作的遮罩
            batch_size = states.size(0)
            action_masks_batch = torch.zeros((batch_size, self.model.n_actions), device=self.device)
            action_masks_batch.scatter_(1, actions.unsqueeze(1), 1)  # 只标记所选动作为合法
        
        # 获取当前策略的log概率和价值
        try:
            log_probs, values = self.model(states, action_masks_batch)
        except Exception as model_e:
            print(f"错误: 模型前向传播失败: {model_e}")
            # 如果模型调用失败，无法继续，返回0损失
            return {
                'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0,
                'kl_div': 0.0, 'epsilon': self.epsilon
            }
        
        # 计算优势（raw advantages）
        values = values.squeeze(-1)  # 确保是1D张量
        next_value = torch.zeros(1, device=self.device)  # 假设最后一个状态的价值为0
        advantages = self.compute_gae(rewards, values.detach(), next_value, dones)
        
        # 计算returns，用于价值损失
        returns = advantages + values.detach()
        
        # 归一化优势，用于策略损失，并裁剪极端值
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            # 裁剪极端值，减少方差
            advantages = torch.clamp(advantages, -5.0, 5.0)
        
        # 使用对数概率计算比率（更稳定）
        new_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 限制新旧概率的差距，防止KL散度过大
        log_ratio = new_log_probs - old_log_probs_batch
        # log_ratio = torch.clamp(log_ratio, -2.0, 2.0) # 限制log_ratio防止极端值
        ratio = torch.exp(log_ratio)
        
        # 裁剪比率，防止过大更新
        ratio_clipped = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon)
        
        # 计算裁剪的策略损失
        surr1 = ratio * advantages
        surr2 = ratio_clipped * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # 计算价值损失，使用returns
        value_loss = F.mse_loss(values, returns)
        
        # 安全计算熵
        entropy = torch.tensor(0.0, device=self.device)
        try:
            probs = torch.exp(log_probs) # log_probs shape: (batch_size, n_actions)
            if action_masks_batch is not None:
                # 确保action_masks_batch是浮点型以便乘法
                masked_probs = probs * action_masks_batch.float() 
            else:
                masked_probs = probs
            
            # 归一化概率
            prob_sum = masked_probs.sum(dim=1, keepdim=True)
            # 对 prob_sum 小于 epsilon 的行，其概率设为均匀分布（在有效动作上）或整体均匀
            # 避免除以0或极小值导致 NaN
            # uniform_probs = torch.ones_like(masked_probs) * (1.0 / self.model.n_actions)
            # if action_masks_batch is not None:
            #     num_legal_actions = action_masks_batch.sum(dim=1, keepdim=True)
            #     uniform_probs = action_masks_batch.float() / torch.max(torch.tensor(1.0, device=self.device), num_legal_actions)
                
            # safe_probs = torch.where(prob_sum > 1e-8, masked_probs / (prob_sum + 1e-10), uniform_probs)
            
            # 更安全的归一化：如果和为0，则该行熵为0
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
                    # 如果所有动作概率和接近0 (例如所有动作都被mask或原始概率极小)
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