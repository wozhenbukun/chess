import torch
import numpy as np
from chess_env import ChineseChessEnv
from models import ChessNet, ChessPPO
import os
from datetime import datetime
from collections import deque
import torch.nn.functional as F
import json

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"全局使用设备: {device}")

class ExperienceBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done, old_prob, action_mask):
        self.buffer.append((state, action, reward, next_state, done, old_prob, action_mask))
    
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]
    
    def __len__(self):
        return len(self.buffer)

def train(env, agent, n_episodes=100, max_steps=200, save_interval=100, batch_size=256, run_name_prefix="run"):
    """训练函数"""
    # 创建本次运行的唯一目录
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs", f"{run_name_prefix}_{current_time}")
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    print(f"训练数据将保存在: {run_dir}")

    best_model_path = os.path.join(run_dir, "best_model.pth")
    latest_model_path = os.path.join(run_dir, "latest_model.pth")
    metrics_path = os.path.join(run_dir, "metrics.json")

    agent.model = agent.model.to(device)
    # agent.device = device # agent内部应该已经有device了，或者从model获取

    buffer = ExperienceBuffer(capacity=50000)
    
    episode_rewards_deque = deque(maxlen=10) # 用于计算最近10个episode的平均奖励
    best_reward = float('-inf')
    
    exploration_epsilon = 1.0
    epsilon_min = 0.2 
    epsilon_decay = 0.999 
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        agent.optimizer, 
        mode='max', 
        factor=0.5, 
        patience=30, 
        verbose=True
    )
    
    agent.entropy_coef = 0.05 
    reward_scale = 0.1 
    
    running_reward_std = 1.0 # 用于奖励归一化
    reward_normalization_factor = 0.99
    
    kl_history = []
    ppo_epochs = 4 
    
    consecutive_nan_count = 0
    max_nan_threshold = 5

    # 初始化指标记录列表
    metrics_history = {
        "episodes": [],
        "avg_rewards": [],
        "policy_losses": [],
        "value_losses": [],
        "entropies": [],
        "kl_divs": [],
        "exploration_epsilons": [],
        "ppo_clip_epsilons": [],
        "learning_rates": []
    }
    
    for episode in range(n_episodes):
        state = env.reset()
        current_episode_reward = 0
        
        for step in range(max_steps):
            valid_actions, action_mask = env.get_valid_actions()
            if not valid_actions:
                # print(f"Episode {episode+1}: No valid actions found at step {step}!")
                break
            
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                # action_mask是numpy array, agent.model会处理
                action_log_probs, _ = agent.model(state_tensor, torch.tensor(action_mask, dtype=torch.float32).unsqueeze(0)) 
            
            if np.random.random() < exploration_epsilon:
                action = np.random.choice(valid_actions)
            else:
                # 从log_probs转换为probs进行采样
                action_probs_dist = torch.exp(action_log_probs.squeeze(0))
                # 确保概率和为1，处理可能的数值问题
                action_probs_dist = action_probs_dist / (action_probs_dist.sum() + 1e-9)
                action = torch.multinomial(action_probs_dist, 1).item()
            
            next_state, reward, done, _ = env.step(action)
            
            # 奖励处理
            reward_clipped = np.clip(reward, -10.0, 10.0) 
            scaled_reward = reward_clipped * reward_scale
            
            # 更新运行标准差的估计 (使用Welford算法思想简化版)
            # 这里简化为只更新一个大致的running_std，更精确的需要保存所有奖励
            # 或者我们可以直接使用一个固定的较小值，或者完全移除这一层归一化，因为优势函数已归一化
            # 简单起见，暂时移除基于running_reward_std的归一化，因为advantages已经归一化了
            # if running_reward_std > 1e-8:
            #    normalized_reward = scaled_reward / (running_reward_std + 1e-8)
            # else:
            #    normalized_reward = scaled_reward
            normalized_reward = scaled_reward # 直接使用缩放后的奖励
            final_reward = np.clip(normalized_reward, -3.0, 3.0)
            
            # 保存的是原始log_probs，而不是采样用的probs
            buffer.add(state, action, final_reward, next_state, done, action_log_probs[0, action].item(), action_mask)
            state = next_state
            current_episode_reward += reward # 记录原始未缩放未归一化的奖励总和
            
            if done:
                break
        
        exploration_epsilon = max(epsilon_min, exploration_epsilon * epsilon_decay)
        episode_rewards_deque.append(current_episode_reward) # 添加本episode的总原始奖励
        
        if len(buffer) >= batch_size:
            avg_policy_loss_epoch = 0
            avg_value_loss_epoch = 0
            avg_entropy_epoch = 0
            avg_kl_div_epoch = 0
            avg_ppo_clip_epoch = 0
            updates_done_in_episode = 0
            
            for _ in range(ppo_epochs):
                try:
                    batch = buffer.sample(batch_size)
                    batch_states = np.array([x[0] for x in batch])
                    batch_actions = np.array([x[1] for x in batch])
                    batch_rewards_processed = np.array([x[2] for x in batch]) # 使用buffer中处理过的奖励
                    batch_dones = np.array([x[4] for x in batch])
                    batch_old_log_probs = np.array([x[5] for x in batch])
                    batch_masks = np.array([x[6] for x in batch])
                    
                    metrics = agent.update(
                        batch_states,
                        batch_actions,
                        batch_old_log_probs,
                        batch_rewards_processed, # 传递处理后的奖励
                        batch_dones,
                        batch_masks
                    )
                    
                    if np.isnan(metrics['policy_loss']) or np.isnan(metrics['value_loss']):
                        print(f"警告：更新{updates_done_in_episode+1}/{ppo_epochs} 产生NaN，跳过")
                        consecutive_nan_count += 1
                        if consecutive_nan_count >= max_nan_threshold:
                            print("连续NaN过多，降低学习率")
                            for param_group in agent.optimizer.param_groups:
                                param_group['lr'] = max(param_group['lr'] * 0.5, 1e-6) # 避免学习率过低
                            consecutive_nan_count = 0
                        continue
                    
                    updates_done_in_episode += 1
                    consecutive_nan_count = 0
                    
                    avg_policy_loss_epoch += metrics['policy_loss']
                    avg_value_loss_epoch += metrics['value_loss']
                    avg_entropy_epoch += metrics['entropy']
                    avg_kl_div_epoch += metrics['kl_div']
                    avg_ppo_clip_epoch += metrics['epsilon'] # PPO的裁剪epsilon
                except Exception as e_update:
                    print(f"PPO更新过程中出错: {e_update}")
                    continue # 跳过此mini-batch的更新
            
            if updates_done_in_episode > 0:
                avg_policy_loss_epoch /= updates_done_in_episode
                avg_value_loss_epoch /= updates_done_in_episode
                avg_entropy_epoch /= updates_done_in_episode
                avg_kl_div_epoch /= updates_done_in_episode
                avg_ppo_clip_epoch /= updates_done_in_episode
            else:
                print("本轮PPO所有更新均失败或跳过")
        
        avg_reward_print = np.mean(episode_rewards_deque) if len(episode_rewards_deque) > 0 else 0.0
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}, Avg Reward (raw): {avg_reward_print:.2f}, Exp Eps: {exploration_epsilon:.3f}, LR: {agent.optimizer.param_groups[0]['lr']:.6f}")
            if len(buffer) >= batch_size and updates_done_in_episode > 0:
                print(f"  Losses: Policy: {avg_policy_loss_epoch:.4f}, Value: {avg_value_loss_epoch:.4f}")
                print(f"  Metrics: Entropy: {avg_entropy_epoch:.4f}, KL: {avg_kl_div_epoch:.4f}, PPO Clip: {avg_ppo_clip_epoch:.3f}")
                
                # 记录指标
                metrics_history["episodes"].append(episode + 1)
                metrics_history["avg_rewards"].append(avg_reward_print)
                metrics_history["policy_losses"].append(avg_policy_loss_epoch)
                metrics_history["value_losses"].append(avg_value_loss_epoch)
                metrics_history["entropies"].append(avg_entropy_epoch)
                metrics_history["kl_divs"].append(avg_kl_div_epoch)
                metrics_history["exploration_epsilons"].append(exploration_epsilon)
                metrics_history["ppo_clip_epsilons"].append(avg_ppo_clip_epoch)
                metrics_history["learning_rates"].append(agent.optimizer.param_groups[0]['lr'])

            scheduler.step(avg_reward_print) # 使用原始奖励的平均值来调整学习率
            
            if avg_reward_print > best_reward:
                best_reward = avg_reward_print
                print(f"新最佳平均奖励: {best_reward:.2f}，保存模型到 {best_model_path}")
                torch.save({
                    'model_state_dict': agent.model.state_dict(),
                    'optimizer_state_dict': agent.optimizer.state_dict(),
                    'episode': episode + 1,
                    'best_reward': best_reward,
                    'exploration_epsilon': exploration_epsilon
                }, best_model_path)
        
        if (episode + 1) % save_interval == 0:
            print(f"定期保存最新模型到 {latest_model_path}")
            torch.save({
                'model_state_dict': agent.model.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'episode': episode + 1,
                'current_avg_reward': avg_reward_print,
                'exploration_epsilon': exploration_epsilon
            }, latest_model_path)
            # 同时保存一次metrics
            with open(metrics_path, 'w') as f:
                json.dump(metrics_history, f, indent=4)

    # 训练结束，最后保存一次metrics
    print("训练完成。正在保存最终metrics...")
    with open(metrics_path, 'w') as f:
        json.dump(metrics_history, f, indent=4)
    print(f"Metrics已保存到 {metrics_path}")

def main():
    env = ChineseChessEnv()
    model = ChessNet() # 默认参数，n_actions=8100
    agent = ChessPPO(model, lr=1e-4, entropy_coef=0.05) # lr 和 entropy_coef 和 train 里匹配
    
    try:
        train(env, agent, n_episodes=20000, save_interval=100, batch_size=256)
    except KeyboardInterrupt:
        print("训练被用户中断。")
    finally:
        env.close()
        print("环境已关闭。")

if __name__ == "__main__":
    main() 