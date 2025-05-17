import torch
import numpy as np
from chess_env import ChineseChessEnv
from models import ChessNet, ChessPPO
import os
from datetime import datetime
from collections import deque
import torch.nn.functional as F
import json
from torch.distributions import Categorical
import chess
import chess.engine # 导入 chess.engine 模块
import sys # 导入 sys 模块用于处理引擎进程
import subprocess
import threading
import re
import time

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ExperienceBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, old_log_prob, returns, advantage, action_mask, current_player):
        # 将 numpy 数组转换为列表存储，避免内存问题，并在 sample 时再转换为 tensor/numpy
        self.buffer.append((state.tolist() if isinstance(state, np.ndarray) else state,\
                           action,\
                           old_log_prob,\
                           returns,\
                           advantage,\
                           action_mask.tolist() if isinstance(action_mask, np.ndarray) else action_mask,\
                           current_player))
    
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        # 返回包含计算好的回报、优势和玩家信息的批次
        batch = [self.buffer[i] for i in indices]

        # 解压批次数据，确保顺序和 add 时一致
        states = np.array([x[0] for x in batch])
        actions = np.array([x[1] for x in batch])
        old_log_probs = np.array([x[2] for x in batch])
        returns = np.array([x[3] for x in batch])
        advantages = np.array([x[4] for x in batch])
        action_masks = np.array([x[5] for x in batch])
        players = np.array([x[6] for x in batch]) # 玩家信息

        return states, actions, old_log_probs, returns, advantages, action_masks, players
    
    def __len__(self):
        return len(self.buffer)

def train(env, agent, n_episodes=100, max_steps=200, save_interval=100, batch_size=256, run_name_prefix="run", engine_play_prob=0.2): # 添加 engine_play_prob 参数
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
        patience=30
    )
    
    agent.entropy_coef = 0.05 
    reward_scale = 0.1 
    
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
    
    # --- 集成 Pikafish 引擎 ---
    engine_path = os.path.join(".", "Pikafish", "pikafish-bmi2.exe")
    if os.path.exists(engine_path):
        engine_path = os.path.abspath(engine_path)
    else:
        # 尝试当前目录
        alt_path = os.path.join("Pikafish", "pikafish-bmi2.exe")
        if os.path.exists(alt_path):
            engine_path = os.path.abspath(alt_path)
        else:
            print(f"警告: 未能找到引擎文件。已尝试的路径:")
            print(f"1. {os.path.abspath(engine_path)}")
            print(f"2. {os.path.abspath(alt_path)}")
            print(f"将只进行自博弈训练，不会使用引擎。")
            engine_proc = None
            engine_path = None
    
    if engine_path:
        engine_dir = os.path.dirname(engine_path)
        engine_proc = None
        
        # 检查NNUE文件是否存在
        nnue_path = os.path.join(engine_dir, "pikafish.nnue")
        nnue_download_url = "https://github.com/official-pikafish/Networks/releases/download/master-net/pikafish.nnue"
        
        if not os.path.exists(nnue_path):
            print(f"警告: NNUE文件不存在: {nnue_path}")
            print(f"请从官方网站下载正确的NNUE文件并放置到引擎目录中:")
            print(f"下载地址: {nnue_download_url}")
            print(f"保存位置: {nnue_path}")
            print(f"将只进行自博弈训练，不会使用引擎。")
            engine_proc = None
        else:
            nnue_size_mb = os.path.getsize(nnue_path) / (1024*1024)
            print(f"NNUE文件已找到，大小: {nnue_size_mb:.2f} MB")
            
            if nnue_size_mb < 10:
                print(f"警告: NNUE文件大小异常（{nnue_size_mb:.2f}MB），正常大小应为30-50MB")
                print(f"当前文件可能不是有效的神经网络权重文件")
                print(f"请从官方网站重新下载: {nnue_download_url}")
                print(f"将只进行自博弈训练。")
                engine_proc = None
            else:
                # Windows下防止弹窗
                CREATE_NO_WINDOW = 0x08000000 if sys.platform == 'win32' else 0
                
                # 实时打印引擎stderr
                def print_engine_stderr(proc):
                    for line in proc.stderr:
                        print("[引擎stderr]", line.strip())
                
                def send_cmd(cmd):
                    if engine_proc is None or engine_proc.poll() is not None:
                        raise RuntimeError("引擎进程已退出")
                    try:
                        engine_proc.stdin.write(cmd + '\n')
                        engine_proc.stdin.flush()
                    except Exception as e:
                        print(f"写入引擎命令失败: {e}")
                        raise
                
                def read_until(keyword, timeout=5):
                    lines = []
                    start_time = time.time()
                    while time.time() - start_time < timeout:
                        if engine_proc is None or engine_proc.poll() is not None:
                            print(f"引擎进程已退出，返回码: {engine_proc.returncode if engine_proc else 'Unknown'}")
                            break
                        try:
                            line = engine_proc.stdout.readline()
                            if not line:
                                break
                            line = line.strip()
                            lines.append(line)
                            if keyword in line:
                                break
                        except Exception as e:
                            print(f"读取引擎输出失败: {e}")
                            break
                    return lines
                
                try:
                    print(f"尝试启动引擎: {engine_path}")
                    engine_proc = subprocess.Popen(
                        [engine_path],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        encoding='utf-8',
                        cwd=engine_dir,
                        creationflags=CREATE_NO_WINDOW
                    )
                    threading.Thread(target=print_engine_stderr, args=(engine_proc,), daemon=True).start()
                    
                    send_cmd('uci')
                    read_until('uciok')
                    
                    # 重要: 确保设置正确的NNUE文件路径 - 只使用文件名
                    send_cmd(f'setoption name EvalFile value pikafish.nnue')
                    
                    send_cmd('isready')
                    response = read_until('readyok')
                    
                    # 检查是否有错误消息
                    error_found = False
                    for line in response:
                        if "ERROR" in line or "error" in line:
                            print(f"引擎报告错误: {line}")
                            error_found = True
                    
                    if error_found:
                        print("引擎报告错误，很可能是NNUE文件不兼容")
                        print(f"请从官方网站下载正确的NNUE文件: {nnue_download_url}")
                        print("引擎将关闭，训练将仅使用自博弈模式")
                        try:
                            send_cmd('quit')
                        except:
                            pass
                        if engine_proc is not None and engine_proc.poll() is None:
                            engine_proc.terminate()
                        engine_proc = None
                    else:
                        send_cmd('ucinewgame')
                        send_cmd('isready')
                        read_until('readyok')
                        print("引擎启动成功!")
                except Exception as e:
                    print(f"无法启动引擎 {engine_path}: {e}")
                    print("将只进行自博弈训练。")
                    if engine_proc is not None and engine_proc.poll() is None:
                        try:
                            engine_proc.terminate()
                        except:
                            pass
                    engine_proc = None

    # 在训练结束时关闭引擎
    # 使用 try...finally 块确保无论是否发生错误都能关闭引擎
    try:
        for episode in range(n_episodes):
            # 随机决定本局中哪个玩家由 PPO 模型控制（0=红方,1=黑方）
            model_player = np.random.choice([0, 1])
            engine_player = 1 - model_player
            print(f"Episode {episode+1}: PPO 控制 {'红方' if model_player==0 else '黑方'}, 引擎控制 {'黑方' if model_player==0 else '红方'}")

            state = env.reset()
            current_episode_reward = 0
            episode_experiences = [] # 临时存储当前episode的原始经验
            
            for step in range(max_steps):
                #达到最大步数，打印当前步数
                if step >= max_steps:
                    print(f"Episode {episode+1}: 达到最大步数 {max_steps}!")
                
                valid_actions, action_mask = env.get_valid_actions()
                if not valid_actions:
                    # print(f"Episode {episode+1}: No valid actions found at step {step}!")
                    break
                
                # 判断当前回合由谁来走棋
                if env.current_player == model_player:
                    # ------------ PPO 模型走棋 ------------
                    with torch.no_grad():
                        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                        action_mask_tensor = torch.tensor(action_mask, dtype=torch.float32).unsqueeze(0).to(device)
                        player_tensor = torch.LongTensor([model_player]).to(device)
                        action_log_probs, value = agent.model(state_tensor, action_mask_tensor, players=player_tensor)

                    if np.random.random() < exploration_epsilon:
                        action = np.random.choice(valid_actions)
                        num_legal_actions = len(valid_actions)
                        old_log_prob_item = np.log(1.0 / num_legal_actions) if num_legal_actions > 0 else 0.0
                    else:
                        action_probs_dist = torch.exp(action_log_probs.squeeze(0))
                        action_probs_dist = action_probs_dist * action_mask_tensor.squeeze(0)
                        action_probs_dist = action_probs_dist / (action_probs_dist.sum() + 1e-9)

                        if action_probs_dist.sum() < 1e-9:
                            action = np.random.choice(valid_actions)
                            num_legal_actions = len(valid_actions)
                            old_log_prob_item = np.log(1.0 / num_legal_actions) if num_legal_actions > 0 else 0.0
                        else:
                            dist = Categorical(action_probs_dist)
                            action = dist.sample().item()
                            if action not in valid_actions:
                                action = np.random.choice(valid_actions)
                                num_legal_actions = len(valid_actions)
                                old_log_prob_item = np.log(1.0 / num_legal_actions) if num_legal_actions > 0 else 0.0
                            else:
                                old_log_prob_item = action_log_probs[0, action].item()

                    # 执行 PPO 动作
                    next_state, reward, done, _ = env.step(action)

                    # 记录 PPO 经验
                    episode_experiences.append((state.copy(), action, reward, next_state.copy(), done, old_log_prob_item, action_mask.copy(), model_player))

                else:
                    # ------------ 引擎走棋 -------------
                    if engine_proc is None:
                        # 如果引擎不可用，使用随机动作
                        print("引擎不可用，随机选择动作")
                        action = np.random.choice(valid_actions)
                        next_state, reward, done, _ = env.step(action)
                    else:
                        try:
                            fen_state = env.to_fen()
                            send_cmd(f'position fen {fen_state}')
                            send_cmd('go movetime 1000') # 限制思考时间为1秒
                            lines = read_until('bestmove')
                            
                            action_uci = None
                            for line in lines:
                                if line.startswith('bestmove'):
                                    action_uci = line.split()[1]
                                    break
                                    
                            if action_uci is None or action_uci == '(none)':
                                # print("引擎未返回合法招法，随机动作")
                                action = np.random.choice(valid_actions)
                                next_state, reward, done, _ = env.step(action)
                            else:
                                # print(f"引擎返回招法 (UCI): {action_uci}")
                                try:
                                    next_state, reward, done, _ = env.step(action_uci)
                                except ValueError as ve:
                                    # print(f"UCI招法解析失败: {ve}，随机动作")
                                    action = np.random.choice(valid_actions)
                                    next_state, reward, done, _ = env.step(action)
                        except Exception as e:
                            print(f"引擎交互失败，随机动作: {e}")
                            
                            # 检查引擎是否仍在运行
                            if engine_proc is not None and engine_proc.poll() is not None:
                                print(f"注意: 引擎已退出，返回码: {engine_proc.returncode}")
                                print("尝试重启引擎...")
                                try:
                                    # 重新启动引擎
                                    engine_proc = subprocess.Popen(
                                        [engine_path],
                                        stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE,
                                        encoding='utf-8',
                                        cwd=engine_dir,
                                        creationflags=CREATE_NO_WINDOW
                                    )
                                    threading.Thread(target=print_engine_stderr, args=(engine_proc,), daemon=True).start()
                                    
                                    send_cmd('uci')
                                    read_until('uciok')
                                    send_cmd(f'setoption name EvalFile value pikafish.nnue')
                                    send_cmd('isready')
                                    read_until('readyok')
                                    send_cmd('ucinewgame')
                                    print("引擎重启成功!")
                                except Exception as restart_e:
                                    print(f"引擎重启失败: {restart_e}，将不再使用引擎")
                                    engine_proc = None
                            
                            # 无论如何执行随机动作
                            action = np.random.choice(valid_actions)
                            next_state, reward, done, _ = env.step(action)
                
                # 更新 state 与累积奖励
                state = next_state
                current_episode_reward += reward
                
                if done:
                    break # Episode 结束
            
            # Episode 结束或达到max_steps，处理 episode 经验并添加到 buffer
            # 需要从 episode_experiences 中提取信息并计算 GAE/Returns
            
            # 1. 从 episode_experiences 提取原始数据
            if not episode_experiences:
                # print("Episode experiences list is empty.")
                continue # 如果 episode 没有任何经验，跳过
            
            # 提取所有在一个episode中的 state, action, reward, done, old_log_prob, action_mask, player
            states_ep, actions_ep, rewards_ep, next_states_ep, dones_ep, old_log_probs_ep, action_masks_ep, players_ep = zip(*episode_experiences)
            
            # 2. 获取每个状态的价值 V(s)
            # 将所有状态 tensor 化，一次通过模型获取价值，更高效
            states_ep_tensor = torch.FloatTensor(np.array(states_ep)).to(device)
            players_ep_tensor = torch.LongTensor(np.array(players_ep)).to(device) # <-- 添加 players_ep_tensor
            with torch.no_grad():
                # 调用模型 forward，传递玩家信息
                # 注意：在这里计算状态价值 V(s)，不需要 action_mask
                _, values_ep_tensor = agent.model(states_ep_tensor, players=players_ep_tensor) # <-- 传递 players 参数
                values_ep = values_ep_tensor.squeeze(-1).cpu().numpy()

            # 3. 计算最后一个状态的 V(s')
            # 最后一个状态的价值，如果 episode 是自然结束 (done)，则为 0
            # 如果是因为 max_steps 结束，则为模型预测的价值
            # 注意：next_states_ep 现在是 numpy 数组的元组，需要先转换为 numpy 数组再 tensor 化
            last_state_ep_tensor = torch.FloatTensor(np.array(next_states_ep[-1])).unsqueeze(0).to(device)
            last_player_ep_tensor = torch.LongTensor([players_ep[-1]]).to(device) # <-- 添加 last_player_ep_tensor
            with torch.no_grad():
                # 计算最后一个状态的价值 V(s')
                # 同样不需要 action_mask
                _, last_value_ep_tensor = agent.model(last_state_ep_tensor, players=last_player_ep_tensor) # <-- 传递 players 参数

            last_value_ep = last_value_ep_tensor.squeeze(-1).item() if not dones_ep[-1] else 0.0
            
            # 4. 计算 GAE 和 Returns
            # compute_gae 需要 rewards, values, next_value, dones
            # 注意：GAE计算是在一个连续序列上，这里episode_experiences就是连续的
            advantages_ep_tensor = agent.compute_gae(list(rewards_ep), list(values_ep), last_value_ep, list(dones_ep)).cpu()
            returns_ep_tensor = advantages_ep_tensor + torch.FloatTensor(values_ep) # 回报 = 优势 + 价值
            
            advantages_ep = advantages_ep_tensor.numpy()
            returns_ep = returns_ep_tensor.numpy()
            
            # 5. 将计算好的经验添加到 Buffer
            # 优势归一化可以在这里或在 update 方法中进行
            # 为了保持和原代码类似，先不在 buffer.add 前归一化优势
            # 这里不再按玩家分离，直接将 episode 的所有经验（带计算好的GAE/Returns）添加到 buffer
            # PPO update 将从 buffer 采样 mini-batch，这些 mini-batch 可能包含不同玩家的经验
            
            for i in range(len(episode_experiences)):
                buffer.add(
                    states_ep[i],
                    actions_ep[i],
                    old_log_probs_ep[i],
                    returns_ep[i], # 使用计算好的回报
                    advantages_ep[i], # 使用计算好的优势
                    action_masks_ep[i],
                    model_player  # 仅存储 PPO 玩家标识，简化 update 过滤
                )
            
            exploration_epsilon = max(epsilon_min, exploration_epsilon * epsilon_decay)
            episode_rewards_deque.append(current_episode_reward) # 添加本episode的总原始奖励
            
            # PPO 更新在 buffer 积累足够经验后进行
            if len(buffer) >= batch_size:
                avg_policy_loss_epoch = 0
                avg_value_loss_epoch = 0
                avg_entropy_epoch = 0
                avg_kl_div_epoch = 0
                avg_ppo_clip_epoch = 0
                updates_done_in_episode = 0 # Note: This is per episode, should probably be total updates
                
                # PPO 更新可以进行多次 mini-batch
                for _ in range(ppo_epochs):
                    try:
                        # 从buffer采样，现在包含计算好的returns和advantages
                        batch_states, batch_actions, batch_old_log_probs, batch_returns, batch_advantages, batch_masks, batch_players = buffer.sample(batch_size)
                        
                        # 调用agent.update，并传递计算好的returns和advantages
                        metrics = agent.update(
                            batch_states,
                            batch_actions,
                            batch_old_log_probs,
                            # 不再传递原始rewards和dones
                            # 而是传递计算好的 returns 和 advantages
                            batch_returns=batch_returns,
                            batch_advantages=batch_advantages,
                            action_masks_batch=batch_masks, # 继续传递 action_mask 给模型 forward
                            # 将玩家信息传递给 update 方法，并在模型 forward 中使用
                            batch_players=batch_players # 添加 batch_players 参数
                        )
                        # ... NaN handling and metrics accumulation ...
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
                        print(f"PPO mini-batch 更新过程中出错: {e_update}")
                        continue # 跳过此mini-batch的更新
                
                if updates_done_in_episode > 0:
                    avg_policy_loss_epoch /= updates_done_in_episode
                    avg_value_loss_epoch /= updates_done_in_episode
                    avg_entropy_epoch /= updates_done_in_episode
                    avg_kl_div_epoch /= updates_done_in_episode
                    avg_ppo_clip_epoch /= updates_done_in_episode
                else:
                    print("本轮PPO所有mini-batch更新均失败或跳过")
                
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

    finally:
        # 训练结束或发生异常时，关闭引擎
        if engine_proc is not None:
            try:
                send_cmd('quit')
                engine_proc.terminate()
                print("引擎已关闭。")
            except Exception as e:
                print(f"关闭引擎时出错: {e}")

    # 训练完成，最后保存一次metrics
    print("训练完成。正在保存最终metrics...")
    with open(metrics_path, 'w') as f:
        json.dump(metrics_history, f, indent=4)
    print(f"Metrics已保存到 {metrics_path}")

def main():
    env = ChineseChessEnv()
    model = ChessNet() # 默认参数，n_actions=8100
    agent = ChessPPO(model, lr=1e-4, entropy_coef=0.05) # lr 和 entropy_coef 和 train 里匹配
    
    try:
        # 调整 engine_play_prob 参数来控制与引擎对弈的频率
        train(env, agent, n_episodes=20000, save_interval=100, batch_size=16, max_steps=200, engine_play_prob=0.3) # 例如，30% 的步数由引擎决定
    except KeyboardInterrupt:
        print("训练被用户中断。")
    finally:
        env.close()
        print("环境已关闭。")

if __name__ == "__main__":
    main() 