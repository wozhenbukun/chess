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
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.model_experiences = []  # 存储模型玩家的经验
        self.engine_experiences = []  # 存储引擎玩家的经验

    def add(self, state, action, old_log_prob, return_, advantage, action_mask, player):
        # 确保存储的是 numpy 数组而不是 tensor
        if torch.is_tensor(state):
            state = state.cpu().numpy()
        if torch.is_tensor(action_mask):
            action_mask = action_mask.cpu().numpy()
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, old_log_prob, return_, advantage, action_mask, player)
        self.position = (self.position + 1) % self.capacity
        
        # 分别存储不同玩家的经验
        if player == 0:  # 模型玩家
            self.model_experiences.append(self.position - 1)
        else:  # 引擎玩家
            self.engine_experiences.append(self.position - 1)

    def sample(self, batch_size):
        # 确保从两种经验中均衡采样
        model_batch_size = batch_size // 2
        engine_batch_size = batch_size - model_batch_size
        
        # 从模型经验中采样
        model_indices = np.random.choice(
            self.model_experiences,
            size=min(model_batch_size, len(self.model_experiences)),
            replace=False
        ).astype(int)  # 转换为整数类型
        
        # 从引擎经验中采样
        engine_indices = np.random.choice(
            self.engine_experiences,
            size=min(engine_batch_size, len(self.engine_experiences)),
            replace=False
        ).astype(int)  # 转换为整数类型
        
        # 合并采样结果
        indices = np.concatenate([model_indices, engine_indices])
        return [self.buffer[i] for i in indices]

    def __len__(self):
        return len(self.buffer)

def train(env, agent, n_episodes=100, max_steps=200, save_interval=100, batch_size=256, run_name_prefix="run", initial_engine_play_prob=0.0, final_engine_play_prob=0.3, engine_ramp_episodes=5000, model_experience_weight=1.0, engine_experience_weight=0.5):
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

    buffer = ExperienceBuffer(capacity=100000)
    
    episode_rewards_deque = deque(maxlen=10) # 用于计算最近10个episode的平均奖励
    best_reward = float('-inf')
    
    exploration_epsilon = 1.0
    epsilon_min = 0.2 
    epsilon_decay = 0.999 
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        agent.optimizer, 
        mode='max', 
        factor=0.8,  # 从0.5改为0.8，使衰减更缓慢
        patience=50,  # 从30改为50，增加等待时间
        min_lr=1e-6,  # 添加最小学习率限制
        verbose=True  # 添加verbose以显示学习率变化
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
        "learning_rates": [],
        "engine_play_probs": [],  # 添加引擎使用概率的记录
        "loss": []  # 添加损失的记录
    }
    
    games_ended_by_max_steps = 0 # 新增: 统计达到max_steps的对局数量
    
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
            # 计算当前引擎使用概率
            if episode < engine_ramp_episodes:
                current_engine_play_prob = initial_engine_play_prob + (final_engine_play_prob - initial_engine_play_prob) * (episode / engine_ramp_episodes)
            else:
                current_engine_play_prob = final_engine_play_prob
            
            # 记录当前引擎使用概率
            metrics_history["engine_play_probs"].append(current_engine_play_prob)

            # 随机决定本局中哪个玩家由 PPO 模型控制（0=红方,1=黑方）
            model_player = np.random.choice([0, 1])
            engine_player = 1 - model_player
            # print(f"Episode {episode+1}: PPO 控制 {'红方' if model_player==0 else '黑方'}, 引擎控制 {'黑方' if model_player==0 else '红方'}")

            state = env.reset()
            current_episode_reward = 0
            episode_experiences = [] # 临时存储当前episode的原始经验
            model_experiences = [] # 存储模型玩家的经验
            engine_experiences = [] # 存储引擎玩家的经验
            
            for step in range(max_steps):
                if step >= max_steps:
                    print(f"Episode {episode+1}: 达到最大步数 {max_steps}!")
                
                valid_actions, action_mask = env.get_valid_actions()
                if not valid_actions:
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
                    next_state, reward, done, info = env.step(action)

                    # 记录 PPO 经验 - 保持原始奖励
                    model_experiences.append((state.copy(), action, reward, next_state.copy(), done, old_log_prob_item, action_mask.copy(), model_player))

                else:
                    # ------------ 引擎走棋 -------------
                    if engine_proc is None or np.random.random() > current_engine_play_prob:
                        action = np.random.choice(valid_actions)
                        next_state, reward, done, info = env.step(action)
                    else:
                        try:
                            fen_state = env.to_fen()
                            send_cmd(f'position fen {fen_state}')
                            send_cmd('go movetime 1000')
                            lines = read_until('bestmove')
                            
                            action_uci = None
                            for line in lines:
                                if line.startswith('bestmove'):
                                    action_uci = line.split()[1]
                                    break
                                    
                            if action_uci is None or action_uci == '(none)':
                                action = np.random.choice(valid_actions)
                                next_state, reward, done, info = env.step(action)
                            else:
                                try:
                                    next_state, reward, done, info = env.step(action_uci)
                                    # 将 UCI 招法转换为整数动作，以便记录到 buffer
                                    action = env._uci_to_action(action_uci)
                                except ValueError as ve:
                                    action = np.random.choice(valid_actions)
                                    next_state, reward, done, info = env.step(action)
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
                            
                            action = np.random.choice(valid_actions)
                            next_state, reward, done, info = env.step(action)
                    
                    # 记录引擎/随机动作的经验 - 保持原始奖励
                    engine_experiences.append((state.copy(), action, reward, next_state.copy(), done, 0.0, action_mask.copy(), engine_player))
                
                # 更新 state 与累积奖励 - 只累积模型玩家的奖励
                state = next_state
                if env.current_player == model_player:
                    current_episode_reward += reward
                
                if done:
                    break

            # 检查是否因为达到max_steps而结束
            if not done and step == max_steps - 1: # 如果是因为达到最大步数而结束
                print(f"Episode {episode+1}: 达到最大步数 {max_steps}，视为和棋。")
                done = True # 将 done 设置为 True
                info['final_reward'] = 0.0 # 明确设置和棋奖励
                # 可以在这里决定是否给双方经验一个小的奖励或惩罚
                # 例如，将 model_experiences 和 engine_experiences 的最后一步奖励设为0
                if model_experiences:
                    last_model_exp = list(model_experiences[-1])
                    last_model_exp[2] = 0.0 # reward
                    model_experiences[-1] = tuple(last_model_exp)
                if engine_experiences:
                    last_engine_exp = list(engine_experiences[-1])
                    last_engine_exp[2] = 0.0 # reward
                    engine_experiences[-1] = tuple(last_engine_exp)


            # Determine final episode reward for logging
            logged_episode_reward = 0.0
            final_fen_for_debug = env.to_fen() # Get FEN for debugging
            
            if done: 
                final_reward_from_env = info.get('final_reward', reward) # 这是模型玩家直接从环境获得的最终奖励

                print(f"\n--- DEBUG: Episode {episode + 1} ENDED (done={done}) ---")
                print(f"Final FEN: {final_fen_for_debug}")
                print(f"Games ended by max_steps so far: {games_ended_by_max_steps}")
                print(f"Episode model_player: {model_player}, engine_player: {engine_player}")
                print(f"  Environment returned final_reward for model_player: {final_reward_from_env:.2f}")

                # 确定获胜方和用于学习的经验
                experiences_to_learn_from = []
                rewards_for_learning = [] # 专门为学习调整的奖励
                
                if final_reward_from_env > 0: # 模型获胜
                    logged_episode_reward = final_reward_from_env
                    experiences_to_learn_from = model_experiences
                    rewards_for_learning = [exp[2] for exp in model_experiences] # 使用模型原始奖励
                    print(f"    => Model WON. Logged reward: {logged_episode_reward:.2f}. Learning from model_experiences.")
                
                elif final_reward_from_env < 0: # 引擎获胜 (模型输)
                    logged_episode_reward = final_reward_from_env # 记录模型的负奖励
                    experiences_to_learn_from = engine_experiences
                    
                    rewards_for_learning = []
                    if engine_experiences:
                        # 为引擎的获胜经验创建正向学习奖励
                        # 最后一步奖励设为正，例如1.0，其余步骤设为小的正奖励或0
                        num_engine_steps = len(engine_experiences)
                        for i, exp in enumerate(engine_experiences):
                            if i == num_engine_steps - 1:
                                rewards_for_learning.append(1.0)  # 引擎获胜的最终学习奖励
                            else:
                                rewards_for_learning.append(0.01) # 引擎中间步骤的小学习奖励
                    
                    print(f"    => Engine WON. Model lost. Logged reward (model's): {logged_episode_reward:.2f}. Learning from engine_experiences with adjusted positive rewards.")

                else: # 和棋
                    logged_episode_reward = 0.0
                    # 和棋时可以选择不学习，或者双方都学但奖励较低
                    # 这里我们选择不从此局学习 (或者可以考虑双方经验都用，奖励为0)
                    print(f"    => DRAW. Logged reward: {logged_episode_reward:.2f}. No experiences added for learning from this draw.")

                # 如果有经验需要学习
                if experiences_to_learn_from and rewards_for_learning:
                    states_ep, actions_ep, _, next_states_ep, dones_ep, old_log_probs_ep, action_masks_ep, players_ep = zip(*experiences_to_learn_from)
                    
                    # 使用调整后的 rewards_for_learning
                    current_rewards_ep = list(rewards_for_learning)
                    
                    # 确保 dones_ep 的最后一步是 True
                    dones_ep_list = list(dones_ep)
                    if not dones_ep_list[-1]:
                        dones_ep_list[-1] = True
                    
                    states_ep_tensor = torch.FloatTensor(np.array(states_ep)).unsqueeze(1).to(device)
                    # players_ep_tensor 需要根据当前学习的是谁的经验来设定
                    # 如果学习模型经验，players_ep[0] 就是 model_player
                    # 如果学习引擎经验，players_ep[0] 就是 engine_player
                    learning_player_id = experiences_to_learn_from[0][7] # 获取该经验条目记录的玩家ID
                    players_ep_tensor = torch.LongTensor(np.full_like(np.array(players_ep), learning_player_id)).to(device)


                    with torch.no_grad():
                        _, values_ep_tensor = agent.model(states_ep_tensor, players=players_ep_tensor)
                        values_ep = values_ep_tensor.squeeze(-1).cpu().numpy()

                    last_state_ep_tensor = torch.FloatTensor(np.array(next_states_ep[-1])).unsqueeze(0).unsqueeze(1).to(device)
                    last_player_ep_tensor = torch.LongTensor([learning_player_id]).to(device) # 使用学习方的ID
                    
                    with torch.no_grad():
                        _, last_value_ep_tensor = agent.model(last_state_ep_tensor, players=last_player_ep_tensor)
                        # 如果最后一帧是done=True，则下一个状态的价值为0
                        last_value_ep = last_value_ep_tensor.squeeze(-1).item() if not dones_ep_list[-1] else 0.0
                    
                    advantages_ep_tensor = agent.compute_gae(current_rewards_ep, list(values_ep), last_value_ep, dones_ep_list).cpu()
                    returns_ep_tensor = advantages_ep_tensor + torch.FloatTensor(values_ep)
                    
                    advantages_ep = advantages_ep_tensor.numpy()
                    returns_ep = returns_ep_tensor.numpy()

                    for i in range(len(experiences_to_learn_from)):
                        buffer.add(
                            states_ep[i],
                            actions_ep[i],
                            old_log_probs_ep[i], # 对于引擎经验，这个可能是0，需要PPO能处理
                            returns_ep[i],
                            advantages_ep[i],
                            action_masks_ep[i],
                            learning_player_id # 存储的是执行该动作的玩家
                        )
                    print(f"    => Added {len(experiences_to_learn_from)} experiences from {'model' if learning_player_id == model_player else 'engine'} to buffer.")

                print(f"--- END DEBUG EPISODE {episode + 1} ---\n")
            
            episode_rewards_deque.append(logged_episode_reward) # Logged reward reflects model's performance
            
            # 从经验池中采样并训练
            if len(buffer) >= batch_size:
                batch = buffer.sample(batch_size)
                
                # 解压批次数据
                states = np.array([x[0] for x in batch])
                actions = np.array([x[1] for x in batch])
                old_log_probs = np.array([x[2] for x in batch])
                returns = np.array([x[3] for x in batch])
                advantages = np.array([x[4] for x in batch])
                action_masks = np.array([x[5] for x in batch])
                players = np.array([x[6] for x in batch])
                
                # 转换为tensor并添加玩家通道
                states = torch.FloatTensor(states).unsqueeze(1).to(device)  # 添加通道维度
                players_tensor = torch.LongTensor(players).to(device)
                actions = torch.LongTensor(actions).to(device)
                old_log_probs = torch.FloatTensor(old_log_probs).to(device)
                returns = torch.FloatTensor(returns).to(device)
                advantages = torch.FloatTensor(advantages).to(device)
                action_masks = torch.FloatTensor(action_masks).to(device)
                
                # 更新策略
                loss = agent.update(states, actions, old_log_probs, returns, advantages, action_masks, players_tensor)
                metrics_history['loss'].append(loss)
            
            exploration_epsilon = max(epsilon_min, exploration_epsilon * epsilon_decay)
            
            avg_reward_print = np.mean(episode_rewards_deque) if len(episode_rewards_deque) > 0 else 0.0
            
            if (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}, Avg Reward (raw): {avg_reward_print:.2f}, Exp Eps: {exploration_epsilon:.3f}, LR: {agent.optimizer.param_groups[0]['lr']:.6f}, Max Steps Hit: {games_ended_by_max_steps}")
                
                # 记录指标
                metrics_history["episodes"].append(episode + 1)
                metrics_history["avg_rewards"].append(avg_reward_print)
                
                # 只在有训练更新时记录loss相关指标
                if len(buffer) >= batch_size:
                    metrics_history["policy_losses"].append(loss['policy_loss'])
                    metrics_history["value_losses"].append(loss['value_loss'])
                    metrics_history["entropies"].append(loss['entropy'])
                    metrics_history["kl_divs"].append(loss['kl_div'])
                    print(f"  Losses: Policy: {loss['policy_loss']:.4f}, Value: {loss['value_loss']:.4f}, Entropy: {loss['entropy']:.4f}")
                else:
                    # 如果没有训练更新，记录默认值
                    metrics_history["policy_losses"].append(0.0)
                    metrics_history["value_losses"].append(0.0)
                    metrics_history["entropies"].append(0.0)
                    metrics_history["kl_divs"].append(0.0)
                
                metrics_history["exploration_epsilons"].append(exploration_epsilon)
                metrics_history["ppo_clip_epsilons"].append(loss['epsilon'] if len(buffer) >= batch_size else 0.0)
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
        # 调整训练参数
        train(
            env, 
            agent, 
            n_episodes=200000, 
            save_interval=100, 
            batch_size=32,        # 增加batch_size以提高训练稳定性
            max_steps=200,
            initial_engine_play_prob=0,     # 初始不使用引擎
            final_engine_play_prob=0.8,       # 降低最终引擎使用概率
            engine_ramp_episodes=1000,        # 增加引擎使用概率的过渡期
            model_experience_weight=1.0,
            engine_experience_weight=0.5
        )
    except KeyboardInterrupt:
        print("训练被用户中断。")
    finally:
        env.close()
        print("环境已关闭。")

if __name__ == "__main__":
    main() 