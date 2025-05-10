import pygame
import torch
import numpy as np
from chess_env import ChineseChessEnv
from models import ChessNet, ChessPPO

class ChessGame:
    def __init__(self, model_path='runs\\run_20250510_235836\\best_model.pth'):
        self.env = ChineseChessEnv()

        # 创建两个AI代理
        self.red_agent = ChessPPO(ChessNet()) # 红方AI
        self.black_agent = ChessPPO(ChessNet()) # 黑方AI

        # 加载模型权重
        try:
            # 加载给红方代理
            checkpoint_red = torch.load(model_path, map_location=self.red_agent.model.device)
            if isinstance(checkpoint_red, dict):
                self.red_agent.model.load_state_dict(checkpoint_red['model_state_dict'])
            else:
                self.red_agent.model.load_state_dict(checkpoint_red)
            print(f"成功加载红方模型: {model_path}")

            # 加载给黑方代理 (可以加载相同或不同的模型)
            # 这里先加载相同的模型
            checkpoint_black = torch.load(model_path, map_location=self.black_agent.model.device)
            if isinstance(checkpoint_black, dict):
                 self.black_agent.model.load_state_dict(checkpoint_black['model_state_dict'])
            else:
                 self.black_agent.model.load_state_dict(checkpoint_black)
            print(f"成功加载黑方模型: {model_path}")

        except Exception as e:
            print(f"无法加载模型: {model_path}，两个AI都使用随机策略. Error: {e}")
            # 如果加载失败，代理将使用初始化的随机模型

        self.red_agent.model.eval() # 设置为评估模式
        self.black_agent.model.eval() # 设置为评估模式

        # 移除游戏状态属性，因为不再有玩家交互
        # self.selected_piece = None
        # self.valid_moves = []

    # 移除 handle_click 方法，因为不再有玩家交互
    # def handle_click(self, pos):
    #     """处理鼠标点击"""
    #     ...

    def run(self):
        """运行游戏"""
        self.env.render() # 在循环开始前初始化 Pygame 视频系统

        done = False
        while not done:
            # 处理 Pygame 事件 (只处理退出事件)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                    break

            if done: # 如果用户关闭窗口，退出游戏循环
                break

            # 获取当前玩家和合法动作
            current_player = self.env.current_player
            valid_actions, action_mask = self.env.get_valid_actions()

            if not valid_actions:
                print("无合法动作，游戏结束。")
                done = True
                continue # 跳过当前回合剩余逻辑

            # 根据当前玩家选择对应的AI代理获取动作
            if current_player == 0: # 红方
                print("红方AI思考...")
                action = self.red_agent.select_action(self.env.board.copy(), action_mask, current_player)
            else: # 黑方
                print("黑方AI思考...")
                action = self.black_agent.select_action(self.env.board.copy(), action_mask, current_player)

            # 执行动作
            print(f"Player {current_player} takes action {action}")
            next_state, reward, done, _ = self.env.step(action)

            # 渲染更新后的棋盘
            self.env.render()

            # 控制游戏速度
            pygame.time.wait(500) # 每步暂停 0.5 秒

            if done:
                print("游戏结束！")
                # 打印最终结果（可以根据 reward 或棋盘状态判断胜负）
                # 这里简单根据环境判断胜负，具体判断逻辑可能需要在 env 中完善
                # 例如检查哪个王被吃掉了
                red_king_exists = False
                black_king_exists = False
                for r in range(10):
                    for c in range(9):
                        if self.env.board[r][c] == self.env.R_KING:
                            red_king_exists = True
                        elif self.env.board[r][c] == self.env.B_KING:
                            black_king_exists = True

                if not red_king_exists:
                     print("黑方胜利！")
                elif not black_king_exists:
                     print("红方胜利！")
                else:
                    print("游戏结束，但胜负未明（可能和棋或超时等）") # 如果双方王都在，需要更复杂的结束判断

        # 等待几秒后关闭
        pygame.time.wait(3000)
        self.env.close()

def main():
    # 可以通过修改这里加载不同的模型文件
    game = ChessGame(model_path='runs\\run_20250510_235416\\best_model.pth') # 示例：加载之前训练的模型
    # game = ChessGame() # 使用默认路径
    game.run()

if __name__ == "__main__":
    main() 