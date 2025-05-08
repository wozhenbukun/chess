import pygame
import torch
import numpy as np
from chess_env import ChineseChessEnv
from models import ChessNet

class ChessGame:
    def __init__(self, model_path='models/best_model.pth'):
        self.env = ChineseChessEnv()
        self.model = ChessNet()
        
        # 加载模型
        try:
            checkpoint = torch.load(model_path, map_location=self.model.device)
            if isinstance(checkpoint, dict):
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"成功加载模型: {model_path}")
        except:
            print(f"无法加载模型: {model_path}，使用随机策略")
        
        self.model.eval()
        
        # 游戏状态
        self.selected_piece = None
        self.valid_moves = []
        
    def get_ai_move(self):
        """获取AI的移动"""
        state = self.env.board.copy()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.model.device)
            action_probs, _ = self.model(state_tensor)
        
        # 获取所有可能的动作
        valid_actions = []
        for from_row in range(10):
            for from_col in range(9):
                for to_row in range(10):
                    for to_col in range(9):
                        action = from_row * 9 * 10 * 9 + from_col * 10 * 9 + to_row * 9 + to_col
                        if self.env._is_valid_move(from_row, from_col, to_row, to_col):
                            valid_actions.append((action, action_probs[0, action].item()))
        
        if not valid_actions:
            return None
        
        # 选择最佳动作
        valid_actions.sort(key=lambda x: x[1], reverse=True)
        return valid_actions[0][0]
    
    def handle_click(self, pos):
        """处理鼠标点击"""
        col = pos[0] // self.env.cell_size
        row = pos[1] // self.env.cell_size
        
        if not (0 <= row < 10 and 0 <= col < 9):
            return
        
        if self.selected_piece is None:
            # 选择棋子
            piece = self.env.board[row, col]
            if piece != self.env.EMPTY and (
                (self.env.current_player == 0 and 1 <= piece <= 7) or
                (self.env.current_player == 1 and 8 <= piece <= 14)
            ):
                self.selected_piece = (row, col)
                # 计算有效移动
                self.valid_moves = []
                for to_row in range(10):
                    for to_col in range(9):
                        if self.env._is_valid_move(row, col, to_row, to_col):
                            self.valid_moves.append((to_row, to_col))
        else:
            # 移动棋子
            from_row, from_col = self.selected_piece
            if (row, col) in self.valid_moves:
                action = from_row * 9 * 10 * 9 + from_col * 10 * 9 + row * 9 + col
                _, reward, done, _ = self.env.step(action)
                
                if done:
                    print("游戏结束！")
                    if reward > 0:
                        print("红方胜利！")
                    else:
                        print("黑方胜利！")
                    return True
            
            self.selected_piece = None
            self.valid_moves = []
        
        return False
    
    def run(self):
        """运行游戏"""
        done = False
        while not done:
            # 处理事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # 左键点击
                        done = self.handle_click(event.pos)
            
            # 渲染
            self.env.render()
            
            # AI回合
            if self.env.current_player == 1 and not done:  # 黑方（AI）
                action = self.get_ai_move()
                if action is not None:
                    _, reward, done, _ = self.env.step(action)
                    if done:
                        print("游戏结束！")
                        if reward > 0:
                            print("红方胜利！")
                        else:
                            print("黑方胜利！")
            
            pygame.time.wait(100)  # 控制游戏速度
        
        # 等待几秒后关闭
        pygame.time.wait(3000)
        self.env.close()

def main():
    game = ChessGame()
    game.run()

if __name__ == "__main__":
    main() 