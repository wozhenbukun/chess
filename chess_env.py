import numpy as np
import gym
from gym import spaces
import pygame
import sys
import torch

class ChineseChessEnv(gym.Env):
    """中国象棋强化学习环境"""
    
    # 棋子类型
    EMPTY = 0
    # 红方棋子
    R_KING = 1    # 帅
    R_ADVISOR = 2 # 仕
    R_ELEPHANT = 3 # 相
    R_HORSE = 4   # 马
    R_CHARIOT = 5 # 车
    R_CANNON = 6  # 炮
    R_PAWN = 7    # 兵
    # 黑方棋子
    B_KING = 8    # 将
    B_ADVISOR = 9 # 士
    B_ELEPHANT = 10 # 象
    B_HORSE = 11  # 马
    B_CHARIOT = 12 # 车
    B_CANNON = 13 # 炮
    B_PAWN = 14   # 卒
    
    def __init__(self):
        super(ChineseChessEnv, self).__init__()
        
        # 棋盘大小: 10行9列
        self.board_size = (10, 9)
        
        # 动作空间: 从一个位置到另一个位置的所有可能移动
        # 每个动作是一个四元组 (from_row, from_col, to_row, to_col)
        self.action_space = spaces.Discrete(10 * 9 * 10 * 9)
        
        # 观察空间: 棋盘状态
        self.observation_space = spaces.Box(
            low=0, high=14, shape=(10, 9), dtype=np.int32
        )
        
        # 初始化棋盘
        self.board = self._init_board()
        
        # 当前玩家 (0: 红方, 1: 黑方)
        self.current_player = 0
        
        # 游戏是否结束
        self.done = False
        
        # 用于渲染的变量
        self.screen = None
        self.cell_size = 60
        # 修改窗口尺寸计算
        self.width = (self.board_size[1] + 1) * self.cell_size # 9列 + 1个单元格边距
        self.height = (self.board_size[0] + 1) * self.cell_size # 10行 + 1个单元格边距
        
        # 添加棋盘历史记录用于检测重复移动
        self.board_history = []
        self.max_history_length = 6 # 记录最近的6个局面
        
    def _init_board(self):
        """初始化棋盘"""
        board = np.zeros(self.board_size, dtype=np.int32)
        
        # 放置红方棋子
        # 底线
        board[9, 0] = self.R_CHARIOT
        board[9, 1] = self.R_HORSE
        board[9, 2] = self.R_ELEPHANT
        board[9, 3] = self.R_ADVISOR
        board[9, 4] = self.R_KING
        board[9, 5] = self.R_ADVISOR
        board[9, 6] = self.R_ELEPHANT
        board[9, 7] = self.R_HORSE
        board[9, 8] = self.R_CHARIOT
        # 炮
        board[7, 1] = self.R_CANNON
        board[7, 7] = self.R_CANNON
        # 兵
        board[6, 0] = self.R_PAWN
        board[6, 2] = self.R_PAWN
        board[6, 4] = self.R_PAWN
        board[6, 6] = self.R_PAWN
        board[6, 8] = self.R_PAWN
        
        # 放置黑方棋子
        # 底线
        board[0, 0] = self.B_CHARIOT
        board[0, 1] = self.B_HORSE
        board[0, 2] = self.B_ELEPHANT
        board[0, 3] = self.B_ADVISOR
        board[0, 4] = self.B_KING
        board[0, 5] = self.B_ADVISOR
        board[0, 6] = self.B_ELEPHANT
        board[0, 7] = self.B_HORSE
        board[0, 8] = self.B_CHARIOT
        # 炮
        board[2, 1] = self.B_CANNON
        board[2, 7] = self.B_CANNON
        # 卒
        board[3, 0] = self.B_PAWN
        board[3, 2] = self.B_PAWN
        board[3, 4] = self.B_PAWN
        board[3, 6] = self.B_PAWN
        board[3, 8] = self.B_PAWN
        
        return board
    
    def reset(self):
        """重置环境"""
        self.board = self._init_board()
        self.current_player = 0
        self.done = False
        self.board_history = [] # 清空历史记录
        return self.board.copy()
    
    def step(self, action):
        """执行一步动作"""
        # --- 修改：处理整数动作编码或 UCI 招法字符串 ---
        if isinstance(action, str): # 如果输入是字符串 (UCI 招法)
            try:
                from_row, from_col, to_row, to_col = self._parse_uci_move(action)
            except ValueError:
                print(f"非法 UCI 招法: {action}")
                return self.board.copy(), -10, True, {"message": "非法 UCI 招法"} # 非法招法给予惩罚并结束游戏
        else: # 如果输入是整数动作编码
             # 解析动作
            from_row = action // (9 * 10 * 9)
            from_col = (action % (9 * 10 * 9)) // (10 * 9)
            to_row = (action % (10 * 9)) // 9
            to_col = action % 9
        # --- 修改结束 ---
        
        # 检查移动是否合法
        if not self._is_valid_move(from_row, from_col, to_row, to_col):
            return self.board.copy(), -10, True, {"message": "非法移动"}  # 非法移动给予惩罚并结束游戏
        
        # 执行移动
        piece = self.board[from_row][from_col]
        captured_piece = self.board[to_row][to_col]
        
        # 计算移动前的局面价值
        pre_value = self._evaluate_position()
        
        # 执行移动
        self.board[to_row][to_col] = piece
        self.board[from_row][from_col] = 0
        
        # 将移动后的新局面添加到历史记录
        self.board_history.append(self.board.copy())
        # 保持历史记录长度不超过最大值
        if len(self.board_history) > self.max_history_length:
            self.board_history.pop(0) # 移除最旧的记录
        
        # 计算移动后的局面价值
        post_value = self._evaluate_position()
        
        # 计算奖励
        reward = 0
        
        # 1. 吃子奖励，后续可能需要加上所吃子的位置价值
        if captured_piece != 0:
            reward += (self._get_piece_value(captured_piece) * 0.1 + self._get_position_value(captured_piece, to_row, to_col) * 0.1 + self._get_mobility_value(captured_piece, to_row, to_col) * 0.1)
        
        # 2. 局面改善奖励
        value_diff = post_value - pre_value
        reward += value_diff * 0.05
        
        # 3. 特殊奖励
        # 将军奖励
        if self._is_check():
            reward += 0.5
        
        # 将死奖励
        if self._is_checkmate():
            reward += 10.0
        
        # 4. 惩罚项
        # 重复移动惩罚
        # 注意：重复移动已经在 _is_valid_move 中被阻止，这里主要用于奖励计算（尽管实际不会发生）
        if self._is_repeated_move(from_row, from_col, to_row, to_col):
            reward -= 0.2
        
        # 被将军惩罚
        if self._is_checked():
            reward -= 0.3
        
        # 更新状态
        self.board = self._get_state()
        
        # 获取下一个状态
        self.next_state = self.board.copy()

        # 检查游戏是否结束
        self.done = self._is_game_over()

        # 切换玩家 (0 -> 1, 1 -> 0)
        self.current_player = 1 - self.current_player

        return self.next_state, reward, self.done, {}
    
    def _is_valid_move(self, from_row, from_col, to_row, to_col):
        """检查移动是否合法"""
        # 基本检查
        if not (0 <= from_row < 10 and 0 <= from_col < 9 and 0 <= to_row < 10 and 0 <= to_col < 9):
            return False
        
        # 检查是否有棋子可以移动
        piece = self.board[from_row][from_col]
        if piece == self.EMPTY:
            return False
        
        # 检查是否是当前玩家的棋子
        if self.current_player == 0 and not (1 <= piece <= 7):  # 红方
            return False
        if self.current_player == 1 and not (8 <= piece <= 14):  # 黑方
            return False
        
        # 检查目标位置是否是自己的棋子
        target = self.board[to_row][to_col]
        if self.current_player == 0 and 1 <= target <= 7:  # 红方
            return False
        if self.current_player == 1 and 8 <= target <= 14:  # 黑方
            return False
        
        # 根据不同棋子类型检查移动规则
        if piece == self.R_KING or piece == self.B_KING:
            return self._is_valid_king_move(from_row, from_col, to_row, to_col, piece)
        elif piece == self.R_ADVISOR or piece == self.B_ADVISOR:
            return self._is_valid_advisor_move(from_row, from_col, to_row, to_col, piece)
        elif piece == self.R_ELEPHANT or piece == self.B_ELEPHANT:
            return self._is_valid_elephant_move(from_row, from_col, to_row, to_col, piece)
        elif piece == self.R_HORSE or piece == self.B_HORSE:
            return self._is_valid_horse_move(from_row, from_col, to_row, to_col)
        elif piece == self.R_CHARIOT or piece == self.B_CHARIOT:
            return self._is_valid_chariot_move(from_row, from_col, to_row, to_col)
        elif piece == self.R_CANNON or piece == self.B_CANNON:
            return self._is_valid_cannon_move(from_row, from_col, to_row, to_col)
        elif piece == self.R_PAWN or piece == self.B_PAWN:
            return self._is_valid_pawn_move(from_row, from_col, to_row, to_col, piece)
        
        # 在验证合法性时，模拟移动并检查是否导致重复局面
        if self._is_repeated_move_check(from_row, from_col, to_row, to_col):
             return False # 如果是重复局面，视为非法移动

        # 模拟移动，检查是否会导致将帅照面
        temp_board = self.board.copy()
        piece = temp_board[from_row][from_col]
        temp_board[to_row][to_col] = piece
        temp_board[from_row][from_col] = self.EMPTY

        if self._is_kings_facing(temp_board):
            return False # 如果移动后将帅照面，视为非法移动

        return True
    
    def _is_valid_king_move(self, from_row, from_col, to_row, to_col, piece):
        """检查将/帅的移动是否合法"""
        # 将/帅只能在九宫格内移动
        if piece == self.R_KING:  # 红方帅
            if not (7 <= to_row <= 9 and 3 <= to_col <= 5):
                return False
        else:  # 黑方将
            if not (0 <= to_row <= 2 and 3 <= to_col <= 5):
                return False
        
        # 将/帅每次只能移动一格（上下左右）
        if abs(from_row - to_row) + abs(from_col - to_col) != 1:
            return False
        
        return True
    
    def _is_valid_advisor_move(self, from_row, from_col, to_row, to_col, piece):
        """检查士/仕的移动是否合法"""
        # 士/仕只能在九宫格内移动
        if piece == self.R_ADVISOR:  # 红方仕
            if not (7 <= to_row <= 9 and 3 <= to_col <= 5):
                return False
        else:  # 黑方士
            if not (0 <= to_row <= 2 and 3 <= to_col <= 5):
                return False
        
        # 士/仕只能斜着走一格
        if abs(from_row - to_row) != 1 or abs(from_col - to_col) != 1:
            return False
        
        return True
    
    def _is_valid_elephant_move(self, from_row, from_col, to_row, to_col, piece):
        """检查象/相的移动是否合法"""
        # 象/相不能过河
        if piece == self.R_ELEPHANT and to_row < 5:  # 红方相不能过河
            return False
        if piece == self.B_ELEPHANT and to_row > 4:  # 黑方象不能过河
            return False
        
        # 象/相走田字
        if abs(from_row - to_row) != 2 or abs(from_col - to_col) != 2:
            return False
        
        # 象/相不能蹩腿
        middle_row = (from_row + to_row) // 2
        middle_col = (from_col + to_col) // 2
        if self.board[middle_row][middle_col] != self.EMPTY:
            return False
        
        return True
    
    def _is_valid_horse_move(self, from_row, from_col, to_row, to_col):
        """检查马的移动是否合法"""
        # 马走日字
        row_diff = abs(from_row - to_row)
        col_diff = abs(from_col - to_col)
        if not ((row_diff == 2 and col_diff == 1) or (row_diff == 1 and col_diff == 2)):
            return False
        
        # 马不能蹩腿
        if row_diff == 2:  # 竖着走
            middle_row = (from_row + to_row) // 2
            middle_col = from_col
        else:  # 横着走
            middle_row = from_row
            middle_col = (from_col + to_col) // 2
        
        if self.board[middle_row][middle_col] != self.EMPTY:
            return False
        
        return True
    
    def _is_valid_chariot_move(self, from_row, from_col, to_row, to_col):
        """检查车的移动是否合法"""
        # 车只能直线移动
        if from_row != to_row and from_col != to_col:
            return False
        
        # 检查路径上是否有其他棋子
        if from_row == to_row:  # 横向移动
            start_col = min(from_col, to_col) + 1
            end_col = max(from_col, to_col)
            for col in range(start_col, end_col):
                if self.board[from_row][col] != self.EMPTY:
                    return False
        else:  # 纵向移动
            start_row = min(from_row, to_row) + 1
            end_row = max(from_row, to_row)
            for row in range(start_row, end_row):
                if self.board[row][from_col] != self.EMPTY:
                    return False
        
        return True
    
    def _is_valid_cannon_move(self, from_row, from_col, to_row, to_col):
        """检查炮的移动是否合法"""
        # 炮只能直线移动
        if from_row != to_row and from_col != to_col:
            return False
        
        # 计算路径上的棋子数量
        pieces_in_path = 0
        if from_row == to_row:  # 横向移动
            start_col = min(from_col, to_col) + 1
            end_col = max(from_col, to_col)
            for col in range(start_col, end_col):
                if self.board[from_row][col] != self.EMPTY:
                    pieces_in_path += 1
        else:  # 纵向移动
            start_row = min(from_row, to_row) + 1
            end_row = max(from_row, to_row)
            for row in range(start_row, end_row):
                if self.board[row][from_col] != self.EMPTY:
                    pieces_in_path += 1
        
        # 炮的移动规则：移动时不能有棋子，吃子时必须有且仅有一个棋子作为炮架
        target = self.board[to_row][to_col]
        if target == self.EMPTY:  # 移动
            return pieces_in_path == 0
        else:  # 吃子
            return pieces_in_path == 1
        
    def _is_valid_pawn_move(self, from_row, from_col, to_row, to_col, piece):
        """检查兵/卒的移动是否合法"""
        # 兵/卒每次只能移动一格
        if abs(from_row - to_row) + abs(from_col - to_col) != 1:
            return False
        
        if piece == self.R_PAWN:  # 红方兵
            # 兵不能后退
            if to_row > from_row:
                return False
            # 兵未过河前不能平移
            if from_row >= 5 and from_col != to_col:
                return False
        else:  # 黑方卒
            # 卒不能后退
            if to_row < from_row:
                return False
            # 卒未过河前不能平移
            if from_row <= 4 and from_col != to_col:
                return False
        
        return True
    
    def _get_piece_value(self, piece):
        """获取棋子基础价值"""
        # Red pieces
        if piece == self.R_KING: return 100
        if piece == self.R_ADVISOR: return 9
        if piece == self.R_ELEPHANT: return 9
        if piece == self.R_HORSE: return 5
        if piece == self.R_CHARIOT: return 5
        if piece == self.R_CANNON: return 2
        if piece == self.R_PAWN: return 1
        # Black pieces
        if piece == self.B_KING: return 100
        if piece == self.B_ADVISOR: return 9
        if piece == self.B_ELEPHANT: return 9
        if piece == self.B_HORSE: return 5
        if piece == self.B_CHARIOT: return 5
        if piece == self.B_CANNON: return 2
        if piece == self.B_PAWN: return 1
        return 0
    
    def _evaluate_position(self):
        """评估当前局面"""
        value = 0
        for row in range(10):
            for col in range(9):
                piece = self.board[row][col]
                if piece != 0:
                    # 基础棋子子力价值
                    piece_value = self._get_piece_value(piece)
                    
                    # 位置价值
                    position_value = self._get_position_value(piece, row, col)
                    
                    # 机动性价值
                    mobility_value = self._get_mobility_value(piece, row, col)
                    
                    # 综合评估
                    # 根据当前玩家区分敌我棋子价值
                    if (self.current_player == 0 and 1 <= piece <= 7) or \
                       (self.current_player == 1 and 8 <= piece <= 14):
                        # 我方棋子，增加价值
                        value += piece_value + position_value + mobility_value
                    else:
                        # 敌方棋子，减去价值
                        value -= piece_value + position_value + mobility_value
        
        return value
    
    def _get_position_value(self, piece, row, col):
        """获取棋子位置价值"""
        # 根据不同棋子类型和位置给出不同的位置价值
        position_value = 0
        
        # 兵/卒的位置价值
        if piece in [self.R_PAWN, self.B_PAWN]:
            # 过河加分
            if (piece == self.R_PAWN and row < 5) or (piece == self.B_PAWN and row > 4):
                position_value += 2
            # 越靠近中线价值越高
            position_value += abs(4 - col) * 0.5
        
        # 马的位置价值
        elif piece in [self.R_HORSE, self.B_HORSE]:
            # 马在中心位置较好
            position_value += (4 - abs(4 - col)) * 0.3
            # 避免太靠近边缘
            if col in [0, 8]:
                position_value -= 1
        
        # 车的位置价值
        elif piece in [self.R_CHARIOT, self.B_CHARIOT]:
            # 车在中线和两边较好
            if col in [0, 4, 8]:
                position_value += 1
        
        # 炮的位置价值
        elif piece in [self.R_CANNON, self.B_CANNON]:
            # 炮在中路较好
            position_value += (4 - abs(4 - col)) * 0.2
        
        # 相/象的位置价值
        elif piece in [self.R_ELEPHANT, self.B_ELEPHANT]:
            # 相象在初始位置较好
            if (piece == self.R_ELEPHANT and row in [7, 9]) or \
               (piece == self.B_ELEPHANT and row in [0, 2]):
                position_value += 1
        
        # 士的位置价值
        elif piece in [self.R_ADVISOR, self.B_ADVISOR]:
            # 士在中心位置最好
            if col == 4:
                position_value += 1
        
        # 将/帅的位置价值
        elif piece in [self.R_KING, self.B_KING]:
            # 将帅在中间位置最安全
            if col == 4:
                position_value += 1
        
        return position_value
    
    def _get_mobility_value(self, piece, row, col):
        """获取棋子机动性价值"""
        # 计算棋子可能的移动数量
        valid_moves = 0
        for to_row in range(10):
            for to_col in range(9):
                if self._is_valid_move(row, col, to_row, to_col):
                    valid_moves += 1
        return valid_moves * 0.1
    
    def _is_repeated_move_check(self, from_row, from_col, to_row, to_col):
        """模拟移动并检查是否导致重复局面"""
        # 创建当前棋盘的副本
        temp_board = self.board.copy()

        # 模拟执行移动
        piece = temp_board[from_row][from_col]
        temp_board[to_row][to_col] = piece
        temp_board[from_row][from_col] = self.EMPTY

        # 检查模拟后的局面是否在历史记录中
        for history_board in self.board_history:
            if np.array_equal(temp_board, history_board):
                return True # 发现重复局面

        return False

    def _is_repeated_move(self, from_row, from_col, to_row, to_col):
        """检查是否重复移动 (旧方法，现在主要用于奖励计算，但逻辑上已经被 _is_valid_move 中的 _is_repeated_move_check 覆盖)"""
        # 这个方法现在主要用于 step 函数中的奖励计算，实际上非法重复移动已经在 _is_valid_move 中被阻止了
        # 为了保持原代码结构，这里简单返回 _is_repeated_move_check 的结果
        return self._is_repeated_move_check(from_row, from_col, to_row, to_col)
    
    def render(self, mode='human'):
        """渲染当前棋盘状态"""
        if mode != 'human':
            return
        
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption('中国象棋')
        
        # 填充背景
        self.screen.fill((255, 223, 128))  # 浅黄色背景
        
        # 绘制偏移量
        offset_x = self.cell_size // 2
        offset_y = self.cell_size // 2

        # 绘制棋盘线条
        # 横线
        for row in range(10):
            # 调整绘制坐标
            pygame.draw.line(self.screen, (0, 0, 0),
                           (offset_x, offset_y + row * self.cell_size),
                           (offset_x + 8 * self.cell_size, offset_y + row * self.cell_size))
        
        # 竖线
        for col in range(9):
            # 上半部分
            # 调整绘制坐标
            pygame.draw.line(self.screen, (0, 0, 0),
                           (offset_x + col * self.cell_size, offset_y),
                           (offset_x + col * self.cell_size, offset_y + 4 * self.cell_size))
            # 下半部分
            # 调整绘制坐标
            pygame.draw.line(self.screen, (0, 0, 0),
                           (offset_x + col * self.cell_size, offset_y + 5 * self.cell_size),
                           (offset_x + col * self.cell_size, offset_y + 9 * self.cell_size))
        
        # 绘制九宫格
        # 红方九宫
        # 调整绘制坐标
        pygame.draw.line(self.screen, (0, 0, 0), 
                        (offset_x + 3 * self.cell_size, offset_y + 7 * self.cell_size), 
                        (offset_x + 5 * self.cell_size, offset_y + 9 * self.cell_size))
        # 调整绘制坐标
        pygame.draw.line(self.screen, (0, 0, 0), 
                        (offset_x + 5 * self.cell_size, offset_y + 7 * self.cell_size), 
                        (offset_x + 3 * self.cell_size, offset_y + 9 * self.cell_size))
        # 黑方九宫
        # 调整绘制坐标
        pygame.draw.line(self.screen, (0, 0, 0), 
                        (offset_x + 3 * self.cell_size, offset_y + 0 * self.cell_size), 
                        (offset_x + 5 * self.cell_size, offset_y + 2 * self.cell_size))
        # 调整绘制坐标
        pygame.draw.line(self.screen, (0, 0, 0), 
                        (offset_x + 5 * self.cell_size, offset_y + 0 * self.cell_size), 
                        (offset_x + 3 * self.cell_size, offset_y + 2 * self.cell_size))
        
        # 绘制棋子
        for row in range(10):
            for col in range(9):
                piece = self.board[row][col]
                if piece != self.EMPTY:
                    color = (255, 0, 0) if 1 <= piece <= 7 else (0, 0, 0)  # 红方红色，黑方黑色
                    # 在交叉点上绘制棋子
                    # 调整绘制坐标
                    center_x = offset_x + col * self.cell_size
                    center_y = offset_y + row * self.cell_size
                    pygame.draw.circle(self.screen, color, 
                                     (center_x, center_y), 
                                     self.cell_size // 2 - 5)
                    pygame.draw.circle(self.screen, (255, 255, 200), 
                                     (center_x, center_y), 
                                     self.cell_size // 2 - 8)
                    
                    # 绘制棋子文字
                    font = pygame.font.SysFont('simsun', 24)
                    piece_names = {
                        self.R_KING: "帅", self.B_KING: "将",
                        self.R_ADVISOR: "仕", self.B_ADVISOR: "士",
                        self.R_ELEPHANT: "相", self.B_ELEPHANT: "象",
                        self.R_HORSE: "马", self.B_HORSE: "马",
                        self.R_CHARIOT: "车", self.B_CHARIOT: "车",
                        self.R_CANNON: "炮", self.B_CANNON: "炮",
                        self.R_PAWN: "兵", self.B_PAWN: "卒"
                    }
                    text = font.render(piece_names[piece], True, color)
                    # 调整绘制坐标
                    text_rect = text.get_rect(center=(center_x, center_y))
                    self.screen.blit(text, text_rect)
        
        pygame.display.flip()
        
        # 处理事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
    
    def close(self):
        """关闭环境"""
        if self.screen is not None:
            pygame.quit()
            self.screen = None

    def _is_check(self):
        """检查当前玩家是否将军了对方"""
        # 找到对方玩家的将/帅位置
        opponent_player = 1 - self.current_player
        king_piece = self.R_KING if opponent_player == 0 else self.B_KING
        king_pos = None
        for row in range(10):
            for col in range(9):
                if self.board[row][col] == king_piece:
                    king_pos = (row, col)
                    break
            if king_pos:
                break

        if not king_pos:
            return False # 对方王不存在，不可能将军

        # 检查当前玩家（刚刚移动的玩家）所有棋子是否可以吃掉对方的将/帅
        current_player_pieces = range(1, 8) if self.current_player == 0 else range(8, 15)
        for row in range(10):
            for col in range(9):
                piece = self.board[row][col]
                if piece in current_player_pieces:
                    # 检查当前玩家的棋子是否可以移动到对方王的位置
                    # 这里调用 _is_valid_move 并临时切换玩家视角，以利用已有的合法移动逻辑进行攻击判定
                    original_current_player = self.current_player # 保存当前玩家
                    self.current_player = self.current_player # 保持为当前玩家，检查其棋子能否攻击
                    can_attack_king = self._is_valid_move(row, col, king_pos[0], king_pos[1])
                    self.current_player = original_current_player # 恢复玩家视角

                    if can_attack_king:
                        return True # 发现可以攻击对方王，即将军了对方

        return False # 没有棋子可以攻击对方王

    def _is_checkmate(self):
        """检查当前玩家是否被将死"""
        if not self._is_checked():
            return False
        
        # 尝试所有可能的移动
        for from_row in range(10):
            for from_col in range(9):
                piece = self.board[from_row][from_col]
                if (self.current_player == 0 and 1 <= piece <= 7) or \
                   (self.current_player == 1 and 8 <= piece <= 14):
                    for to_row in range(10):
                        for to_col in range(9):
                            if self._is_valid_move(from_row, from_col, to_row, to_col):
                                # 尝试移动
                                original_piece = self.board[to_row][to_col]
                                self.board[to_row][to_col] = piece
                                self.board[from_row][from_col] = 0
                                
                                # 检查移动后是否仍被将军
                                still_checked = self._is_checked()
                                
                                # 恢复棋盘
                                self.board[from_row][from_col] = piece
                                self.board[to_row][to_col] = original_piece
                                
                                if not still_checked:
                                    return False
        
        return True

    def _is_checked(self):
        """检查当前玩家是否被将军"""
        # 找到当前玩家的将/帅位置
        king_piece = self.R_KING if self.current_player == 0 else self.B_KING
        king_pos = None
        for row in range(10):
            for col in range(9):
                if self.board[row][col] == king_piece:
                    king_pos = (row, col)
                    break
            if king_pos:
                break

        if not king_pos:
            return False # 当前玩家的王不存在，不可能被将军

        # 检查对方所有棋子是否可以攻击到当前玩家的将/帅
        opponent_pieces = range(8, 15) if self.current_player == 0 else range(1, 8)
        for row in range(10):
            for col in range(9):
                piece = self.board[row][col]
                if piece in opponent_pieces:
                     # 检查对方的棋子是否可以移动到当前玩家王的位置
                     # 这里需要调用 _is_valid_move，并临时切换玩家视角来模拟对方的攻击
                    original_current_player = self.current_player # 保存当前玩家
                    # 暂时切换到对方玩家的视角，以便 _is_valid_move 检查对方棋子的合法移动
                    self.current_player = 1 - original_current_player
                    can_attack_king = self._is_valid_move(row, col, king_pos[0], king_pos[1])
                    self.current_player = original_current_player # 恢复玩家视角

                    if can_attack_king:
                        return True # 发现对方棋子可以攻击当前玩家的王，即被将军

        return False # 没有对方棋子可以攻击当前玩家的王

    def _is_game_over(self):
        """检查游戏是否结束"""
        # 检查是否将死
        if self._is_checkmate():
            return True
        
        # 检查是否有一方的将/帅被吃掉
        red_king_exists = False
        black_king_exists = False
        for row in range(10):
            for col in range(9):
                if self.board[row][col] == self.R_KING:
                    red_king_exists = True
                elif self.board[row][col] == self.B_KING:
                    black_king_exists = True
        
        if not red_king_exists or not black_king_exists:
            return True
        
        return False

    def _get_state(self):
        """获取当前棋盘状态"""
        return self.board.copy()

    def get_valid_actions(self):
        """获取当前玩家的所有合法移动"""
        valid_actions = []
        # 创建遮罩数组 (8100个可能的动作)
        action_mask = np.zeros(10 * 9 * 10 * 9, dtype=np.bool_)
        
        for from_row in range(10):
            for from_col in range(9):
                piece = self.board[from_row][from_col]
                # 检查是否是当前玩家的棋子
                if (self.current_player == 0 and 1 <= piece <= 7) or \
                   (self.current_player == 1 and 8 <= piece <= 14):
                    for to_row in range(10):
                        for to_col in range(9):
                            if self._is_valid_move(from_row, from_col, to_row, to_col):
                                action = from_row * 9 * 10 * 9 + from_col * 10 * 9 + to_row * 9 + to_col
                                valid_actions.append(action)
                                action_mask[action] = True
        
        return valid_actions, action_mask

    def _is_kings_facing(self, board):
        """检查当前棋盘局面下将帅是否直接照面"""
        red_king_pos = None
        black_king_pos = None

        # 找到将帅位置
        for row in range(10):
            for col in range(9):
                if board[row, col] == self.R_KING:
                    red_king_pos = (row, col)
                elif board[row, col] == self.B_KING:
                    black_king_pos = (row, col)
            if red_king_pos and black_king_pos:
                break # 都找到了，可以退出循环

        # 如果没有找到将或帅，则不可能照面（游戏应该已经结束了，但作为安全检查）
        if not red_king_pos or not black_king_pos:
            return False

        # 检查是否在同一列
        if red_king_pos[1] != black_king_pos[1]:
            return False

        # 检查之间是否有棋子
        col = red_king_pos[1]
        # 将帅总是一个在上面一个在下面，确定遍历的起始和结束行
        start_row = min(red_king_pos[0], black_king_pos[0]) + 1
        end_row = max(red_king_pos[0], black_king_pos[0])

        for row in range(start_row, end_row):
            if board[row, col] != self.EMPTY:
                return False # 之间有棋子，不照面

        return True # 在同一列且之间没有棋子，照面

    # --- 添加 FEN 转换和 UCI 招法处理 ---
    def to_fen(self):
        """将当前棋盘状态转换为中国象棋 FEN 字符串"""
        fen = ""
        piece_to_fen = {
            self.R_CHARIOT: 'R', self.R_HORSE: 'N', self.R_ELEPHANT: 'B', self.R_ADVISOR: 'A',
            self.R_KING: 'K', self.R_CANNON: 'C', self.R_PAWN: 'P',
            self.B_CHARIOT: 'r', self.B_HORSE: 'n', self.B_ELEPHANT: 'b', self.B_ADVISOR: 'a',
            self.B_KING: 'k', self.B_CANNON: 'c', self.B_PAWN: 'p'
        }

        for row in range(10):
            empty_count = 0
            for col in range(9):
                piece = self.board[row, col]
                if piece == self.EMPTY:
                    empty_count += 1
                else:
                    if empty_count > 0:
                        fen += str(empty_count)
                        empty_count = 0
                    fen_char = piece_to_fen.get(piece, '?')
                    fen += fen_char
            if empty_count > 0:
                fen += str(empty_count)
            if row < 9:
                fen += '/'

        # 添加当前玩家信息 ('w' for红方, 'b' for黑方)
        fen += " " + ('w' if self.current_player == 0 else 'b')

        # 检查FEN中是否有未知棋子
        if '?' in fen:
            print(f"警告：FEN 包含未知棋子: {fen}")

        # 保持简化表示
        fen += " - - 0 1"
        return fen

    def _parse_uci_move(self, uci_move):
        """解析 UCI 格式的招法字符串"""
        # 期望格式如 "a1a5"
        if len(uci_move) != 4:
            raise ValueError(f"非法 UCI 招法格式: {uci_move}")

        # UCI 坐标到棋盘索引的映射 (例如 a1 对应 (9, 0) 在您的环境中)
        # UCI 列 'a' -> 0, 'b' -> 1, ..., 'i' -> 8
        # UCI 行 '1' -> 9, '2' -> 8, ..., '10' -> 0 (对于红方视角)
        # 由于您的环境是固定索引 (0-9行, 0-8列)，我们需要根据 UCI 行号和列号转换为您的索引

        # UCI 坐标到 (row, col) 的映射
        def uci_to_coords(uci_str):
            col_str = uci_str[0]
            row_str = uci_str[1:]
            
            col = ord(col_str) - ord('a')
            # 中国象棋的行号是从下面往上数，所以需要转换
            # UCI 的行号 1-10 对应您的环境行索引 9-0
            try:
                row = 10 - int(row_str)
            except ValueError:
                 raise ValueError(f"非法 UCI 行号: {row_str}")

            if not (0 <= row < 10 and 0 <= col < 9):
                 raise ValueError(f"UCI 坐标超出棋盘范围: {uci_str}")

            return row, col

        from_coords_str = uci_move[:2]
        to_coords_str = uci_move[2:]

        from_row, from_col = uci_to_coords(from_coords_str)
        to_row, to_col = uci_to_coords(to_coords_str)

        return from_row, from_col, to_row, to_col

    def _uci_to_action(self, uci_move):
        """将 UCI 招法转换为环境的整数动作编码"""
        from_row, from_col, to_row, to_col = self._parse_uci_move(uci_move)
        action = from_row * 9 * 10 * 9 + from_col * 10 * 9 + to_row * 9 + to_col
        return action
    # --- 添加结束 ---

# 示例使用
if __name__ == "__main__":
    env = ChineseChessEnv()
    obs = env.reset()
    env.render()
    
    # 简单的随机走子示例
    done = False
    while not done:
        # 随机选择一个动作 (现在也可以传递 UCI 字符串)
        # action = env.action_space.sample() # 原来的整数动作
        
        # 示例：走一个 UCI 招法 (假设这是一个合法招法)
        # 您需要根据当前局面确定合法的 UCI 招法来测试
        # 例如，开局红方炮二平五 (c2e2)，在您的环境中是 (7, 1) -> (7, 4)
        # 转换为 UCI 坐标：(7, 1) -> b3, (7, 4) -> e3 (从下往上数，10-row)
        # 所以 UCI 招法是 b3e3
        # 或者红方马二进三 (b1c3) -> (9,1) -> (7,2)
        # 转换为 UCI 坐标：(9,1) -> b1, (7,2) -> c3
        # 所以 UCI 招法是 b1c3

        # 为了测试 UCI 输入，我们可以尝试一个已知的开局招法
        # 假设红方先手 (current_player == 0)
        if env.current_player == 0:
            # 红方开局炮二平五 (c2->e2) 对应的 UCI 是 c3-e3 (从红方视角)
            # 但是 UCI 坐标是固定的，与玩家无关，所以 c2 -> e2 对应的棋盘坐标是 (7, 1) -> (7, 4)
            # 转换为 UCI 坐标 (列 0-8 -> a-i, 行 0-9 -> 10-1)
            # (7, 1) -> 列 1 -> b, 行 7 -> 10-7=3 -> b3
            # (7, 4) -> 列 4 -> e, 行 7 -> 10-7=3 -> e3
            # 所以 UCI 招法是 b3e3
            uci_action = "b3e3" 
        else:
            # 黑方开局马8进7 (h9->g7) 对应的 UCI 是 h10-g8
            # 棋盘坐标 (0, 7) -> (2, 6)
            # (0, 7) -> 列 7 -> h, 行 0 -> 10-0=10 -> h10
            # (2, 6) -> 列 6 -> g, 行 2 -> 10-2=8 -> g8
            # 所以 UCI 招法是 h10g8
            uci_action = "h10g8"
            
        # 检查这个 UCI 招法是否合法，避免非法招法结束游戏
        try:
            from_row, from_col, to_row, to_col = env._parse_uci_move(uci_action)
            is_legal = env._is_valid_move(from_row, from_col, to_row, to_col)
            if is_legal:
                 print(f"执行 UCI 招法: {uci_action}")
                 obs, reward, done, info = env.step(uci_action) # 现在 step 方法可以接受 UCI 字符串
            else:
                 print(f"UCI 招法 {uci_action} 非法！")
                 # 如果非法，可以选择一个合法动作或者结束游戏
                 valid_actions, _ = env.get_valid_actions()
                 if valid_actions:
                      action = np.random.choice(valid_actions)
                      print(f"执行随机合法动作: {action}")
                      obs, reward, done, info = env.step(action)
                 else:
                      print("无合法动作，游戏结束。")
                      done = True
        except ValueError as e:
            print(f"解析 UCI 招法出错: {e}")
            done = True # 解析出错也结束游戏


        # if "message" not in info: # 检查 info 字典中是否有 message 键
        #     print(f"Player: {'Red' if env.current_player == 0 else 'Black'}, Reward: {reward}")
        #     env.render()
        #     pygame.time.wait(500)  # 暂停0.5秒
        
        # 简化测试，直接打印状态和奖励
        print(f"Player: {'Red' if env.current_player == 0 else 'Black'}, Reward: {reward}")
        env.render()
        pygame.time.wait(500)  # 暂停0.5秒


    
    print("Game Over!")
    pygame.time.wait(3000)  # 游戏结束后等待3秒
    env.close()
