import numpy as np
from typing import List, Tuple, Optional, Set
from enum import Enum

class GamePhase(Enum):
    PLACING = 1
    MOVING = 2
    FLYING = 3
    MILL_REMOVAL = 4

class Player(Enum):
    PLAYER1 = 1
    PLAYER2 = -1
    EMPTY = 0

class NineMensMorris:
    def __init__(self):
        self.board = np.zeros(24, dtype=int)
        self.current_player = Player.PLAYER1
        self.phase = GamePhase.PLACING
        self.pieces_to_place = {Player.PLAYER1: 9, Player.PLAYER2: 9}
        self.pieces_on_board = {Player.PLAYER1: 0, Player.PLAYER2: 0}
        self.selected_position = None
        self.waiting_for_mill_removal = False
        
        self.valid_positions = set(range(24))
        self.connections = self._initialize_connections()
        self.mills = self._initialize_mills()
        
    def _initialize_connections(self) -> dict:
        connections = {
            0: [1, 9], 1: [0, 2, 4], 2: [1, 14],
            3: [4, 10], 4: [1, 3, 5, 7], 5: [4, 13],
            6: [7, 11], 7: [4, 6, 8], 8: [7, 12],
            9: [0, 10, 21], 10: [3, 9, 11, 18], 11: [6, 10, 15],
            12: [8, 13, 17], 13: [5, 12, 14, 20], 14: [2, 13, 23],
            15: [11, 16], 16: [15, 17, 19], 17: [12, 16],
            18: [10, 19], 19: [16, 18, 20, 22], 20: [13, 19],
            21: [9, 22], 22: [19, 21, 23], 23: [14, 22]
        }
        return connections
    
    def _initialize_mills(self) -> List[List[int]]:
        mills = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],
            [9, 10, 11], [12, 13, 14], [15, 16, 17],
            [18, 19, 20], [21, 22, 23],
            [0, 9, 21], [3, 10, 18], [6, 11, 15],
            [1, 4, 7], [16, 19, 22], [8, 12, 17],
            [5, 13, 20], [2, 14, 23]
        ]
        return mills
    
    def get_board_state(self) -> np.ndarray:
        return self.board.copy()
    
    def get_current_player(self) -> Player:
        return self.current_player
    
    def get_phase(self) -> GamePhase:
        return self.phase
    
    def get_pieces_to_place(self) -> dict:
        return self.pieces_to_place.copy()
    
    def get_pieces_on_board(self) -> dict:
        return self.pieces_on_board.copy()
    
    def is_valid_position(self, position: int) -> bool:
        return 0 <= position < 24
    
    def is_position_empty(self, position: int) -> bool:
        return self.board[position] == Player.EMPTY.value
    
    def get_valid_moves(self) -> List[Tuple[int, ...]]:
        if self.waiting_for_mill_removal:
            # Return removable pieces as valid moves
            removable_pieces = self.get_removable_pieces(self.current_player)
            return [(pos,) for pos in removable_pieces]
        
        if self.phase == GamePhase.PLACING:
            return [(pos,) for pos in range(24) if self.is_position_empty(pos)]
        
        elif self.phase == GamePhase.MOVING:
            moves = []
            player_positions = [i for i, piece in enumerate(self.board) 
                             if piece == self.current_player.value]
            
            for pos in player_positions:
                for neighbor in self.connections[pos]:
                    if self.is_position_empty(neighbor):
                        moves.append((pos, neighbor))
            return moves
        
        elif self.phase == GamePhase.FLYING:
            moves = []
            player_positions = [i for i, piece in enumerate(self.board) 
                             if piece == self.current_player.value]
            empty_positions = [i for i, piece in enumerate(self.board) 
                             if piece == Player.EMPTY.value]
            
            # 현재 플레이어가 3개 말을 가지고 있을 때만 Flying 가능
            if self.pieces_on_board[self.current_player] == 3:
                for from_pos in player_positions:
                    for to_pos in empty_positions:
                        moves.append((from_pos, to_pos))
            else:
                # 3개가 아니면 일반 이동 규칙 적용
                for pos in player_positions:
                    for neighbor in self.connections[pos]:
                        if self.is_position_empty(neighbor):
                            moves.append((pos, neighbor))
            return moves
        
        return []
    
    def check_mill(self, position: int) -> bool:
        player = self.board[position]
        if player == Player.EMPTY.value:
            return False
        
        for mill in self.mills:
            if position in mill:
                if all(self.board[pos] == player for pos in mill):
                    return True
        return False
    
    def get_removable_pieces(self, player: Player) -> List[int]:
        opponent = Player.PLAYER1 if player == Player.PLAYER2 else Player.PLAYER2
        opponent_positions = [i for i, piece in enumerate(self.board) 
                            if piece == opponent.value]
        
        non_mill_pieces = []
        mill_pieces = []
        
        for pos in opponent_positions:
            if self.check_mill(pos):
                mill_pieces.append(pos)
            else:
                non_mill_pieces.append(pos)
        
        return non_mill_pieces if non_mill_pieces else mill_pieces
    
    def make_move(self, move: Tuple[int, ...]) -> bool:
        if self.waiting_for_mill_removal:
            return self._remove_piece_after_mill(move[0])
        elif self.phase == GamePhase.PLACING:
            return self._place_piece(move[0])
        elif self.phase == GamePhase.MOVING or self.phase == GamePhase.FLYING:
            return self._move_piece(move[0], move[1])
        return False
    
    def _place_piece(self, position: int) -> bool:
        if not self.is_valid_position(position) or not self.is_position_empty(position):
            return False
        
        if self.pieces_to_place[self.current_player] <= 0:
            return False
        
        self.board[position] = self.current_player.value
        self.pieces_to_place[self.current_player] -= 1
        self.pieces_on_board[self.current_player] += 1
        
        mill_formed = self.check_mill(position)
        
        if sum(self.pieces_to_place.values()) == 0:
            self._update_phase()
        
        if mill_formed:
            # Set waiting for mill removal instead of auto-removing
            removable_pieces = self.get_removable_pieces(self.current_player)
            if removable_pieces:
                self.waiting_for_mill_removal = True
                return True  # Don't switch player yet
        
        self._switch_player()
        return True
    
    def _move_piece(self, from_pos: int, to_pos: int) -> bool:
        if not self.is_valid_position(from_pos) or not self.is_valid_position(to_pos):
            return False
        
        if self.board[from_pos] != self.current_player.value:
            return False
        
        if not self.is_position_empty(to_pos):
            return False
        
        if self.phase == GamePhase.MOVING:
            if to_pos not in self.connections[from_pos]:
                return False
        elif self.phase == GamePhase.FLYING:
            # Flying phase에서는 현재 플레이어가 3개 말을 가지고 있을 때만 자유 이동
            if self.pieces_on_board[self.current_player] != 3:
                if to_pos not in self.connections[from_pos]:
                    return False
        
        self.board[from_pos] = Player.EMPTY.value
        self.board[to_pos] = self.current_player.value
        
        mill_formed = self.check_mill(to_pos)
        
        if mill_formed:
            # Set waiting for mill removal instead of auto-removing
            removable_pieces = self.get_removable_pieces(self.current_player)
            if removable_pieces:
                self.waiting_for_mill_removal = True
                return True  # Don't switch player yet
        
        self._switch_player()
        return True
    
    def remove_piece(self, position: int) -> bool:
        if not self.is_valid_position(position):
            return False
        
        opponent = Player.PLAYER1 if self.current_player == Player.PLAYER2 else Player.PLAYER2
        
        if self.board[position] != opponent.value:
            return False
        
        removable_pieces = self.get_removable_pieces(self.current_player)
        if position not in removable_pieces:
            return False
        
        self.board[position] = Player.EMPTY.value
        self.pieces_on_board[opponent] -= 1
        
        self._update_phase()
        self._switch_player()
        
        return True
    
    def _remove_piece_after_mill(self, position: int) -> bool:
        """Remove opponent piece after mill formation"""
        if not self.waiting_for_mill_removal:
            return False
        
        removable_pieces = self.get_removable_pieces(self.current_player)
        if position not in removable_pieces:
            return False
        
        opponent = Player.PLAYER1 if self.current_player == Player.PLAYER2 else Player.PLAYER2
        self.board[position] = Player.EMPTY.value
        self.pieces_on_board[opponent] -= 1
        
        # Reset mill removal state and switch player
        self.waiting_for_mill_removal = False
        self._update_phase()
        self._switch_player()
        
        return True
    
    def _switch_player(self):
        self.current_player = Player.PLAYER1 if self.current_player == Player.PLAYER2 else Player.PLAYER2
    
    def _update_phase(self):
        if sum(self.pieces_to_place.values()) > 0:
            self.phase = GamePhase.PLACING
        elif (self.pieces_on_board[Player.PLAYER1] == 3 or 
              self.pieces_on_board[Player.PLAYER2] == 3):
            self.phase = GamePhase.FLYING
        else:
            self.phase = GamePhase.MOVING
    
    def is_game_over(self) -> Tuple[bool, Optional[Player]]:
        if self.phase == GamePhase.PLACING:
            return False, None
        
        for player in [Player.PLAYER1, Player.PLAYER2]:
            if self.pieces_on_board[player] < 3:
                winner = Player.PLAYER1 if player == Player.PLAYER2 else Player.PLAYER2
                return True, winner
        
        valid_moves = self.get_valid_moves()
        if not valid_moves:
            winner = Player.PLAYER1 if self.current_player == Player.PLAYER2 else Player.PLAYER2
            return True, winner
        
        return False, None
    
    def get_neural_network_input(self) -> np.ndarray:
        """CNN/ResNet을 위한 입력 형태로 변환"""
        # 3개 채널: Player1 pieces, Player2 pieces, Empty positions
        channels = np.zeros((3, 24), dtype=np.float32)
        
        # Player 1 pieces
        channels[0] = (self.board == Player.PLAYER1.value).astype(np.float32)
        # Player 2 pieces  
        channels[1] = (self.board == Player.PLAYER2.value).astype(np.float32)
        # Empty positions
        channels[2] = (self.board == Player.EMPTY.value).astype(np.float32)
        
        return channels
    
    def get_board_tensor(self) -> np.ndarray:
        """7x7 그리드 형태로 보드를 표현 (CNN 친화적)"""
        # Nine Men's Morris 보드를 7x7 그리드로 매핑
        board_mapping = {
            0: (0, 0), 1: (0, 3), 2: (0, 6),
            3: (1, 1), 4: (1, 3), 5: (1, 5),
            6: (2, 2), 7: (2, 3), 8: (2, 4),
            9: (3, 0), 10: (3, 1), 11: (3, 2),
            12: (3, 4), 13: (3, 5), 14: (3, 6),
            15: (4, 2), 16: (4, 3), 17: (4, 4),
            18: (5, 1), 19: (5, 3), 20: (5, 5),
            21: (6, 0), 22: (6, 3), 23: (6, 6)
        }
        
        # 3 channels for different piece types
        tensor = np.zeros((3, 7, 7), dtype=np.float32)
        
        for pos, (row, col) in board_mapping.items():
            if self.board[pos] == Player.PLAYER1.value:
                tensor[0, row, col] = 1.0
            elif self.board[pos] == Player.PLAYER2.value:
                tensor[1, row, col] = 1.0
            else:
                tensor[2, row, col] = 1.0
        
        return tensor
    
    def clone(self):
        """게임 상태 복사"""
        new_game = NineMensMorris()
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        new_game.phase = self.phase
        new_game.pieces_to_place = self.pieces_to_place.copy()
        new_game.pieces_on_board = self.pieces_on_board.copy()
        new_game.selected_position = self.selected_position
        new_game.waiting_for_mill_removal = self.waiting_for_mill_removal
        return new_game