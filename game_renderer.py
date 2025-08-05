import pygame
import sys
from typing import Tuple, Optional
from nine_mens_morris import NineMensMorris, Player, GamePhase

class GameRenderer:
    def __init__(self, width: int = 800, height: int = 800):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Nine Men's Morris")
        
        # Colors
        self.BACKGROUND = (240, 217, 181)
        self.BOARD_COLOR = (139, 69, 19)
        self.PLAYER1_COLOR = (255, 0, 0)  # Red
        self.PLAYER2_COLOR = (0, 0, 255)  # Blue
        self.HIGHLIGHT_COLOR = (255, 255, 0)  # Yellow
        self.TEXT_COLOR = (0, 0, 0)  # Black
        
        # Board layout
        self.margin = 80
        self.board_size = min(width, height) - 2 * self.margin
        self.center_x = width // 2
        self.center_y = height // 2
        
        # Position coordinates on screen
        self.positions = self._calculate_positions()
        self.position_radius = 20
        self.piece_radius = 15
        
        # Font
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        
        # Game state
        self.selected_position = None
        self.waiting_for_removal = False
        
    def _calculate_positions(self) -> dict:
        """Calculate screen coordinates for each board position"""
        positions = {}
        
        # Nine Men's Morris 표준 보드 레이아웃
        outer_size = self.board_size
        mid_size = outer_size * 2 // 3
        inner_size = outer_size // 3
        
        # Position 0-2: 바깥 사각형 위쪽
        positions[0] = (self.center_x - outer_size//2, self.center_y - outer_size//2)
        positions[1] = (self.center_x, self.center_y - outer_size//2)
        positions[2] = (self.center_x + outer_size//2, self.center_y - outer_size//2)
        
        # Position 3-5: 중간 사각형 위쪽
        positions[3] = (self.center_x - mid_size//2, self.center_y - mid_size//2)
        positions[4] = (self.center_x, self.center_y - mid_size//2)
        positions[5] = (self.center_x + mid_size//2, self.center_y - mid_size//2)
        
        # Position 6-8: 안쪽 사각형 위쪽
        positions[6] = (self.center_x - inner_size//2, self.center_y - inner_size//2)
        positions[7] = (self.center_x, self.center_y - inner_size//2)
        positions[8] = (self.center_x + inner_size//2, self.center_y - inner_size//2)
        
        # Position 9-11: 왼쪽 세로줄
        positions[9] = (self.center_x - outer_size//2, self.center_y)
        positions[10] = (self.center_x - mid_size//2, self.center_y)
        positions[11] = (self.center_x - inner_size//2, self.center_y)
        
        # Position 12-14: 오른쪽 세로줄
        positions[12] = (self.center_x + inner_size//2, self.center_y)
        positions[13] = (self.center_x + mid_size//2, self.center_y)
        positions[14] = (self.center_x + outer_size//2, self.center_y)
        
        # Position 15-17: 안쪽 사각형 아래쪽
        positions[15] = (self.center_x - inner_size//2, self.center_y + inner_size//2)
        positions[16] = (self.center_x, self.center_y + inner_size//2)
        positions[17] = (self.center_x + inner_size//2, self.center_y + inner_size//2)
        
        # Position 18-20: 중간 사각형 아래쪽
        positions[18] = (self.center_x - mid_size//2, self.center_y + mid_size//2)
        positions[19] = (self.center_x, self.center_y + mid_size//2)
        positions[20] = (self.center_x + mid_size//2, self.center_y + mid_size//2)
        
        # Position 21-23: 바깥 사각형 아래쪽
        positions[21] = (self.center_x - outer_size//2, self.center_y + outer_size//2)
        positions[22] = (self.center_x, self.center_y + outer_size//2)
        positions[23] = (self.center_x + outer_size//2, self.center_y + outer_size//2)
        
        return positions
    
    def get_position_at_mouse(self, mouse_pos: Tuple[int, int]) -> Optional[int]:
        """Get board position at mouse coordinates"""
        mouse_x, mouse_y = mouse_pos
        
        for pos, (x, y) in self.positions.items():
            distance = ((mouse_x - x) ** 2 + (mouse_y - y) ** 2) ** 0.5
            if distance <= self.position_radius:
                return pos
        return None
    
    def draw_board(self):
        """Draw the game board"""
        self.screen.fill(self.BACKGROUND)
        
        # Draw board lines
        line_width = 3
        
        # Outer square
        outer_corners = [self.positions[i] for i in [0, 1, 2, 14, 23, 22, 21, 9]]
        pygame.draw.polygon(self.screen, self.BOARD_COLOR, outer_corners, line_width)
        
        # Middle square  
        mid_corners = [self.positions[i] for i in [3, 4, 5, 13, 20, 19, 18, 10]]
        pygame.draw.polygon(self.screen, self.BOARD_COLOR, mid_corners, line_width)
        
        # Inner square
        inner_corners = [self.positions[i] for i in [6, 7, 8, 12, 17, 16, 15, 11]]
        pygame.draw.polygon(self.screen, self.BOARD_COLOR, inner_corners, line_width)
        
        # Connecting lines
        connections = [
            (1, 4), (4, 7), (16, 19), (19, 22), (9, 10), (10, 11), (12, 13), (13, 14)
        ]
        
        for start, end in connections:
            pygame.draw.line(self.screen, self.BOARD_COLOR, 
                           self.positions[start], self.positions[end], line_width)
    
    def draw_pieces(self, game: NineMensMorris):
        """Draw game pieces"""
        board = game.get_board_state()
        
        for pos in range(24):
            x, y = self.positions[pos]
            
            # Draw position circle
            pygame.draw.circle(self.screen, self.BOARD_COLOR, (x, y), self.position_radius, 2)
            
            # Draw piece if present
            if board[pos] == Player.PLAYER1.value:
                pygame.draw.circle(self.screen, self.PLAYER1_COLOR, (x, y), self.piece_radius)
                pygame.draw.circle(self.screen, self.BOARD_COLOR, (x, y), self.piece_radius, 2)
            elif board[pos] == Player.PLAYER2.value:
                pygame.draw.circle(self.screen, self.PLAYER2_COLOR, (x, y), self.piece_radius)
                pygame.draw.circle(self.screen, self.BOARD_COLOR, (x, y), self.piece_radius, 2)
            
            # Highlight selected position
            if pos == self.selected_position:
                pygame.draw.circle(self.screen, self.HIGHLIGHT_COLOR, (x, y), self.position_radius + 5, 3)
    
    def draw_ui(self, game: NineMensMorris):
        """Draw user interface elements"""
        # Current player
        player_text = f"Current Player: {'Red' if game.get_current_player() == Player.PLAYER1 else 'Blue'}"
        text_surface = self.font.render(player_text, True, self.TEXT_COLOR)
        self.screen.blit(text_surface, (10, 10))
        
        # Game phase
        phase_text = f"Phase: {game.get_phase().name}"
        text_surface = self.font.render(phase_text, True, self.TEXT_COLOR)
        self.screen.blit(text_surface, (10, 50))
        
        # Pieces to place
        pieces_to_place = game.get_pieces_to_place()
        if pieces_to_place[Player.PLAYER1] > 0 or pieces_to_place[Player.PLAYER2] > 0:
            pieces_text = f"Pieces to place - Red: {pieces_to_place[Player.PLAYER1]}, Blue: {pieces_to_place[Player.PLAYER2]}"
            text_surface = self.small_font.render(pieces_text, True, self.TEXT_COLOR)
            self.screen.blit(text_surface, (10, 90))
        
        # Pieces on board
        pieces_on_board = game.get_pieces_on_board()
        board_text = f"Pieces on board - Red: {pieces_on_board[Player.PLAYER1]}, Blue: {pieces_on_board[Player.PLAYER2]}"
        text_surface = self.small_font.render(board_text, True, self.TEXT_COLOR)
        self.screen.blit(text_surface, (10, 120))
        
        # Instructions
        if game.waiting_for_mill_removal:
            instruction = "Mill formed! Click opponent piece to remove"
        elif game.get_phase() == GamePhase.PLACING:
            instruction = "Click empty position to place piece"
        else:
            if self.selected_position is None:
                instruction = "Click your piece to select"
            else:
                instruction = "Click empty position to move"
        
        instruction_surface = self.small_font.render(instruction, True, self.TEXT_COLOR)
        self.screen.blit(instruction_surface, (10, self.height - 30))
        
        # Game over check
        game_over, winner = game.is_game_over()
        if game_over:
            winner_text = f"Game Over! {'Red' if winner == Player.PLAYER1 else 'Blue'} wins!"
            text_surface = self.font.render(winner_text, True, self.TEXT_COLOR)
            text_rect = text_surface.get_rect(center=(self.width//2, self.height//2 + 200))
            pygame.draw.rect(self.screen, self.BACKGROUND, text_rect.inflate(20, 20))
            self.screen.blit(text_surface, text_rect)
    
    def render(self, game: NineMensMorris):
        """Render the complete game state"""
        self.draw_board()
        self.draw_pieces(game)
        self.draw_ui(game)
        pygame.display.flip()
    
    def handle_click(self, game: NineMensMorris, mouse_pos: Tuple[int, int]) -> bool:
        """Handle mouse click"""
        clicked_pos = self.get_position_at_mouse(mouse_pos)
        if clicked_pos is None:
            return False
        
        if game.waiting_for_mill_removal:
            # Handle piece removal after mill
            success = game.make_move((clicked_pos,))
            return success
        
        elif game.get_phase() == GamePhase.PLACING:
            # Placing phase
            if game.is_position_empty(clicked_pos):
                success = game.make_move((clicked_pos,))
                return success
        
        else:
            # Moving/Flying phase
            current_player = game.get_current_player()
            board = game.get_board_state()
            
            if self.selected_position is None:
                # Select piece
                if board[clicked_pos] == current_player.value:
                    self.selected_position = clicked_pos
            else:
                # Move piece or select different piece
                if board[clicked_pos] == current_player.value:
                    # Select different piece
                    self.selected_position = clicked_pos
                elif game.is_position_empty(clicked_pos):
                    # Make move
                    success = game.make_move((self.selected_position, clicked_pos))
                    if success:
                        self.selected_position = None
                        return True
                else:
                    # Invalid move, deselect
                    self.selected_position = None
        
        return False
    
    def quit(self):
        """Clean up and quit"""
        pygame.quit()
        sys.exit()