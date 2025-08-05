import numpy as np
from nine_mens_morris import NineMensMorris, Player, GamePhase

def test_basic_functionality():
    """기본 게임 기능 테스트"""
    print("Testing basic game functionality...")
    
    game = NineMensMorris()
    
    # Initial state test
    assert game.get_current_player() == Player.PLAYER1
    assert game.get_phase() == GamePhase.PLACING
    assert game.get_pieces_to_place()[Player.PLAYER1] == 9
    assert game.get_pieces_to_place()[Player.PLAYER2] == 9
    
    # Place a piece
    success = game.make_move((0,))
    assert success == True
    assert game.get_board_state()[0] == Player.PLAYER1.value
    assert game.get_current_player() == Player.PLAYER2
    
    print("OK Basic functionality test passed")

def test_mill_detection():
    """Mill 감지 테스트"""
    print("Testing mill detection...")
    
    game = NineMensMorris()
    
    # Place pieces to form a mill
    game.make_move((0,))  # Player 1
    game.make_move((3,))  # Player 2
    game.make_move((1,))  # Player 1
    game.make_move((4,))  # Player 2
    game.make_move((2,))  # Player 1 - should form mill (0,1,2)
    
    # Check if mill is detected
    assert game.check_mill(2) == True
    print("OK Mill detection test passed")

def test_neural_network_input():
    """Neural network 입력 형태 테스트"""
    print("Testing neural network input conversion...")
    
    game = NineMensMorris()
    
    # Place some pieces
    game.make_move((0,))  # Player 1
    game.make_move((3,))  # Player 2
    
    # Test 1D input (24 positions, 3 channels)
    nn_input = game.get_neural_network_input()
    assert nn_input.shape == (3, 24)
    assert nn_input[0, 0] == 1.0  # Player 1 at position 0
    assert nn_input[1, 3] == 1.0  # Player 2 at position 3
    assert np.sum(nn_input[2]) == 22  # 22 empty positions
    
    # Test 2D tensor input (7x7 grid, 3 channels)
    board_tensor = game.get_board_tensor()
    assert board_tensor.shape == (3, 7, 7)
    
    print("OK Neural network input test passed")

def test_game_phases():
    """게임 페이즈 전환 테스트"""
    print("Testing game phase transitions...")
    
    game = NineMensMorris()
    
    # Placing phase
    assert game.get_phase() == GamePhase.PLACING
    
    # Place all pieces quickly
    positions = list(range(18))  # First 18 positions
    for i, pos in enumerate(positions):
        if game.get_pieces_to_place()[Player.PLAYER1] > 0 or game.get_pieces_to_place()[Player.PLAYER2] > 0:
            game.make_move((pos,))
    
    # Should be in moving phase now
    assert game.get_phase() in [GamePhase.MOVING, GamePhase.FLYING]
    
    print("OK Game phase transition test passed")

def test_valid_moves():
    """유효한 움직임 생성 테스트"""
    print("Testing valid move generation...")
    
    game = NineMensMorris()
    
    # Placing phase - should have 24 valid moves initially
    valid_moves = game.get_valid_moves()
    assert len(valid_moves) == 24
    
    # Place a piece
    game.make_move((0,))
    valid_moves = game.get_valid_moves()
    assert len(valid_moves) == 23  # One less empty position
    
    print("OK Valid move generation test passed")

def test_game_clone():
    """게임 상태 복사 테스트"""
    print("Testing game state cloning...")
    
    game = NineMensMorris()
    game.make_move((0,))
    game.make_move((1,))
    
    # Clone the game
    cloned_game = game.clone()
    
    # Check if states are identical
    assert np.array_equal(game.get_board_state(), cloned_game.get_board_state())
    assert game.get_current_player() == cloned_game.get_current_player()
    assert game.get_phase() == cloned_game.get_phase()
    
    # Check if they are independent
    game.make_move((2,))
    assert not np.array_equal(game.get_board_state(), cloned_game.get_board_state())
    
    print("OK Game state cloning test passed")

def test_flying_phase():
    """Flying phase 상세 테스트"""
    print("Testing flying phase...")
    
    game = NineMensMorris()
    
    # 직접 보드 상태를 설정 (Player1이 3개, Player2가 4개)
    game.board[0] = Player.PLAYER1.value
    game.board[1] = Player.PLAYER1.value
    game.board[2] = Player.PLAYER1.value
    game.board[3] = Player.PLAYER2.value
    game.board[4] = Player.PLAYER2.value
    game.board[5] = Player.PLAYER2.value
    game.board[6] = Player.PLAYER2.value
    
    game.pieces_to_place = {Player.PLAYER1: 0, Player.PLAYER2: 0}
    game.pieces_on_board = {Player.PLAYER1: 3, Player.PLAYER2: 4}
    game.current_player = Player.PLAYER1
    game._update_phase()
    
    # Flying phase여야 함
    assert game.get_phase() == GamePhase.FLYING
    
    # Player1 (3개 말)은 자유롭게 이동 가능
    valid_moves = game.get_valid_moves()
    # 3개 말 × 17개 빈 위치 = 51개 이동
    assert len(valid_moves) == 3 * 17
    
    # Player1이 자유 이동 가능한지 테스트
    success = game.make_move((0, 23))  # 0번에서 23번으로 자유 이동
    assert success == True
    
    # 이제 Player2 차례 (4개 말이므로 인접 이동만 가능)
    assert game.current_player == Player.PLAYER2
    player2_moves = game.get_valid_moves()
    # Player2는 일반 이동 규칙 적용
    assert len(player2_moves) < 3 * 17  # 자유 이동보다 적어야 함
    
    print("OK Flying phase test passed")

def run_all_tests():
    """모든 테스트 실행"""
    print("Running Nine Men's Morris Game Tests...\n")
    
    try:
        test_basic_functionality()
        test_mill_detection()
        test_neural_network_input()
        test_game_phases()
        test_valid_moves()
        test_game_clone()
        test_flying_phase()
        
        print("\nAll tests passed successfully!")
        print("\nGame is ready for:")
        print("- Human vs Human play")
        print("- MCTS implementation")
        print("- Neural network training (AlphaGo Zero style)")
        
    except AssertionError as e:
        print(f"\nTest failed: {e}")
    except Exception as e:
        print(f"\nUnexpected error: {e}")

if __name__ == "__main__":
    run_all_tests()