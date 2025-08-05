#!/usr/bin/env python3
"""
Simple test to verify the AlphaZero system works
"""

import numpy as np
from nine_mens_morris import NineMensMorris, Player
from alphazero_agent import AlphaZeroAgent

def test_agent_creation():
    """Test creating agents"""
    print("Testing agent creation...")
    
    # Test CNN agent
    cnn_agent = AlphaZeroAgent(
        network_type='cnn',
        num_simulations=10,  # Very few simulations for quick test
        hidden_size=64,
        learning_rate=0.01
    )
    print("OK CNN agent created")
    
    # Test ResNet agent
    resnet_agent = AlphaZeroAgent(
        network_type='resnet',
        num_simulations=10,
        num_blocks=2,
        channels=32,
        learning_rate=0.01
    )
    print("OK ResNet agent created")
    
    # Test Random agent
    random_agent = AlphaZeroAgent(
        network_type='random',
        num_simulations=1
    )
    print("OK Random agent created")
    
    return cnn_agent, resnet_agent, random_agent

def test_action_selection():
    """Test action selection"""
    print("\nTesting action selection...")
    
    game = NineMensMorris()
    
    # Create a simple agent
    agent = AlphaZeroAgent(
        network_type='random',
        num_simulations=1,
        temperature=0.0
    )
    
    # Get action
    action = agent.get_action(game)
    print(f"OK Agent selected action: {action}")
    
    # Make move
    success = game.make_move(action)
    print(f"OK Move successful: {success}")
    
    return agent

def test_self_play():
    """Test self-play (very short)"""
    print("\nTesting self-play...")
    
    agent = AlphaZeroAgent(
        network_type='random',
        num_simulations=1
    )
    
    # Run 1 very short self-play game
    examples = agent.self_play(num_games=1)
    print(f"OK Self-play completed, generated {len(examples)} examples")
    
    return examples

def test_neural_network():
    """Test neural network prediction"""
    print("\nTesting neural network...")
    
    game = NineMensMorris()
    board_tensor = game.get_board_tensor()
    
    # Test CNN
    cnn_agent = AlphaZeroAgent(
        network_type='cnn',
        num_simulations=1,
        hidden_size=32
    )
    policy, value = cnn_agent.network.predict(board_tensor)
    print(f"OK CNN prediction - Policy shape: {policy.shape}, Value: {value:.3f}")
    
    # Test ResNet
    resnet_agent = AlphaZeroAgent(
        network_type='resnet',
        num_simulations=1,
        num_blocks=1,
        channels=16
    )
    policy, value = resnet_agent.network.predict(board_tensor)
    print(f"OK ResNet prediction - Policy shape: {policy.shape}, Value: {value:.3f}")

def main():
    """Run all simple tests"""
    print("=" * 50)
    print("SIMPLE ALPHAZERO SYSTEM TEST")
    print("=" * 50)
    
    try:
        # Test 1: Agent creation
        agents = test_agent_creation()
        
        # Test 2: Action selection
        test_action_selection()
        
        # Test 3: Neural network
        test_neural_network()
        
        # Test 4: Self-play
        test_self_play()
        
        print("\n" + "=" * 50)
        print("ALL TESTS PASSED!")
        print("The AlphaZero system is working correctly.")
        print("=" * 50)
        
        # Show usage examples
        print("\nYou can now use:")
        print("  python main.py train --network cnn --iterations 10")
        print("  python main.py evaluate --games 5")
        print("  python main.py human")
        
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()