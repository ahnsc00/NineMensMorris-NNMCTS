#!/usr/bin/env python3
"""
Nine Men's Morris AlphaZero Training and Evaluation System

This is the main entry point for training and evaluating AlphaZero agents
for Nine Men's Morris game.
"""

import os
import sys
import argparse
from typing import Dict, List

from training_loop import TrainingManager, get_default_config
from evaluation import ModelEvaluator, quick_evaluation_demo
from alphazero_agent import AlphaZeroAgent
from play_game import main as play_human_game


def train_agent(args):
    """Train an AlphaZero agent"""
    print("=" * 60)
    print("ALPHAZERO TRAINING FOR NINE MEN'S MORRIS")
    print("=" * 60)
    
    # Get configuration
    config = get_default_config(args.network)
    
    # Override config with command line arguments
    if args.iterations:
        config['num_iterations'] = args.iterations
    if args.simulations:
        config['num_simulations'] = args.simulations
    if args.games:
        config['self_play_games'] = args.games
    if args.batch_size:
        config['batch_size'] = args.batch_size
    
    # Set directories
    config['model_dir'] = args.model_dir
    config['log_dir'] = args.log_dir
    
    print(f"Configuration:")
    print(f"  Network: {config['network_type']}")
    print(f"  Iterations: {args.iterations}")
    print(f"  MCTS Simulations: {config['num_simulations']}")
    print(f"  Self-play Games: {config['self_play_games']}")
    print(f"  Batch Size: {config['batch_size']}")
    print(f"  Max Workers: {config['max_workers']}")
    print(f"  GPU Optimization: {config['enable_gpu_optimization']}")
    print(f"  Mixed Precision: {config['mixed_precision']}")
    print(f"  Model Directory: {config['model_dir']}")
    print(f"  Log Directory: {config['log_dir']}")
    print()
    
    # Create trainer and run
    trainer = TrainingManager(config)
    trainer.run_training(args.iterations)


def evaluate_agents(args):
    """Evaluate trained agents"""
    print("=" * 60)
    print("AGENT EVALUATION")
    print("=" * 60)
    
    evaluator = ModelEvaluator(args.model_dir)
    
    # Create baseline agents
    evaluator.create_baseline_agents()
    
    # Load trained agents if they exist
    agent_configs = []
    
    # Look for saved models
    model_files = []
    if os.path.exists(args.model_dir):
        for file in os.listdir(args.model_dir):
            if file.endswith('_agent.pkl'):
                model_files.append(file)
    
    # Add found models
    for model_file in model_files:
        model_name = model_file.replace('_agent.pkl', '')
        model_path = os.path.join(args.model_dir, model_name)
        
        # Try to determine network type from saved agent data
        try:
            import pickle
            with open(os.path.join(args.model_dir, model_file), 'rb') as f:
                agent_data = pickle.load(f)
            network_type = agent_data.get('network_type', 'cnn')
        except:
            # Fallback: determine from filename
            if 'cnn' in model_name.lower():
                network_type = 'cnn'
            elif 'resnet' in model_name.lower():
                network_type = 'resnet'
            else:
                network_type = 'cnn'  # Default to CNN for safety
        
        agent_configs.append({
            'name': f"{model_name}_{network_type}",
            'network_type': network_type,
            'model_path': model_path,
            'num_simulations': args.simulations
        })
    
    if agent_configs:
        evaluator.load_agents(agent_configs)
        print(f"Loaded {len(agent_configs)} trained agents")
    else:
        print("No trained agents found. Running with baseline agents only.")
    
    # Run tournament
    print(f"Running tournament with {args.games} games per matchup...")
    results = evaluator.round_robin_tournament(args.games)
    
    # Generate report
    report_file = os.path.join(args.log_dir, 'tournament_report.txt')
    evaluator.generate_report(results, report_file)
    
    # Plot results
    plot_file = os.path.join(args.log_dir, 'tournament_results.png')
    evaluator.plot_tournament_results(results, plot_file)
    
    print(f"Evaluation complete. Results saved to {args.log_dir}")


def play_vs_agent(args):
    """Play against a trained agent"""
    print("=" * 60)
    print("PLAY AGAINST AI")
    print("=" * 60)
    
    # Load agent
    model_path = args.model_path
    if not model_path:
        # Look for best model
        best_model = os.path.join(args.model_dir, 'final_model_best')
        if os.path.exists(best_model + '_agent.pkl'):
            model_path = best_model
        else:
            print("No trained model found. Please specify --model-path or train a model first.")
            return
    
    # Try to determine network type from saved agent data
    try:
        import pickle
        with open(model_path + '_agent.pkl', 'rb') as f:
            agent_data = pickle.load(f)
        network_type = agent_data.get('network_type', 'cnn')
    except:
        # Fallback: determine from filename
        if 'cnn' in model_path.lower():
            network_type = 'cnn'
        else:
            network_type = 'resnet'
    
    print(f"Detected network type: {network_type}")
    
    # Create agent with correct network type
    agent = AlphaZeroAgent(
        network_type=network_type,
        num_simulations=args.simulations,
        temperature=0.0
    )
    
    # Load model
    try:
        agent.load_agent(model_path)
        print(f"Loaded AI agent from {model_path}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        print("This might be due to network architecture mismatch.")
        print("Try training a new model or specify the correct model path.")
        return
    
    # Import and modify the game renderer to work with AI
    from game_renderer import GameRenderer
    from nine_mens_morris import NineMensMorris, Player
    import pygame
    
    # Create game and renderer
    game = NineMensMorris()
    renderer = GameRenderer()
    
    clock = pygame.time.Clock()
    running = True
    
    # Choose who goes first
    try:
        human_is_player1 = input("Do you want to go first? (y/n): ").lower().startswith('y')
    except EOFError:
        human_is_player1 = True  # Default to human first
    
    print(f"You are {'Player 1 (Red)' if human_is_player1 else 'Player 2 (Blue)'}")
    print("Game started! Close the window to quit.")
    
    while running:
        current_player = game.get_current_player()
        is_human_turn = (human_is_player1 and current_player == Player.PLAYER1) or \
                       (not human_is_player1 and current_player == Player.PLAYER2)
        
        if is_human_turn:
            # Human turn - handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        renderer.handle_click(game, event.pos)
        else:
            # AI turn
            if not game.is_game_over()[0]:
                try:
                    action = agent.get_action(game, temperature=0.0)
                    game.make_move(action)
                    print(f"AI played: {action}")
                except Exception as e:
                    print(f"AI error: {e}")
                    # Fallback to random move
                    valid_moves = game.get_valid_moves()
                    if valid_moves:
                        import random
                        action = random.choice(valid_moves)
                        game.make_move(action)
        
        # Check for game end
        game_over, winner = game.is_game_over()
        if game_over:
            if winner is None:
                print("Game Over - Draw!")
            elif (winner == Player.PLAYER1 and human_is_player1) or \
                 (winner == Player.PLAYER2 and not human_is_player1):
                print("Congratulations! You won!")
            else:
                print("AI wins! Better luck next time.")
            
            # Wait for user to close window
            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        waiting = False
                        running = False
        
        # Render game
        renderer.render(game)
        clock.tick(60)
    
    renderer.quit()


def quick_demo():
    """Run a quick demonstration"""
    print("=" * 60)
    print("NINE MEN'S MORRIS ALPHAZERO - QUICK DEMO")
    print("=" * 60)
    
    print("This demo will:")
    print("1. Create baseline agents (Random, Simple CNN)")
    print("2. Run a small tournament")
    print("3. Show results")
    print()
    
    print("Starting demo...")
    
    # Run quick evaluation
    results = quick_evaluation_demo()
    
    print("\nDemo completed!")
    print("Check the generated files:")
    print("- quick_evaluation_report.txt")
    print("- quick_evaluation_results.png")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Nine Men's Morris AlphaZero Training and Evaluation System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Train an AlphaZero agent')
    train_parser.add_argument('--network', choices=['cnn', 'resnet'], default='resnet',
                             help='Neural network architecture')
    train_parser.add_argument('--iterations', type=int, default=50,
                             help='Number of training iterations')
    train_parser.add_argument('--simulations', type=int, default=400,
                             help='MCTS simulations per move')
    train_parser.add_argument('--games', type=int, default=25,
                             help='Self-play games per iteration')
    train_parser.add_argument('--batch-size', type=int, default=32,
                             help='Training batch size')
    train_parser.add_argument('--model-dir', default='models',
                             help='Directory to save models')
    train_parser.add_argument('--log-dir', default='logs',
                             help='Directory to save logs')
    
    # Evaluation command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate trained agents')
    eval_parser.add_argument('--games', type=int, default=20,
                            help='Games per matchup in tournament')
    eval_parser.add_argument('--simulations', type=int, default=400,
                            help='MCTS simulations per move')
    eval_parser.add_argument('--model-dir', default='models',
                            help='Directory containing trained models')
    eval_parser.add_argument('--log-dir', default='logs',
                            help='Directory to save evaluation results')
    
    # Play command
    play_parser = subparsers.add_parser('play', help='Play against a trained agent')
    play_parser.add_argument('--model-path', 
                            help='Path to trained model (without _agent.pkl suffix)')
    play_parser.add_argument('--model-dir', default='models',
                            help='Directory containing trained models')
    play_parser.add_argument('--simulations', type=int, default=400,
                            help='MCTS simulations per move for AI')
    
    # Human vs Human command
    subparsers.add_parser('human', help='Play human vs human')
    
    # Demo command
    subparsers.add_parser('demo', help='Run quick demonstration')
    
    # Performance test command
    subparsers.add_parser('test-performance', help='Test parallel processing performance')
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command == 'train':
        train_agent(args)
    elif args.command == 'evaluate':
        evaluate_agents(args)
    elif args.command == 'play':
        play_vs_agent(args)
    elif args.command == 'human':
        play_human_game()
    elif args.command == 'demo':
        quick_demo()
    elif args.command == 'test-performance':
        from test_parallel_performance import main as test_performance
        test_performance()
    else:
        parser.print_help()
        print("\nAvailable commands:")
        print("  train            - Train an AlphaZero agent")
        print("  evaluate         - Evaluate trained agents")
        print("  play             - Play against a trained agent")
        print("  human            - Play human vs human")
        print("  demo             - Run quick demonstration")
        print("  test-performance - Test parallel processing performance")
        print("\nExample usage:")
        print("  python main.py train --network resnet --iterations 100")
        print("  python main.py evaluate --games 50")
        print("  python main.py play --model-path models/final_model_best")
        print("  python main.py human")
        print("  python main.py demo")


if __name__ == "__main__":
    main()