import os
import time
import json
import numpy as np
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
from collections import defaultdict
import torch

from alphazero_agent import AlphaZeroAgent
from neural_networks import create_network


class TrainingManager:
    """Manages the AlphaZero training process"""
    
    def __init__(self, config: Dict):
        """
        Initialize training manager
        
        Args:
            config: Training configuration dictionary
        """
        self.config = config
        self.setup_directories()
        
        # Create agents
        self.current_agent = AlphaZeroAgent(
            network_type=config['network_type'],
            num_simulations=config['num_simulations'],
            c_puct=config['c_puct'],
            temperature=config['temperature'],
            **config.get('network_kwargs', {})
        )
        
        self.best_agent = AlphaZeroAgent(
            network_type=config['network_type'],
            num_simulations=config['num_simulations'],
            c_puct=config['c_puct'],
            temperature=0.0,  # Deterministic for evaluation
            **config.get('network_kwargs', {})
        )
        
        # Try to load existing models
        self.load_existing_models()
        
        # Training history
        self.training_history = {
            'iteration': [],
            'total_loss': [],
            'policy_loss': [],
            'value_loss': [],
            'win_rate_vs_best': [],
            'games_played': [],
            'avg_game_length': [],
            'evaluation_time': [],
            'training_time': [],
            'self_play_time': []
        }
        
        # Best model tracking
        self.best_win_rate = 0.0
        self.iterations_without_improvement = 0
        
    def setup_directories(self):
        """Setup directories for saving models and logs"""
        self.model_dir = self.config.get('model_dir', 'models')
        self.log_dir = self.config.get('log_dir', 'logs')
        
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
    
    def load_existing_models(self):
        """Load existing models if available"""
        current_model_path = os.path.join(self.model_dir, 'final_model_current')
        best_model_path = os.path.join(self.model_dir, 'final_model_best')
        
        # Try to load current model
        if os.path.exists(current_model_path + '_agent.pkl'):
            try:
                self.current_agent.load_agent(current_model_path)
                print(f"[OK] Loaded existing current model from {current_model_path}")
            except Exception as e:
                print(f"[WARNING] Failed to load current model: {e}")
        
        # Try to load best model
        if os.path.exists(best_model_path + '_agent.pkl'):
            try:
                self.best_agent.load_agent(best_model_path)
                print(f"[OK] Loaded existing best model from {best_model_path}")
            except Exception as e:
                print(f"[WARNING] Failed to load best model: {e}")
        
        # Try to load training history
        history_path = os.path.join(self.log_dir, 'complete_training_history.json')
        if os.path.exists(history_path):
            try:
                with open(history_path, 'r') as f:
                    self.training_history = json.load(f)
                print(f"[OK] Loaded training history with {len(self.training_history['iteration'])} iterations")
            except Exception as e:
                print(f"[WARNING] Failed to load training history: {e}")
    
    def run_training(self, num_iterations: int = 100):
        """Run the main training loop"""
        # Determine starting iteration
        start_iteration = len(self.training_history['iteration'])
        total_iterations = start_iteration + num_iterations
        
        if start_iteration > 0:
            print(f"Resuming AlphaZero training from iteration {start_iteration + 1}")
            print(f"Will train for {num_iterations} more iterations (total: {total_iterations})")
        else:
            print(f"Starting AlphaZero training for {num_iterations} iterations")
        
        print(f"Network type: {self.config['network_type']}")
        print(f"MCTS simulations: {self.config['num_simulations']}")
        print("-" * 60)
        
        for i in range(num_iterations):
            iteration = start_iteration + i + 1
            print(f"\nIteration {iteration}/{total_iterations}")
            
            # Self-play
            start_time = time.time()
            self_play_examples = self.run_self_play()
            self_play_time = time.time() - start_time
            
            # Training
            start_time = time.time()
            training_losses = self.train_network()
            training_time = time.time() - start_time
            
            # Evaluation
            start_time = time.time()
            evaluation_results = self.evaluate_agent()
            evaluation_time = time.time() - start_time
            
            # Update training history
            self.update_history(iteration, training_losses, evaluation_results,
                              self_play_time, training_time, evaluation_time)
            
            # Model management
            self.manage_models(evaluation_results['win_rate'])
            
            # Logging
            self.log_iteration(iteration, training_losses, evaluation_results,
                             self_play_time, training_time, evaluation_time)
            
            # Save checkpoint
            if iteration % self.config.get('checkpoint_freq', 10) == 0:
                self.save_checkpoint(iteration)
            
            # Early stopping check
            if self.check_early_stopping():
                print(f"Early stopping at iteration {iteration}")
                break
        
        # Final save and plotting
        self.save_final_results()
        self.plot_training_curves()
        
        print("\nTraining completed!")
    
    def run_self_play(self) -> int:
        """Run self-play games"""
        num_games = self.config.get('self_play_games', 25)
        temperature_schedule = self.config.get('temperature_schedule', None)
        
        print(f"Running {num_games} self-play games...")
        
        examples = self.current_agent.self_play(
            num_games=num_games,
            temperature_schedule=temperature_schedule
        )
        
        print(f"Generated {len(examples)} training examples")
        return len(examples)
    
    def train_network(self) -> Dict[str, float]:
        """Train the neural network"""
        batch_size = self.config.get('batch_size', 32)
        training_epochs = self.config.get('training_epochs', 10)
        
        print(f"Training network for {training_epochs} epochs...")
        
        losses = self.current_agent.train(batch_size=batch_size, epochs=training_epochs)
        
        print(f"Training loss: {losses['total_loss']:.4f}")
        return losses
    
    def evaluate_agent(self) -> Dict[str, float]:
        """Evaluate current agent against best agent"""
        eval_games = self.config.get('evaluation_games', 40)
        
        print(f"Evaluating agent over {eval_games} games...")
        
        results = self.current_agent.evaluate_against(self.best_agent, num_games=eval_games)
        
        print(f"Win rate vs best: {results['win_rate']:.3f} "
              f"({results['wins']}/{eval_games})")
        
        return results
    
    def manage_models(self, win_rate: float):
        """Manage best model updates"""
        improvement_threshold = self.config.get('improvement_threshold', 0.55)
        
        if win_rate >= improvement_threshold:
            print(f"New best model! Win rate: {win_rate:.3f}")
            
            # Copy current agent to best agent
            self.best_agent = AlphaZeroAgent(
                network_type=self.config['network_type'],
                num_simulations=self.config['num_simulations'],
                c_puct=self.config['c_puct'],
                temperature=0.0,
                **self.config.get('network_kwargs', {})
            )
            
            # Copy network weights
            self.best_agent.network = create_network(
                self.config['network_type'],
                **self.config.get('network_kwargs', {})
            )
            
            # Copy state dict
            if hasattr(self.current_agent.network, 'state_dict'):
                self.best_agent.network.load_state_dict(
                    self.current_agent.network.state_dict()
                )
            
            self.best_win_rate = win_rate
            self.iterations_without_improvement = 0
        else:
            self.iterations_without_improvement += 1
    
    def update_history(self, iteration: int, losses: Dict[str, float], 
                      eval_results: Dict[str, float], self_play_time: float,
                      training_time: float, evaluation_time: float):
        """Update training history"""
        stats = self.current_agent.get_stats()
        
        self.training_history['iteration'].append(iteration)
        self.training_history['total_loss'].append(losses['total_loss'])
        self.training_history['policy_loss'].append(losses['policy_loss'])
        self.training_history['value_loss'].append(losses['value_loss'])
        self.training_history['win_rate_vs_best'].append(eval_results['win_rate'])
        self.training_history['games_played'].append(stats['games_played'])
        self.training_history['avg_game_length'].append(stats['avg_game_length'])
        self.training_history['self_play_time'].append(self_play_time)
        self.training_history['training_time'].append(training_time)
        self.training_history['evaluation_time'].append(evaluation_time)
    
    def log_iteration(self, iteration: int, losses: Dict[str, float],
                     eval_results: Dict[str, float], self_play_time: float,
                     training_time: float, evaluation_time: float):
        """Log iteration results"""
        stats = self.current_agent.get_stats()
        
        total_time = self_play_time + training_time + evaluation_time
        
        print(f"Losses - Total: {losses['total_loss']:.4f}, "
              f"Policy: {losses['policy_loss']:.4f}, "
              f"Value: {losses['value_loss']:.4f}")
        print(f"Games played: {stats['games_played']}, "
              f"Avg length: {stats['avg_game_length']:.1f}")
        print(f"Times - Self-play: {self_play_time:.1f}s, "
              f"Training: {training_time:.1f}s, "
              f"Evaluation: {evaluation_time:.1f}s, "
              f"Total: {total_time:.1f}s")
        print(f"Iterations without improvement: {self.iterations_without_improvement}")
    
    def check_early_stopping(self) -> bool:
        """Check if early stopping criteria are met"""
        max_no_improvement = self.config.get('max_iterations_without_improvement', 50)
        
        return self.iterations_without_improvement >= max_no_improvement
    
    def save_checkpoint(self, iteration: int):
        """Save training checkpoint"""
        checkpoint_path = os.path.join(self.model_dir, f'checkpoint_iter_{iteration}')
        
        # Save current agent
        self.current_agent.save_agent(checkpoint_path + '_current')
        
        # Save best agent
        self.best_agent.save_agent(checkpoint_path + '_best')
        
        # Save training history
        history_path = os.path.join(self.log_dir, f'training_history_iter_{iteration}.json')
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        print(f"Checkpoint saved at iteration {iteration}")
    
    def save_final_results(self):
        """Save final training results"""
        # Save final models
        final_path = os.path.join(self.model_dir, 'final_model')
        self.current_agent.save_agent(final_path + '_current')
        self.best_agent.save_agent(final_path + '_best')
        
        # Save complete training history
        history_path = os.path.join(self.log_dir, 'complete_training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        # Save config
        config_path = os.path.join(self.log_dir, 'training_config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def plot_training_curves(self):
        """Plot training curves"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Training Curves - {self.config["network_type"].upper()}', fontsize=16)
        
        # Loss curves
        axes[0, 0].plot(self.training_history['iteration'], self.training_history['total_loss'])
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(self.training_history['iteration'], self.training_history['policy_loss'], label='Policy')
        axes[0, 1].plot(self.training_history['iteration'], self.training_history['value_loss'], label='Value')
        axes[0, 1].set_title('Policy and Value Loss')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Win rate
        axes[0, 2].plot(self.training_history['iteration'], self.training_history['win_rate_vs_best'])
        axes[0, 2].axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
        axes[0, 2].set_title('Win Rate vs Best Model')
        axes[0, 2].set_xlabel('Iteration')
        axes[0, 2].set_ylabel('Win Rate')
        axes[0, 2].grid(True)
        
        # Game statistics
        axes[1, 0].plot(self.training_history['iteration'], self.training_history['games_played'])
        axes[1, 0].set_title('Total Games Played')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Games')
        axes[1, 0].grid(True)
        
        axes[1, 1].plot(self.training_history['iteration'], self.training_history['avg_game_length'])
        axes[1, 1].set_title('Average Game Length')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Moves')
        axes[1, 1].grid(True)
        
        # Timing
        axes[1, 2].plot(self.training_history['iteration'], self.training_history['self_play_time'], label='Self-play')
        axes[1, 2].plot(self.training_history['iteration'], self.training_history['training_time'], label='Training')
        axes[1, 2].plot(self.training_history['iteration'], self.training_history['evaluation_time'], label='Evaluation')
        axes[1, 2].set_title('Time per Iteration')
        axes[1, 2].set_xlabel('Iteration')
        axes[1, 2].set_ylabel('Time (seconds)')
        axes[1, 2].legend()
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.log_dir, f'training_curves_{self.config["network_type"]}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        
        # Only show plot if in interactive mode
        try:
            import matplotlib
            if matplotlib.get_backend() != 'Agg':
                plt.show()
            else:
                print(f"Training curves saved to {plot_path}")
        except:
            print(f"Training curves saved to {plot_path}")
        
        plt.close()  # Close the figure to free memory


def get_default_config(network_type: str = 'resnet') -> Dict:
    """Get default training configuration"""
    config = {
        'network_type': network_type,
        'num_simulations': 200,  # Reduced for faster training
        'c_puct': 1.0,
        'temperature': 1.0,
        'temperature_schedule': [1.0] * 10 + [0.5] * 10 + [0.1],  # Decay temperature
        
        # Training parameters
        'self_play_games': 25,
        'batch_size': 32,
        'training_epochs': 10,
        'evaluation_games': 20,  # Reduced from 40 for faster training
        'improvement_threshold': 0.55,
        'max_iterations_without_improvement': 30,
        
        # Network parameters
        'network_kwargs': {
            'input_channels': 3,
            'num_actions': 600,
            'learning_rate': 0.001
        },
        
        # Logging
        'model_dir': 'models',
        'log_dir': 'logs',
        'checkpoint_freq': 10
    }
    
    # Network-specific configurations
    if network_type.lower() == 'cnn':
        config['network_kwargs'].update({
            'hidden_size': 256
        })
    elif network_type.lower() == 'resnet':
        config['network_kwargs'].update({
            'num_blocks': 6,
            'channels': 128
        })
    
    return config


def main():
    """Main training function"""
    # Choose network type
    network_type = input("Choose network type (cnn/resnet) [resnet]: ").strip().lower()
    if not network_type:
        network_type = 'resnet'
    
    if network_type not in ['cnn', 'resnet']:
        print("Invalid network type. Using 'resnet'")
        network_type = 'resnet'
    
    # Get configuration
    config = get_default_config(network_type)
    
    # Ask for training iterations
    try:
        num_iterations = int(input("Number of training iterations [50]: ") or "50")
    except ValueError:
        num_iterations = 50
    
    print(f"\nTraining configuration:")
    print(f"Network: {network_type}")
    print(f"Iterations: {num_iterations}")
    print(f"MCTS simulations: {config['num_simulations']}")
    print(f"Self-play games per iteration: {config['self_play_games']}")
    
    # Create and run training manager
    trainer = TrainingManager(config)
    trainer.run_training(num_iterations)


if __name__ == "__main__":
    main()