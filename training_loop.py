import os
import time
import json
import numpy as np
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
from collections import defaultdict
import torch
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import psutil
import threading
from functools import partial

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
        
        # Setup parallel processing
        self.setup_parallel_processing()
        
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
            'self_play_time': [],
            'cpu_usage': [],
            'gpu_usage': [],
            'memory_usage': []
        }
        
        # Best model tracking
        self.best_win_rate = 0.0
        self.iterations_without_improvement = 0
        
        # Resource monitoring
        self.resource_monitor = ResourceMonitor()
        self.monitoring_active = False
        
    def setup_directories(self):
        """Setup directories for saving models and logs"""
        self.model_dir = self.config.get('model_dir', 'models')
        self.log_dir = self.config.get('log_dir', 'logs')
        
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
    
    def setup_parallel_processing(self):
        """Setup parallel processing configuration"""
        # CPU cores
        self.num_cpu_cores = mp.cpu_count()
        self.max_workers = self.config.get('max_workers', max(1, self.num_cpu_cores - 1))
        
        # GPU availability
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gpu_available = torch.cuda.is_available()
        
        print(f"Parallel Processing Setup:")
        print(f"  CPU cores: {self.num_cpu_cores}")
        print(f"  Max workers: {self.max_workers}")
        print(f"  Device: {self.device}")
        print(f"  GPU available: {self.gpu_available}")
        
        if self.gpu_available:
            print(f"  GPU: {torch.cuda.get_device_name()}")
            print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
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
        """Run self-play games in parallel"""
        num_games = self.config.get('self_play_games', 25)
        temperature_schedule = self.config.get('temperature_schedule', None)
        
        print(f"Running {num_games} self-play games in parallel...")
        
        # Start resource monitoring
        self.start_resource_monitoring()
        
        try:
            # Parallel self-play
            if self.max_workers > 1 and num_games > 1:
                examples = self._parallel_self_play(num_games, temperature_schedule)
            else:
                # Fallback to sequential
                examples = self.current_agent.self_play(
                    num_games=num_games,
                    temperature_schedule=temperature_schedule
                )
        finally:
            # Stop resource monitoring
            self.stop_resource_monitoring()
        
        print(f"Generated {len(examples)} training examples")
        return len(examples)
    
    def _parallel_self_play(self, num_games: int, temperature_schedule: Optional[List[float]]) -> List:
        """Run self-play games in parallel using multiple processes"""
        # Distribute games across workers
        games_per_worker = max(1, num_games // self.max_workers)
        remaining_games = num_games % self.max_workers
        
        # Create tasks
        tasks = []
        for i in range(self.max_workers):
            worker_games = games_per_worker + (1 if i < remaining_games else 0)
            if worker_games > 0:
                tasks.append((worker_games, temperature_schedule, self.config))
        
        print(f"  Distributing {num_games} games across {len(tasks)} workers")
        
        all_examples = []
        
        # Use ProcessPoolExecutor for CPU parallelization
        with ProcessPoolExecutor(max_workers=len(tasks)) as executor:
            # Submit tasks
            future_to_task = {
                executor.submit(_worker_self_play, task): task 
                for task in tasks
            }
            
            # Collect results
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    examples = future.result()
                    all_examples.extend(examples)
                    print(f"  Worker completed {task[0]} games -> {len(examples)} examples")
                except Exception as e:
                    print(f"  Worker failed: {e}")
        
        return all_examples
    
    def train_network(self) -> Dict[str, float]:
        """Train the neural network with GPU optimization"""
        batch_size = self.config.get('batch_size', 32)
        training_epochs = self.config.get('training_epochs', 10)
        
        # Optimize batch size for GPU
        if self.gpu_available:
            # Increase batch size for GPU to maximize utilization
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            if gpu_memory_gb >= 8:  # High-end GPU
                batch_size = min(batch_size * 2, 128)
            elif gpu_memory_gb >= 4:  # Mid-range GPU
                batch_size = min(batch_size * 1.5, 96)
            
            # Enable mixed precision training if supported
            if hasattr(torch.cuda, 'is_available') and torch.cuda.get_device_capability()[0] >= 7:
                torch.backends.cudnn.benchmark = True
        
        print(f"Training network for {training_epochs} epochs (batch_size={batch_size})...")
        
        # Start resource monitoring for training
        self.start_resource_monitoring()
        
        try:
            losses = self.current_agent.train(batch_size=batch_size, epochs=training_epochs)
        finally:
            self.stop_resource_monitoring()
        
        print(f"Training loss: {losses['total_loss']:.4f}")
        return losses
    
    def evaluate_agent(self) -> Dict[str, float]:
        """Evaluate current agent against best agent in parallel"""
        eval_games = self.config.get('evaluation_games', 40)
        
        print(f"Evaluating agent over {eval_games} games in parallel...")
        
        # Start resource monitoring
        self.start_resource_monitoring()
        
        try:
            if self.max_workers > 1 and eval_games > 1:
                results = self._parallel_evaluation(eval_games)
            else:
                # Fallback to sequential
                results = self.current_agent.evaluate_against(self.best_agent, num_games=eval_games)
        finally:
            self.stop_resource_monitoring()
        
        print(f"Win rate vs best: {results['win_rate']:.3f} "
              f"({results['wins']}/{eval_games})")
        
        return results
    
    def _parallel_evaluation(self, num_games: int) -> Dict[str, float]:
        """Run evaluation games in parallel"""
        # Distribute games across workers
        games_per_worker = max(1, num_games // self.max_workers)
        remaining_games = num_games % self.max_workers
        
        # Create tasks
        tasks = []
        for i in range(self.max_workers):
            worker_games = games_per_worker + (1 if i < remaining_games else 0)
            if worker_games > 0:
                tasks.append((worker_games, self.config))
        
        print(f"  Distributing {num_games} evaluation games across {len(tasks)} workers")
        
        total_wins = 0
        total_draws = 0
        total_losses = 0
        total_moves = 0
        total_time = 0
        
        # Use ProcessPoolExecutor for CPU parallelization
        with ProcessPoolExecutor(max_workers=len(tasks)) as executor:
            # Submit tasks
            future_to_task = {
                executor.submit(_worker_evaluation, task): task 
                for task in tasks
            }
            
            # Collect results
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    total_wins += result['wins']
                    total_draws += result['draws']
                    total_losses += result['losses']
                    total_moves += result['total_moves']
                    total_time += result['total_time']
                    print(f"  Worker completed {task[0]} evaluation games")
                except Exception as e:
                    print(f"  Evaluation worker failed: {e}")
        
        # Calculate final results
        total_games = total_wins + total_draws + total_losses
        return {
            'wins': total_wins,
            'draws': total_draws,
            'losses': total_losses,
            'win_rate': total_wins / total_games if total_games > 0 else 0.0,
            'avg_game_length': total_moves / total_games if total_games > 0 else 0.0,
            'avg_time_per_game': total_time / total_games if total_games > 0 else 0.0
        }
    
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
    
    def start_resource_monitoring(self):
        """Start resource monitoring in background thread"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.resource_monitor.start_monitoring()
    
    def stop_resource_monitoring(self):
        """Stop resource monitoring and record metrics"""
        if self.monitoring_active:
            self.monitoring_active = False
            metrics = self.resource_monitor.stop_monitoring()
            
            # Add to training history
            self.training_history['cpu_usage'].append(metrics['avg_cpu_usage'])
            self.training_history['gpu_usage'].append(metrics['avg_gpu_usage'])
            self.training_history['memory_usage'].append(metrics['avg_memory_usage'])


class ResourceMonitor:
    """Monitor CPU, GPU, and memory usage during training"""
    
    def __init__(self):
        self.monitoring = False
        self.monitor_thread = None
        self.metrics = {
            'cpu_usage': [],
            'gpu_usage': [],
            'memory_usage': []
        }
    
    def start_monitoring(self):
        """Start monitoring in background thread"""
        if not self.monitoring:
            self.monitoring = True
            self.metrics = {'cpu_usage': [], 'gpu_usage': [], 'memory_usage': []}
            self.monitor_thread = threading.Thread(target=self._monitor_loop)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
    
    def stop_monitoring(self) -> Dict[str, float]:
        """Stop monitoring and return average metrics"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        
        # Calculate averages
        avg_metrics = {
            'avg_cpu_usage': np.mean(self.metrics['cpu_usage']) if self.metrics['cpu_usage'] else 0.0,
            'avg_gpu_usage': np.mean(self.metrics['gpu_usage']) if self.metrics['gpu_usage'] else 0.0,
            'avg_memory_usage': np.mean(self.metrics['memory_usage']) if self.metrics['memory_usage'] else 0.0
        }
        
        return avg_metrics
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=0.1)
                self.metrics['cpu_usage'].append(cpu_percent)
                
                # Memory usage
                memory_percent = psutil.virtual_memory().percent
                self.metrics['memory_usage'].append(memory_percent)
                
                # GPU usage (if available)
                if torch.cuda.is_available():
                    try:
                        gpu_memory_used = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100
                        self.metrics['gpu_usage'].append(gpu_memory_used)
                    except:
                        self.metrics['gpu_usage'].append(0.0)
                else:
                    self.metrics['gpu_usage'].append(0.0)
                
                time.sleep(1.0)  # Sample every second
            except Exception:
                # Continue monitoring even if one measurement fails
                continue


def _worker_self_play(task: Tuple) -> List:
    """Worker function for parallel self-play"""
    num_games, temperature_schedule, config = task
    
    # Create agent for this worker
    agent = AlphaZeroAgent(
        network_type=config['network_type'],
        num_simulations=config['num_simulations'],
        c_puct=config['c_puct'],
        temperature=config['temperature'],
        **config.get('network_kwargs', {})
    )
    
    # Load model if exists
    model_path = os.path.join(config.get('model_dir', 'models'), 'final_model_current')
    if os.path.exists(model_path + '_agent.pkl'):
        try:
            agent.load_agent(model_path)
        except Exception:
            pass  # Use fresh agent if loading fails
    
    # Run self-play
    return agent.self_play(num_games=num_games, temperature_schedule=temperature_schedule)


def _worker_evaluation(task: Tuple) -> Dict:
    """Worker function for parallel evaluation"""
    num_games, config = task
    
    # Create current and best agents for this worker
    current_agent = AlphaZeroAgent(
        network_type=config['network_type'],
        num_simulations=config['num_simulations'],
        c_puct=config['c_puct'],
        temperature=0.0,
        **config.get('network_kwargs', {})
    )
    
    best_agent = AlphaZeroAgent(
        network_type=config['network_type'],
        num_simulations=config['num_simulations'],
        c_puct=config['c_puct'],
        temperature=0.0,
        **config.get('network_kwargs', {})
    )
    
    # Load models
    model_dir = config.get('model_dir', 'models')
    
    current_path = os.path.join(model_dir, 'final_model_current')
    if os.path.exists(current_path + '_agent.pkl'):
        try:
            current_agent.load_agent(current_path)
        except Exception:
            pass
    
    best_path = os.path.join(model_dir, 'final_model_best')
    if os.path.exists(best_path + '_agent.pkl'):
        try:
            best_agent.load_agent(best_path)
        except Exception:
            pass
    
    # Run evaluation
    result = current_agent.evaluate_against(best_agent, num_games=num_games)
    
    # Return detailed metrics for aggregation
    return {
        'wins': result['wins'],
        'draws': result.get('draws', 0),
        'losses': result.get('losses', num_games - result['wins'] - result.get('draws', 0)),
        'total_moves': result.get('avg_game_length', 50) * num_games,
        'total_time': result.get('avg_time_per_game', 1.0) * num_games
    }


def get_default_config(network_type: str = 'resnet') -> Dict:
    """Get default training configuration with parallel processing"""
    # Detect available resources
    num_cores = mp.cpu_count()
    gpu_available = torch.cuda.is_available()
    
    config = {
        'network_type': network_type,
        'num_simulations': 400 if gpu_available else 200,  # More simulations if GPU available
        'c_puct': 1.0,
        'temperature': 1.0,
        'temperature_schedule': [1.0] * 10 + [0.5] * 10 + [0.1],  # Decay temperature
        
        # Training parameters - optimized for parallel processing
        'self_play_games': max(25, num_cores * 4),  # Scale with CPU cores
        'batch_size': 64 if gpu_available else 32,  # Larger batches for GPU
        'training_epochs': 10,
        'evaluation_games': max(20, num_cores * 2),  # Scale evaluation games
        'improvement_threshold': 0.55,
        'max_iterations_without_improvement': 30,
        
        # Parallel processing parameters
        'max_workers': max(1, num_cores - 1),  # Leave one core for system
        'enable_gpu_optimization': gpu_available,
        'mixed_precision': gpu_available and torch.cuda.get_device_capability()[0] >= 7,
        
        # Network parameters
        'network_kwargs': {
            'input_channels': 3,
            'num_actions': 600,
            'learning_rate': 0.002 if gpu_available else 0.001  # Higher LR for GPU
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