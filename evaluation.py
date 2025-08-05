import os
import time
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import random
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

from nine_mens_morris import NineMensMorris, Player
from alphazero_agent import AlphaZeroAgent
from neural_networks import create_network


class ModelEvaluator:
    """Evaluate and compare different models"""
    
    def __init__(self, models_dir: str = 'models', max_workers: Optional[int] = None):
        self.models_dir = models_dir
        self.agents = {}
        self.evaluation_results = {}
        
        # Parallel processing setup
        self.max_workers = max_workers or max(1, mp.cpu_count() - 1)
        print(f"ModelEvaluator initialized with {self.max_workers} max workers")
    
    def load_agents(self, agent_configs: List[Dict]):
        """
        Load agents from configurations
        
        Args:
            agent_configs: List of dictionaries with agent configurations
                Each dict should have: name, network_type, model_path, and network_kwargs
        """
        for config in agent_configs:
            name = config['name']
            network_type = config['network_type']
            model_path = config.get('model_path', None)
            network_kwargs = config.get('network_kwargs', {})
            
            # Create agent
            agent = AlphaZeroAgent(
                network_type=network_type,
                num_simulations=config.get('num_simulations', 400),
                c_puct=config.get('c_puct', 1.0),
                temperature=0.0,  # Deterministic for evaluation
                **network_kwargs
            )
            
            # Load model if path provided
            if model_path and os.path.exists(model_path + '_agent.pkl'):
                agent.load_agent(model_path)
                print(f"Loaded {name} from {model_path}")
            else:
                print(f"Created new {name} (no saved model found)")
            
            self.agents[name] = agent
    
    def create_baseline_agents(self):
        """Create baseline agents for comparison"""
        # Random agent
        random_agent = AlphaZeroAgent(
            network_type='random',
            num_simulations=1,  # No MCTS needed for random
            c_puct=1.0,
            temperature=0.0
        )
        self.agents['Random'] = random_agent
        
        # Simple CNN agent
        simple_cnn = AlphaZeroAgent(
            network_type='cnn',
            num_simulations=100,  # Fewer simulations
            c_puct=1.0,
            temperature=0.0,
            hidden_size=128,
            learning_rate=0.001
        )
        self.agents['Simple_CNN'] = simple_cnn
    
    def round_robin_tournament(self, num_games_per_match: int = 20) -> Dict[str, Dict]:
        """
        Run round-robin tournament between all agents
        
        Args:
            num_games_per_match: Number of games per matchup
            
        Returns:
            Tournament results
        """
        agent_names = list(self.agents.keys())
        results = defaultdict(lambda: defaultdict(dict))
        
        print(f"Running round-robin tournament with {len(agent_names)} agents")
        print(f"Games per matchup: {num_games_per_match}")
        print("-" * 50)
        
        total_matches = len(agent_names) * (len(agent_names) - 1) // 2
        match_count = 0
        
        for i, agent1_name in enumerate(agent_names):
            for j, agent2_name in enumerate(agent_names[i+1:], i+1):
                match_count += 1
                print(f"Match {match_count}/{total_matches}: {agent1_name} vs {agent2_name}")
                
                # Play games
                match_results = self.play_match(
                    self.agents[agent1_name],
                    self.agents[agent2_name],
                    num_games_per_match
                )
                
                # Store results
                results[agent1_name][agent2_name] = match_results
                
                # Store reverse results
                reverse_results = {
                    'wins': match_results['losses'],
                    'losses': match_results['wins'],
                    'draws': match_results['draws'],
                    'win_rate': 1.0 - match_results['win_rate'],
                    'avg_game_length': match_results['avg_game_length'],
                    'avg_time_per_move': match_results['avg_time_per_move']
                }
                results[agent2_name][agent1_name] = reverse_results
                
                print(f"  {agent1_name}: {match_results['win_rate']:.3f} win rate")
                print(f"  Avg game length: {match_results['avg_game_length']:.1f} moves")
                print()
        
        # Calculate overall statistics
        overall_stats = self.calculate_overall_stats(results, agent_names)
        
        tournament_results = {
            'matchup_results': dict(results),
            'overall_stats': overall_stats,
            'tournament_info': {
                'num_agents': len(agent_names),
                'games_per_match': num_games_per_match,
                'total_games': total_matches * num_games_per_match
            }
        }
        
        return tournament_results
    
    def play_match(self, agent1: AlphaZeroAgent, agent2: AlphaZeroAgent, 
                   num_games: int) -> Dict[str, float]:
        """Play a match between two agents with parallel processing"""
        print(f"  Playing {num_games} games...")
        
        if self.max_workers > 1 and num_games > 1:
            # Parallel execution
            return self._play_match_parallel(agent1, agent2, num_games)
        else:
            # Sequential execution (fallback)
            return self._play_match_sequential(agent1, agent2, num_games)
    
    def _play_match_sequential(self, agent1: AlphaZeroAgent, agent2: AlphaZeroAgent, 
                              num_games: int) -> Dict[str, float]:
        """Play a match sequentially (original implementation)"""
        wins = 0
        losses = 0
        draws = 0
        total_moves = 0
        total_time = 0
        
        for game_idx in range(num_games):
            start_time = time.time()
            
            # Alternate who goes first
            if game_idx % 2 == 0:
                result, moves = self.play_single_game(agent1, agent2)
            else:
                result, moves = self.play_single_game(agent2, agent1)
                result = -result  # Flip result for agent1's perspective
            
            game_time = time.time() - start_time
            total_time += game_time
            total_moves += moves
            
            if result == 1:
                wins += 1
            elif result == -1:
                losses += 1
            else:
                draws += 1
        
        return {
            'wins': wins,
            'losses': losses,
            'draws': draws,
            'win_rate': wins / num_games,
            'avg_game_length': total_moves / num_games,
            'avg_time_per_move': total_time / total_moves if total_moves > 0 else 0
        }
    
    def _play_match_parallel(self, agent1: AlphaZeroAgent, agent2: AlphaZeroAgent, 
                            num_games: int) -> Dict[str, float]:
        """Play a match using parallel processing"""
        # Distribute games across workers
        games_per_worker = max(1, num_games // self.max_workers)
        remaining_games = num_games % self.max_workers
        
        # Create tasks for workers
        tasks = []
        for i in range(self.max_workers):
            worker_games = games_per_worker + (1 if i < remaining_games else 0)
            if worker_games > 0:
                # Create agent configs for serialization
                agent1_config = self._extract_agent_config(agent1)
                agent2_config = self._extract_agent_config(agent2)
                tasks.append((agent1_config, agent2_config, worker_games, i))
        
        # Execute tasks in parallel
        total_wins = 0
        total_losses = 0
        total_draws = 0
        total_moves = 0
        total_time = 0
        
        with ProcessPoolExecutor(max_workers=len(tasks)) as executor:
            # Submit tasks
            future_to_task = {
                executor.submit(_worker_play_match, task): task 
                for task in tasks
            }
            
            # Collect results
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    total_wins += result['wins']
                    total_losses += result['losses']
                    total_draws += result['draws']
                    total_moves += result['total_moves']
                    total_time += result['total_time']
                    print(f"    Worker {task[3]} completed {task[2]} games")
                except Exception as e:
                    print(f"    Worker {task[3]} failed: {e}")
        
        return {
            'wins': total_wins,
            'losses': total_losses,
            'draws': total_draws,
            'win_rate': total_wins / num_games if num_games > 0 else 0.0,
            'avg_game_length': total_moves / num_games if num_games > 0 else 0.0,
            'avg_time_per_move': total_time / total_moves if total_moves > 0 else 0.0
        }
    
    def _extract_agent_config(self, agent: AlphaZeroAgent) -> Dict:
        """Extract agent configuration for serialization"""
        return {
            'network_type': agent.network_type,
            'num_simulations': agent.num_simulations,
            'c_puct': agent.c_puct,
            'temperature': agent.temperature,
            'model_path': None  # Will be handled by worker
        }
    
    def play_single_game(self, player1: AlphaZeroAgent, player2: AlphaZeroAgent) -> Tuple[int, int]:
        """Play a single game between two agents"""
        game = NineMensMorris()
        move_count = 0
        max_moves = 300  # Prevent infinite games
        
        while move_count < max_moves:
            current_player = game.get_current_player()
            
            # Select agent
            if current_player == Player.PLAYER1:
                agent = player1
            else:
                agent = player2
            
            # Get action
            try:
                action = agent.get_action(game, temperature=0.0)
            except:
                # If agent fails, random move
                valid_moves = game.get_valid_moves()
                action = random.choice(valid_moves) if valid_moves else (0,)
            
            # Make move
            success = game.make_move(action)
            if not success:
                # Invalid move, opponent wins
                winner = Player.PLAYER2 if current_player == Player.PLAYER1 else Player.PLAYER1
                result = 1 if winner == Player.PLAYER1 else -1
                return result, move_count
            
            move_count += 1
            
            # Check for game end
            game_over, winner = game.is_game_over()
            if game_over:
                if winner is None:
                    return 0, move_count  # Draw
                elif winner == Player.PLAYER1:
                    return 1, move_count  # Player 1 wins
                else:
                    return -1, move_count  # Player 2 wins
        
        return 0, move_count  # Draw (game too long)
    
    def calculate_overall_stats(self, results: Dict, agent_names: List[str]) -> Dict[str, Dict]:
        """Calculate overall tournament statistics"""
        overall_stats = {}
        
        for agent_name in agent_names:
            total_games = 0
            total_wins = 0
            total_draws = 0
            total_moves = 0
            opponents_beaten = 0
            
            for opponent in agent_names:
                if agent_name != opponent:
                    match_result = results[agent_name][opponent]
                    games_played = match_result['wins'] + match_result['losses'] + match_result['draws']
                    
                    total_games += games_played
                    total_wins += match_result['wins']
                    total_draws += match_result['draws']
                    total_moves += match_result['avg_game_length'] * games_played
                    
                    if match_result['win_rate'] > 0.5:
                        opponents_beaten += 1
            
            overall_stats[agent_name] = {
                'total_games': total_games,
                'overall_win_rate': total_wins / total_games if total_games > 0 else 0,
                'draw_rate': total_draws / total_games if total_games > 0 else 0,
                'avg_game_length': total_moves / total_games if total_games > 0 else 0,
                'opponents_beaten': opponents_beaten,
                'total_opponents': len(agent_names) - 1
            }
        
        return overall_stats
    
    def generate_report(self, tournament_results: Dict, output_file: str = 'tournament_report.txt'):
        """Generate detailed tournament report"""
        with open(output_file, 'w') as f:
            f.write("NINE MEN'S MORRIS - TOURNAMENT REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Tournament info
            info = tournament_results['tournament_info']
            f.write(f"Tournament Information:\n")
            f.write(f"  Number of agents: {info['num_agents']}\n")
            f.write(f"  Games per matchup: {info['games_per_match']}\n")
            f.write(f"  Total games played: {info['total_games']}\n\n")
            
            # Overall rankings
            overall_stats = tournament_results['overall_stats']
            sorted_agents = sorted(overall_stats.items(), 
                                 key=lambda x: x[1]['overall_win_rate'], 
                                 reverse=True)
            
            f.write("OVERALL RANKINGS:\n")
            f.write("-" * 30 + "\n")
            for rank, (agent_name, stats) in enumerate(sorted_agents, 1):
                f.write(f"{rank}. {agent_name}\n")
                f.write(f"   Win Rate: {stats['overall_win_rate']:.3f}\n")
                f.write(f"   Draw Rate: {stats['draw_rate']:.3f}\n")
                f.write(f"   Opponents Beaten: {stats['opponents_beaten']}/{stats['total_opponents']}\n")
                f.write(f"   Avg Game Length: {stats['avg_game_length']:.1f} moves\n\n")
            
            # Detailed matchup results
            f.write("DETAILED MATCHUP RESULTS:\n")
            f.write("-" * 40 + "\n")
            
            matchup_results = tournament_results['matchup_results']
            for agent1 in sorted([name for name, _ in sorted_agents]):
                f.write(f"\n{agent1} vs others:\n")
                for agent2 in sorted([name for name, _ in sorted_agents]):
                    if agent1 != agent2:
                        result = matchup_results[agent1][agent2]
                        f.write(f"  vs {agent2}: {result['win_rate']:.3f} "
                               f"({result['wins']}-{result['losses']}-{result['draws']})\n")
        
        print(f"Tournament report saved to {output_file}")
    
    def plot_tournament_results(self, tournament_results: Dict, output_file: str = 'tournament_results.png'):
        """Plot tournament results"""
        overall_stats = tournament_results['overall_stats']
        agent_names = list(overall_stats.keys())
        
        # Sort by win rate
        sorted_items = sorted(overall_stats.items(), 
                            key=lambda x: x[1]['overall_win_rate'], 
                            reverse=True)
        sorted_names = [name for name, _ in sorted_items]
        win_rates = [stats['overall_win_rate'] for _, stats in sorted_items]
        draw_rates = [stats['draw_rate'] for _, stats in sorted_items]
        avg_lengths = [stats['avg_game_length'] for _, stats in sorted_items]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Tournament Results', fontsize=16)
        
        # Win rates
        axes[0, 0].bar(range(len(sorted_names)), win_rates, color='skyblue')
        axes[0, 0].set_title('Overall Win Rates')
        axes[0, 0].set_ylabel('Win Rate')
        axes[0, 0].set_xticks(range(len(sorted_names)))
        axes[0, 0].set_xticklabels(sorted_names, rotation=45, ha='right')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Win + Draw rates
        bottom_rates = np.array(win_rates)
        axes[0, 1].bar(range(len(sorted_names)), win_rates, color='skyblue', label='Wins')
        axes[0, 1].bar(range(len(sorted_names)), draw_rates, bottom=bottom_rates, 
                      color='lightgreen', label='Draws')
        axes[0, 1].set_title('Win and Draw Rates')
        axes[0, 1].set_ylabel('Rate')
        axes[0, 1].set_xticks(range(len(sorted_names)))
        axes[0, 1].set_xticklabels(sorted_names, rotation=45, ha='right')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Average game length
        axes[1, 0].bar(range(len(sorted_names)), avg_lengths, color='lightcoral')
        axes[1, 0].set_title('Average Game Length')
        axes[1, 0].set_ylabel('Moves')
        axes[1, 0].set_xticks(range(len(sorted_names)))
        axes[1, 0].set_xticklabels(sorted_names, rotation=45, ha='right')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Head-to-head heatmap
        matchup_matrix = np.zeros((len(agent_names), len(agent_names)))
        matchup_results = tournament_results['matchup_results']
        
        for i, agent1 in enumerate(sorted_names):
            for j, agent2 in enumerate(sorted_names):
                if agent1 != agent2:
                    matchup_matrix[i, j] = matchup_results[agent1][agent2]['win_rate']
                else:
                    matchup_matrix[i, j] = 0.5  # Self-play
        
        im = axes[1, 1].imshow(matchup_matrix, cmap='RdYlBu_r', vmin=0, vmax=1)
        axes[1, 1].set_title('Head-to-Head Win Rates')
        axes[1, 1].set_xticks(range(len(sorted_names)))
        axes[1, 1].set_yticks(range(len(sorted_names)))
        axes[1, 1].set_xticklabels(sorted_names, rotation=45, ha='right')
        axes[1, 1].set_yticklabels(sorted_names)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=axes[1, 1])
        cbar.set_label('Win Rate')
        
        # Add text annotations
        for i in range(len(sorted_names)):
            for j in range(len(sorted_names)):
                if i != j:
                    text = axes[1, 1].text(j, i, f'{matchup_matrix[i, j]:.2f}',
                                         ha="center", va="center", color="white" if matchup_matrix[i, j] < 0.5 else "black")
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        
        # Only show plot if in interactive mode
        try:
            import matplotlib
            if matplotlib.get_backend() != 'Agg':
                plt.show(block=False)
                plt.pause(0.1)  # Brief pause to allow plot to render
        except:
            pass  # Skip showing if not in interactive mode
        finally:
            plt.close()  # Always close the plot to free memory
        
        print(f"Tournament results plot saved to {output_file}")


def _worker_play_match(task: Tuple) -> Dict:
    """Worker function for parallel match evaluation"""
    agent1_config, agent2_config, num_games, worker_id = task
    
    # Create agents for this worker
    agent1 = AlphaZeroAgent(
        network_type=agent1_config['network_type'],
        num_simulations=agent1_config['num_simulations'],
        c_puct=agent1_config['c_puct'],
        temperature=agent1_config['temperature']
    )
    
    agent2 = AlphaZeroAgent(
        network_type=agent2_config['network_type'],
        num_simulations=agent2_config['num_simulations'],
        c_puct=agent2_config['c_puct'],
        temperature=agent2_config['temperature']
    )
    
    # Load models if they exist (best effort)
    try:
        if os.path.exists('models/final_model_current_agent.pkl'):
            agent1.load_agent('models/final_model_current')
        if os.path.exists('models/final_model_best_agent.pkl'):
            agent2.load_agent('models/final_model_best')
    except Exception:
        pass  # Use fresh agents if loading fails
    
    # Play games
    wins = 0
    losses = 0
    draws = 0
    total_moves = 0
    total_time = 0
    
    for game_idx in range(num_games):
        start_time = time.time()
        
        # Alternate who goes first
        if game_idx % 2 == 0:
            result, moves = _play_single_game_worker(agent1, agent2)
        else:
            result, moves = _play_single_game_worker(agent2, agent1)
            result = -result  # Flip result for agent1's perspective
        
        game_time = time.time() - start_time
        total_time += game_time
        total_moves += moves
        
        if result == 1:
            wins += 1
        elif result == -1:
            losses += 1
        else:
            draws += 1
    
    return {
        'wins': wins,
        'losses': losses,
        'draws': draws,
        'total_moves': total_moves,
        'total_time': total_time
    }


def _play_single_game_worker(player1: AlphaZeroAgent, player2: AlphaZeroAgent) -> Tuple[int, int]:
    """Play a single game between two agents (worker version)"""
    game = NineMensMorris()
    move_count = 0
    max_moves = 300  # Prevent infinite games
    
    while move_count < max_moves:
        current_player = game.get_current_player()
        
        # Select agent
        if current_player == Player.PLAYER1:
            agent = player1
        else:
            agent = player2
        
        # Get action
        try:
            action = agent.get_action(game, temperature=0.0)
        except:
            # If agent fails, random move
            valid_moves = game.get_valid_moves()
            action = random.choice(valid_moves) if valid_moves else (0,)
        
        # Make move
        success = game.make_move(action)
        if not success:
            # Invalid move, opponent wins
            winner = Player.PLAYER2 if current_player == Player.PLAYER1 else Player.PLAYER1
            result = 1 if winner == Player.PLAYER1 else -1
            return result, move_count
        
        move_count += 1
        
        # Check for game end
        game_over, winner = game.is_game_over()
        if game_over:
            if winner is None:
                return 0, move_count  # Draw
            elif winner == Player.PLAYER1:
                return 1, move_count  # Player 1 wins
            else:
                return -1, move_count  # Player 2 wins
    
    return 0, move_count  # Draw (game too long)


def quick_evaluation_demo():
    """Run a quick evaluation demo with parallel processing"""
    print("Running Quick Evaluation Demo (Parallel)")
    print("=" * 50)
    
    # Create evaluator with parallel processing
    evaluator = ModelEvaluator(max_workers=mp.cpu_count() - 1)
    
    # Create baseline agents
    evaluator.create_baseline_agents()
    
    # Add a simple trained agent (if exists)
    if os.path.exists('models/final_model_best_agent.pkl'):
        evaluator.load_agents([{
            'name': 'Trained_Model',
            'network_type': 'resnet',
            'model_path': 'models/final_model_best',
            'num_simulations': 200
        }])
    
    # Run tournament with more games due to parallel processing
    results = evaluator.round_robin_tournament(num_games_per_match=20)
    
    # Generate report
    evaluator.generate_report(results, 'quick_evaluation_report.txt')
    
    # Plot results
    evaluator.plot_tournament_results(results, 'quick_evaluation_results.png')
    
    return results


if __name__ == "__main__":
    quick_evaluation_demo()