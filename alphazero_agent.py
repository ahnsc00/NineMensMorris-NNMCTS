import numpy as np
import random
from typing import List, Tuple, Dict, Optional
from collections import deque
import pickle
import os

from nine_mens_morris import NineMensMorris, Player, GamePhase
from mcts import MCTS, MCTSNode, TrainingExample
from neural_networks import NeuralNetwork, create_network


class AlphaZeroAgent:
    """AlphaGo Zero style agent for Nine Men's Morris"""
    
    def __init__(self, network_type: str = 'resnet', num_simulations: int = 800,
                 c_puct: float = 1.0, temperature: float = 1.0, **network_kwargs):
        """
        Initialize AlphaZero agent
        
        Args:
            network_type: 'cnn', 'resnet', or 'random'
            num_simulations: Number of MCTS simulations
            c_puct: Exploration constant for UCB
            temperature: Temperature for action selection
            **network_kwargs: Additional arguments for network creation
        """
        self.network_type = network_type
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature = temperature
        
        # Create neural network
        self.network = create_network(network_type, **network_kwargs)
        
        # Create MCTS
        self.mcts = MCTS(self.network, c_puct=c_puct, num_simulations=num_simulations)
        
        # Training data buffer
        self.training_examples = deque(maxlen=10000)
        
        # Training statistics
        self.training_stats = {
            'games_played': 0,
            'total_moves': 0,
            'avg_game_length': 0.0,
            'wins_as_player1': 0,
            'wins_as_player2': 0,
            'draws': 0
        }
    
    def get_action(self, game_state: NineMensMorris, temperature: Optional[float] = None) -> Tuple[int, ...]:
        """Get action using MCTS"""
        if temperature is None:
            temperature = self.temperature
        
        return self.mcts.get_best_move(game_state, temperature)
    
    def get_action_probs(self, game_state: NineMensMorris, temperature: Optional[float] = None) -> Dict[Tuple[int, ...], float]:
        """Get action probabilities using MCTS"""
        if temperature is None:
            temperature = self.temperature
        
        return self.mcts.get_action_probs(game_state, temperature)
    
    def self_play(self, num_games: int = 1, temperature_schedule: Optional[List[float]] = None) -> List[TrainingExample]:
        """
        Play games against itself to generate training data
        
        Args:
            num_games: Number of games to play
            temperature_schedule: Temperature values for different game phases
                                If None, uses constant temperature
        
        Returns:
            List of training examples
        """
        training_examples = []
        
        for game_idx in range(num_games):
            game_examples = self._play_single_game(temperature_schedule)
            training_examples.extend(game_examples)
            
            # Update statistics
            self.training_stats['games_played'] += 1
            self.training_stats['total_moves'] += len(game_examples)
            self.training_stats['avg_game_length'] = (
                self.training_stats['total_moves'] / self.training_stats['games_played']
            )
            
            if game_idx % 10 == 0:
                print(f"Self-play game {game_idx + 1}/{num_games} completed")
        
        # Add to training buffer
        self.training_examples.extend(training_examples)
        
        return training_examples
    
    def _play_single_game(self, temperature_schedule: Optional[List[float]] = None) -> List[TrainingExample]:
        """Play a single self-play game"""
        game = NineMensMorris()
        examples = []
        move_count = 0
        
        while True:
            # Determine temperature
            if temperature_schedule:
                if move_count < len(temperature_schedule):
                    temp = temperature_schedule[move_count]
                else:
                    temp = temperature_schedule[-1]  # Use last temperature
            else:
                temp = self.temperature
            
            # Get current board state
            board_tensor = game.get_board_tensor()
            current_player = game.get_current_player()
            
            # Get action probabilities from MCTS
            action_probs = self.get_action_probs(game, temperature=temp)
            
            # Convert action probabilities to policy vector
            policy_vector = self._action_probs_to_policy_vector(action_probs, game)
            
            # Store example (value will be filled later)
            examples.append({
                'board_state': board_tensor.copy(),
                'policy_target': policy_vector,
                'current_player': current_player,
                'move_count': move_count
            })
            
            # Sample action
            if action_probs:
                moves = list(action_probs.keys())
                probs = list(action_probs.values())
                if len(moves) > 1:
                    # Use indices for np.random.choice
                    move_idx = np.random.choice(len(moves), p=probs)
                    action = moves[move_idx]
                else:
                    action = moves[0]
            else:
                # Fallback: random valid move
                valid_moves = game.get_valid_moves()
                action = random.choice(valid_moves) if valid_moves else (0,)
            
            # Make move
            success = game.make_move(action)
            if not success:
                print(f"Invalid move: {action}")
                break
            
            move_count += 1
            
            # Check for game end
            game_over, winner = game.is_game_over()
            if game_over or move_count > 200:  # Prevent infinite games
                # Assign values to all examples
                training_examples = []
                for example in examples:
                    player = example['current_player']
                    
                    if winner is None:
                        value = 0.0  # Draw
                        self.training_stats['draws'] += 1
                    elif winner == player:
                        value = 1.0  # Win
                        if player == Player.PLAYER1:
                            self.training_stats['wins_as_player1'] += 1
                        else:
                            self.training_stats['wins_as_player2'] += 1
                    else:
                        value = -1.0  # Loss
                    
                    # Create training example
                    training_example = TrainingExample(
                        board_state=example['board_state'],
                        policy_target=example['policy_target'],
                        value_target=value,
                        current_player=example['current_player']
                    )
                    training_examples.append(training_example)
                
                examples = training_examples
                
                break
        
        return examples
    
    def _action_probs_to_policy_vector(self, action_probs: Dict[Tuple[int, ...], float], 
                                      game: NineMensMorris) -> np.ndarray:
        """Convert action probabilities to policy vector"""
        # Always use fixed size policy vector (24*24 + 24 = 600)
        policy_size = 600  # 24*24 for moves + 24 for placements
        policy_vector = np.zeros(policy_size)
        
        for move, prob in action_probs.items():
            if len(move) == 1:
                # Placing phase: use first 24 indices
                idx = move[0]
            else:
                # Moving/Flying phase: use indices 24 onwards
                idx = 24 + move[0] * 24 + move[1]
            
            if idx < policy_size:
                policy_vector[idx] = prob
        
        return policy_vector
    
    def train(self, batch_size: int = 32, epochs: int = 10) -> Dict[str, float]:
        """Train the neural network on collected examples"""
        if len(self.training_examples) < batch_size:
            return {'total_loss': 0.0, 'policy_loss': 0.0, 'value_loss': 0.0}
        
        total_losses = []
        policy_losses = []
        value_losses = []
        
        for epoch in range(epochs):
            # Sample batch
            batch = random.sample(self.training_examples, batch_size)
            
            # Prepare batch data
            batch_boards = np.array([ex.board_state for ex in batch])
            batch_policies = np.array([ex.policy_target for ex in batch])
            batch_values = np.array([ex.value_target for ex in batch])
            
            # Train step
            losses = self.network.train_step(batch_boards, batch_policies, batch_values)
            
            total_losses.append(losses['total_loss'])
            policy_losses.append(losses['policy_loss'])
            value_losses.append(losses['value_loss'])
        
        return {
            'total_loss': np.mean(total_losses),
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses)
        }
    
    def evaluate_against(self, opponent: 'AlphaZeroAgent', num_games: int = 100) -> Dict[str, float]:
        """Evaluate this agent against another agent"""
        wins = 0
        losses = 0
        draws = 0
        
        for game_idx in range(num_games):
            # Alternate who goes first
            if game_idx % 2 == 0:
                result = self._play_evaluation_game(self, opponent)
                if result == 1:
                    wins += 1
                elif result == -1:
                    losses += 1
                else:
                    draws += 1
            else:
                result = self._play_evaluation_game(opponent, self)
                if result == -1:
                    wins += 1
                elif result == 1:
                    losses += 1
                else:
                    draws += 1
        
        win_rate = wins / num_games
        return {
            'wins': wins,
            'losses': losses,
            'draws': draws,
            'win_rate': win_rate
        }
    
    def _play_evaluation_game(self, player1: 'AlphaZeroAgent', player2: 'AlphaZeroAgent') -> int:
        """Play an evaluation game between two agents"""
        game = NineMensMorris()
        move_count = 0
        
        while move_count < 200:  # Prevent infinite games
            current_player = game.get_current_player()
            
            # Select agent
            if current_player == Player.PLAYER1:
                agent = player1
            else:
                agent = player2
            
            # Get action (deterministic for evaluation)
            action = agent.get_action(game, temperature=0.0)
            
            # Make move
            success = game.make_move(action)
            if not success:
                # Invalid move, opponent wins
                return -1 if current_player == Player.PLAYER1 else 1
            
            move_count += 1
            
            # Check for game end
            game_over, winner = game.is_game_over()
            if game_over:
                if winner is None:
                    return 0  # Draw
                elif winner == Player.PLAYER1:
                    return 1  # Player 1 wins
                else:
                    return -1  # Player 2 wins
        
        return 0  # Draw (game too long)
    
    def save_agent(self, filepath: str):
        """Save the agent to file"""
        # Save network
        network_path = filepath + '_network.pt'
        self.network.save_model(network_path)
        
        # Save agent data
        agent_data = {
            'network_type': self.network_type,
            'num_simulations': self.num_simulations,
            'c_puct': self.c_puct,
            'temperature': self.temperature,
            'training_stats': self.training_stats,
            'training_examples': list(self.training_examples)
        }
        
        with open(filepath + '_agent.pkl', 'wb') as f:
            pickle.dump(agent_data, f)
    
    def load_agent(self, filepath: str):
        """Load the agent from file"""
        # Load agent data
        with open(filepath + '_agent.pkl', 'rb') as f:
            agent_data = pickle.load(f)
        
        # Restore attributes
        self.network_type = agent_data['network_type']
        self.num_simulations = agent_data['num_simulations']
        self.c_puct = agent_data['c_puct']
        self.temperature = agent_data['temperature']
        self.training_stats = agent_data['training_stats']
        self.training_examples = deque(agent_data['training_examples'], maxlen=10000)
        
        # Load network
        network_path = filepath + '_network.pt'
        if os.path.exists(network_path):
            self.network.load_model(network_path)
    
    def get_stats(self) -> Dict[str, any]:
        """Get training statistics"""
        return self.training_stats.copy()
    
    def clear_training_examples(self):
        """Clear training examples buffer"""
        self.training_examples.clear()