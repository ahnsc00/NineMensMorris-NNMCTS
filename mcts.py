import numpy as np
import math
from typing import Dict, List, Optional, Tuple
from nine_mens_morris import NineMensMorris, Player

class MCTSNode:
    def __init__(self, game_state: NineMensMorris, parent: Optional['MCTSNode'] = None, 
                 move: Optional[Tuple[int, ...]] = None, prior: float = 0.0):
        self.game_state = game_state.clone()
        self.parent = parent
        self.move = move
        self.prior = prior
        
        # MCTS statistics
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: Dict[Tuple[int, ...], MCTSNode] = {}
        
        # Neural network predictions
        self.policy_probs: Optional[np.ndarray] = None
        self.value_prediction: Optional[float] = None
        
        # Game state info
        self.is_expanded = False
        self.is_terminal = False
        self.terminal_value: Optional[float] = None
        
        self._check_terminal()
    
    def _check_terminal(self):
        """Check if this node represents a terminal game state"""
        game_over, winner = self.game_state.is_game_over()
        if game_over:
            self.is_terminal = True
            if winner is None:
                self.terminal_value = 0.0  # Draw
            elif winner == self.game_state.get_current_player():
                self.terminal_value = 1.0  # Current player wins
            else:
                self.terminal_value = -1.0  # Current player loses
    
    def is_leaf(self) -> bool:
        """Check if this is a leaf node (not expanded)"""
        return not self.is_expanded
    
    def get_value(self) -> float:
        """Get the average value of this node"""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def get_ucb_score(self, c_puct: float = 1.0) -> float:
        """Calculate UCB score for node selection"""
        if self.visit_count == 0:
            return float('inf')
        
        # UCB = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        q_value = self.get_value()
        exploration = c_puct * self.prior * math.sqrt(self.parent.visit_count) / (1 + self.visit_count)
        return q_value + exploration
    
    def expand(self, policy_probs: np.ndarray, value_prediction: float):
        """Expand this node using neural network predictions"""
        if self.is_terminal:
            return
        
        self.policy_probs = policy_probs.copy()
        self.value_prediction = value_prediction
        self.is_expanded = True
        
        # Get valid moves
        valid_moves = self.game_state.get_valid_moves()
        
        # Create child nodes for valid moves
        for move in valid_moves:
            # Create new game state
            new_game = self.game_state.clone()
            move_success = new_game.make_move(move)
            
            if move_success:
                # Get prior probability for this move
                move_idx = self._move_to_index(move)
                prior = policy_probs[move_idx] if move_idx < len(policy_probs) else 0.001
                
                # Create child node
                child = MCTSNode(new_game, parent=self, move=move, prior=prior)
                self.children[move] = child
    
    def _move_to_index(self, move: Tuple[int, ...]) -> int:
        """Convert move to policy network index"""
        if len(move) == 1:
            # Placing phase: just the position
            return move[0]
        elif len(move) == 2:
            # Moving/Flying phase: from_pos * 24 + to_pos
            return move[0] * 24 + move[1]
        return 0
    
    def select_child(self, c_puct: float = 1.0) -> Optional['MCTSNode']:
        """Select best child using UCB"""
        if not self.children:
            return None
        
        best_score = float('-inf')
        best_child = None
        
        for child in self.children.values():
            score = child.get_ucb_score(c_puct)
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child
    
    def backup(self, value: float):
        """Backup value through the tree"""
        self.visit_count += 1
        self.value_sum += value
        
        if self.parent is not None:
            # Flip value for opponent
            self.parent.backup(-value)
    
    def get_action_probs(self, temperature: float = 1.0) -> Dict[Tuple[int, ...], float]:
        """Get action probabilities based on visit counts"""
        if not self.children:
            return {}
        
        if temperature == 0:
            # Deterministic: choose most visited
            best_move = max(self.children.keys(), key=lambda m: self.children[m].visit_count)
            return {move: 1.0 if move == best_move else 0.0 for move in self.children.keys()}
        
        # Temperature scaling
        visit_counts = np.array([self.children[move].visit_count for move in self.children.keys()])
        
        if temperature == float('inf'):
            # Uniform distribution
            probs = np.ones_like(visit_counts) / len(visit_counts)
        else:
            # Apply temperature
            log_probs = np.log(visit_counts + 1e-10) / temperature
            log_probs = log_probs - np.max(log_probs)  # Numerical stability
            probs = np.exp(log_probs)
            probs = probs / np.sum(probs)
        
        return {move: prob for move, prob in zip(self.children.keys(), probs)}


class MCTS:
    def __init__(self, neural_network, c_puct: float = 1.0, num_simulations: int = 800):
        self.neural_network = neural_network
        self.c_puct = c_puct
        self.num_simulations = num_simulations
    
    def search(self, root_state: NineMensMorris) -> MCTSNode:
        """Perform MCTS search and return the root node"""
        root = MCTSNode(root_state)
        
        # Always expand root if not terminal
        if not root.is_terminal:
            board_tensor = root_state.get_board_tensor()
            policy, value = self.neural_network.predict(board_tensor)
            root.expand(policy, value)
            
            # If no children after expansion, return root with fallback
            if not root.children:
                # Create fallback action probabilities based on valid moves
                valid_moves = root_state.get_valid_moves()
                if valid_moves:
                    uniform_prob = 1.0 / len(valid_moves)
                    for move in valid_moves:
                        # Create a fake child node for action probability calculation
                        fake_child = MCTSNode(root_state.clone(), parent=root, move=move)
                        fake_child.visit_count = 1  # Give it a visit count
                        root.children[move] = fake_child
        
        # Perform simulations
        for _ in range(self.num_simulations):
            self._simulate(root)
        
        return root
    
    def _simulate(self, root: MCTSNode):
        """Perform one MCTS simulation"""
        path = []
        current = root
        
        # Selection: traverse down the tree
        while not current.is_leaf() and not current.is_terminal:
            next_child = current.select_child(self.c_puct)
            if next_child is None:
                break
            current = next_child
            path.append(current)
        
        # Expansion and Evaluation
        if current.is_terminal:
            value = current.terminal_value
        else:
            # Neural network evaluation
            board_tensor = current.game_state.get_board_tensor()
            policy, value = self.neural_network.predict(board_tensor)
            current.expand(policy, value)
        
        # Backup
        current.backup(value)
    
    def get_action_probs(self, game_state: NineMensMorris, temperature: float = 1.0) -> Dict[Tuple[int, ...], float]:
        """Get action probabilities after MCTS search"""
        root = self.search(game_state)
        return root.get_action_probs(temperature)
    
    def get_best_move(self, game_state: NineMensMorris, temperature: float = 0.0) -> Tuple[int, ...]:
        """Get the best move after MCTS search"""
        action_probs = self.get_action_probs(game_state, temperature)
        if not action_probs:
            # Fallback: return first valid move
            valid_moves = game_state.get_valid_moves()
            return valid_moves[0] if valid_moves else (0,)
        
        # Sample from action probabilities
        moves = list(action_probs.keys())
        probs = list(action_probs.values())
        
        if temperature == 0.0:
            # Deterministic: choose highest probability
            best_idx = np.argmax(probs)
            return moves[best_idx]
        else:
            # Stochastic: sample from distribution
            move_idx = np.random.choice(len(moves), p=probs)
            return moves[move_idx]


class TrainingExample:
    """Training example for neural network"""
    def __init__(self, board_state: np.ndarray, policy_target: np.ndarray, 
                 value_target: float, current_player: Player):
        self.board_state = board_state
        self.policy_target = policy_target
        self.value_target = value_target
        self.current_player = current_player