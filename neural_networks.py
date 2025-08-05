import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
from abc import ABC, abstractmethod

class NeuralNetwork(ABC):
    """Abstract base class for neural networks"""
    
    @abstractmethod
    def predict(self, board_state: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Predict policy and value for given board state
        Args:
            board_state: Shape (3, 7, 7) - board representation
        Returns:
            policy: Action probabilities
            value: Position value estimation
        """
        pass
    
    @abstractmethod
    def train_step(self, batch_boards: np.ndarray, batch_policies: np.ndarray, 
                   batch_values: np.ndarray) -> dict:
        """
        Perform one training step
        Returns:
            Dictionary with loss information
        """
        pass
    
    @abstractmethod
    def save_model(self, filepath: str):
        """Save model to file"""
        pass
    
    @abstractmethod
    def load_model(self, filepath: str):
        """Load model from file"""
        pass


class CNNNetwork(nn.Module, NeuralNetwork):
    """CNN-based neural network for Nine Men's Morris"""
    
    def __init__(self, input_channels: int = 3, hidden_size: int = 256, 
                 num_actions: int = 600, learning_rate: float = 0.001):
        super(CNNNetwork, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_size = hidden_size
        self.num_actions = num_actions
        
        # CNN layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Calculate flattened size
        self.flattened_size = 256 * 7 * 7  # 256 channels * 7x7 board
        
        # Fully connected layers
        self.fc_common = nn.Linear(self.flattened_size, hidden_size)
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_actions),
            nn.Softmax(dim=-1)
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Tanh()
        )
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass"""
        # CNN layers with batch norm and ReLU
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Common fully connected layer
        x = F.relu(self.fc_common(x))
        
        # Policy and value heads
        policy = self.policy_head(x)
        value = self.value_head(x)
        
        return policy, value.squeeze(-1)
    
    def predict(self, board_state: np.ndarray) -> Tuple[np.ndarray, float]:
        """Predict policy and value for given board state"""
        self.eval()
        with torch.no_grad():
            # Convert to tensor and add batch dimension
            if board_state.ndim == 3:
                board_state = board_state[np.newaxis, ...]
            
            x = torch.FloatTensor(board_state).to(self.device)
            policy, value = self.forward(x)
            
            # Convert back to numpy
            policy = policy.cpu().numpy()[0]
            value = value.cpu().item()
            
        return policy, value
    
    def train_step(self, batch_boards: np.ndarray, batch_policies: np.ndarray, 
                   batch_values: np.ndarray) -> dict:
        """Perform one training step"""
        self.train()
        
        # Convert to tensors
        boards = torch.FloatTensor(batch_boards).to(self.device)
        target_policies = torch.FloatTensor(batch_policies).to(self.device)
        target_values = torch.FloatTensor(batch_values).to(self.device)
        
        # Forward pass
        pred_policies, pred_values = self.forward(boards)
        
        # Calculate losses
        # Use KL divergence for policy loss instead of cross-entropy
        policy_loss = F.kl_div(F.log_softmax(pred_policies, dim=1), target_policies, reduction='batchmean')
        value_loss = F.mse_loss(pred_values, target_values)
        total_loss = policy_loss + value_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item()
        }
    
    def save_model(self, filepath: str):
        """Save model to file"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'input_channels': self.input_channels,
            'hidden_size': self.hidden_size,
            'num_actions': self.num_actions
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load model from file"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


class ResidualBlock(nn.Module):
    """Residual block for ResNet"""
    
    def __init__(self, channels: int):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += residual
        out = F.relu(out)
        
        return out


class ResNetNetwork(nn.Module, NeuralNetwork):
    """ResNet-based neural network for Nine Men's Morris"""
    
    def __init__(self, input_channels: int = 3, num_blocks: int = 8, 
                 channels: int = 256, num_actions: int = 600, learning_rate: float = 0.001):
        super(ResNetNetwork, self).__init__()
        
        self.input_channels = input_channels
        self.num_blocks = num_blocks
        self.channels = channels
        self.num_actions = num_actions
        
        # Initial convolution
        self.conv_initial = nn.Conv2d(input_channels, channels, kernel_size=3, padding=1)
        self.bn_initial = nn.BatchNorm2d(channels)
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(channels) for _ in range(num_blocks)
        ])
        
        # Calculate flattened size
        self.flattened_size = channels * 7 * 7
        
        # Policy head
        self.policy_conv = nn.Conv2d(channels, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 7 * 7, num_actions)
        
        # Value head
        self.value_conv = nn.Conv2d(channels, 32, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 7 * 7, 256)
        self.value_fc2 = nn.Linear(256, 1)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass"""
        # Initial convolution
        x = F.relu(self.bn_initial(self.conv_initial(x)))
        
        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        policy = F.softmax(self.policy_fc(policy), dim=-1)
        
        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value.squeeze(-1)
    
    def predict(self, board_state: np.ndarray) -> Tuple[np.ndarray, float]:
        """Predict policy and value for given board state"""
        self.eval()
        with torch.no_grad():
            # Convert to tensor and add batch dimension
            if board_state.ndim == 3:
                board_state = board_state[np.newaxis, ...]
            
            x = torch.FloatTensor(board_state).to(self.device)
            policy, value = self.forward(x)
            
            # Convert back to numpy
            policy = policy.cpu().numpy()[0]
            value = value.cpu().item()
            
        return policy, value
    
    def train_step(self, batch_boards: np.ndarray, batch_policies: np.ndarray, 
                   batch_values: np.ndarray) -> dict:
        """Perform one training step"""
        self.train()
        
        # Convert to tensors
        boards = torch.FloatTensor(batch_boards).to(self.device)
        target_policies = torch.FloatTensor(batch_policies).to(self.device)
        target_values = torch.FloatTensor(batch_values).to(self.device)
        
        # Forward pass
        pred_policies, pred_values = self.forward(boards)
        
        # Calculate losses
        # Use KL divergence for policy loss instead of cross-entropy
        policy_loss = F.kl_div(F.log_softmax(pred_policies, dim=1), target_policies, reduction='batchmean')
        value_loss = F.mse_loss(pred_values, target_values)
        total_loss = policy_loss + value_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item()
        }
    
    def save_model(self, filepath: str):
        """Save model to file"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'input_channels': self.input_channels,
            'num_blocks': self.num_blocks,
            'channels': self.channels,
            'num_actions': self.num_actions
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load model from file"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


class RandomNetwork(NeuralNetwork):
    """Random baseline network for comparison"""
    
    def __init__(self, num_actions: int = 600):
        self.num_actions = num_actions
    
    def predict(self, board_state: np.ndarray) -> Tuple[np.ndarray, float]:
        """Return random policy and value"""
        policy = np.random.random(self.num_actions)
        policy = policy / np.sum(policy)  # Normalize
        value = np.random.random() * 2 - 1  # Random value between -1 and 1
        return policy, value
    
    def train_step(self, batch_boards: np.ndarray, batch_policies: np.ndarray, 
                   batch_values: np.ndarray) -> dict:
        """No-op training for random network"""
        return {'total_loss': 0.0, 'policy_loss': 0.0, 'value_loss': 0.0}
    
    def save_model(self, filepath: str):
        """No-op save for random network"""
        pass
    
    def load_model(self, filepath: str):
        """No-op load for random network"""
        pass


def create_network(network_type: str, **kwargs) -> NeuralNetwork:
    """Factory function to create neural networks"""
    if network_type.lower() == 'cnn':
        return CNNNetwork(**kwargs)
    elif network_type.lower() == 'resnet':
        return ResNetNetwork(**kwargs)
    elif network_type.lower() == 'random':
        return RandomNetwork(**kwargs)
    else:
        raise ValueError(f"Unknown network type: {network_type}")