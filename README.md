# NeuralMill-MCTS 🎯

> *AlphaZero-style Monte Carlo Tree Search agent for Nine Men's Morris*

An advanced AI system that combines deep reinforcement learning with Monte Carlo Tree Search to master the ancient strategy game of Nine Men's Morris. Built with PyTorch and featuring both CNN and ResNet architectures for optimal performance.

## 🏗️ Project Structure

```
NeuralMill-MCTS/
├── 🎮 Game Implementation
│   ├── nine_mens_morris.py      # Core game logic and rules
│   ├── game_renderer.py         # PyGame-based visual interface
│   └── play_game.py            # Human vs Human gameplay
├── 🧠 AI Engine
│   ├── mcts.py                  # Monte Carlo Tree Search implementation
│   ├── neural_networks.py       # CNN & ResNet architectures
│   ├── alphazero_agent.py       # Unified AlphaZero agent
│   └── training_loop.py         # Self-play training orchestration
├── 🔧 Tools & Utilities
│   ├── main.py                  # Command-line interface
│   ├── evaluation.py            # Model performance assessment
│   ├── test_game.py            # Game logic validation
│   └── simple_test.py          # Quick functionality tests
├── 📊 Training Assets
│   ├── models/                  # Trained neural networks
│   ├── logs/                   # Training metrics & visualizations
│   └── requirements.txt        # Dependencies
```

## ✨ Key Features

### 🎯 **Game Implementation**
- **Complete Nine Men's Morris rules** with all traditional gameplay mechanics
- **Three-phase gameplay**: Placing → Moving → Flying transitions
- **Mill formation detection** and opponent piece removal logic
- **Win/draw condition validation** with comprehensive endgame scenarios

### 🚀 **AlphaZero AI Architecture**
- **Monte Carlo Tree Search** with UCB1 exploration policy
- **Dual Neural Network Architectures**:
  - 🏃‍♂️ **CNN**: Fast training, memory-efficient (200MB)
  - 🏋️‍♂️ **ResNet**: Deep residual blocks, superior performance (400MB)
- **Self-Play Learning**: Continuous improvement through self-competition
- **Policy + Value Networks**: Simultaneous move probability and position evaluation

### 📈 **Training & Evaluation System**
- **Automated training pipeline** with configurable hyperparameters
- **Real-time performance tracking** with matplotlib visualizations
- **Tournament-style model comparison** for objective performance assessment
- **Comprehensive logging** with JSON training histories and PNG curves

## ⚡ Quick Start

### 📦 Installation

```bash
# Clone the repository
git clone https://github.com/ahnsc00/NeuralMill-MCTS.git
cd NeuralMill-MCTS

# Install dependencies
pip install -r requirements.txt

# For GPU acceleration (optional, requires CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 🎮 Usage Examples

#### **🔥 Quick Demo**
```bash
python main.py demo
```

#### **🧠 Train Your AI**
 ```bash
# High-performance ResNet training
python main.py train --network resnet --iterations 100

# Fast CNN prototyping
python main.py train --network cnn --iterations 50 --simulations 200
```

#### **📊 Evaluate Performance**
```bash
# Tournament-style model comparison
python main.py evaluate --games 50

# Evaluate specific model directory
python main.py evaluate --model-dir models --games 100
```

#### **🎯 Play Against AI**
```bash
# Challenge the best trained model
python main.py play

# Play against specific model
python main.py play --model-path models/final_model_best
```

#### **👥 Human vs Human**
```bash
python main.py human
```

## ⚙️ Configuration & Hyperparameters

### 🎛️ **Default Settings**
| Parameter | Value | Description |
|-----------|-------|-------------|
| **MCTS Simulations** | 400 | Tree search depth per move |
| **Self-play Games** | 25 | Games per training iteration |
| **Batch Size** | 32 | Neural network training batch |
| **Learning Rate** | 0.001 | Adam optimizer step size |

### 🚀 **Advanced Configurations**

#### **High-Performance Setup** (GPU Recommended)
```bash
python main.py train --network resnet --iterations 200 \
                     --simulations 800 --games 50 --batch-size 64
```

#### **Fast Prototyping Setup** (CPU Friendly)
```bash
python main.py train --network cnn --iterations 25 \
                     --simulations 100 --games 10 --batch-size 16
```

## 🏗️ Neural Network Architectures

### 🏃‍♂️ **CNN Architecture**
```
Input (3×7×7) → Conv2D(32) → BatchNorm → ReLU
               → Conv2D(64) → BatchNorm → ReLU  
               → Conv2D(128) → BatchNorm → ReLU
               → Conv2D(256) → BatchNorm → ReLU
               ├── Policy Head → FC → Softmax
               └── Value Head → FC → Tanh
```

### 🏋️‍♂️ **ResNet Architecture**
```
Input (3×7×7) → Initial Conv2D(128) → BatchNorm → ReLU
               → 6× Residual Blocks [Conv2D→BN→ReLU→Conv2D→BN + Skip]
               ├── Policy Head → Global Pool → FC → Softmax
               └── Value Head → Global Pool → FC → Tanh
```

## 📊 Performance Benchmarks

### ⏱️ **Training Time** (50 iterations)
| Architecture | CPU | GPU |
|--------------|-----|-----|
| **CNN** | 2-3 hours | ~30 minutes |
| **ResNet** | 4-5 hours | ~1 hour |

### 💾 **Memory Usage**
| Architecture | RAM | VRAM |
|--------------|-----|------|
| **CNN** | ~200MB | ~150MB |
| **ResNet** | ~400MB | ~300MB |

## 🎛️ Advanced Features

### 🌡️ **Temperature Scheduling**
Exploration-to-exploitation transition during training:
```python
temperature_schedule = [1.0] * 10 + [0.5] * 10 + [0.1]  # Hot → Warm → Cool
```

### ⏹️ **Early Stopping**
Automatic termination when performance plateaus:
```python
max_iterations_without_improvement = 30
```

### 💾 **Checkpoint Management**
Periodic model saving for recovery:
```python
checkpoint_freq = 10  # Save every 10 iterations
```

## 🔍 Technical Deep Dive

### 🧩 **Core Classes**
| Class | Purpose | Key Methods |
|-------|---------|-------------|
| `NineMensMorris` | Game engine | `get_valid_moves()`, `make_move()` |
| `MCTSNode` | Search tree node | `select()`, `expand()`, `backup()` |
| `MCTS` | Tree search algorithm | `search()`, `get_action_probs()` |
| `AlphaZeroAgent` | Unified AI system | `self_play()`, `train()` |
| `CNNNetwork`/`ResNetNetwork` | Neural models | `forward()`, `predict()` |

### 📊 **Data Representations**
- **Board State**: `(3, 7, 7)` tensor → Player1/Player2/Empty channels
- **Policy Vector**: Action probability distribution over valid moves
- **Value Scalar**: Position evaluation ∈ [-1, +1] (Loss ↔ Win)

## 🛠️ Troubleshooting

### 🚫 **CUDA Issues**
```bash
# Force CPU usage
export CUDA_VISIBLE_DEVICES=""
```

### 💾 **Memory Constraints**
```bash
# Reduce memory footprint
python main.py train --batch-size 16 --games 10
```

### 🐌 **Slow Training**
```bash
# Decrease simulation count
python main.py train --simulations 200
```

## 🔮 Extensibility & Future Work

### 🧪 **Adding New Architectures**
Extend `neural_networks.py` with custom models:
```python
class TransformerNetwork(nn.Module, NeuralNetwork):
    def __init__(self, input_dim=147):
        super().__init__()
        self.transformer = nn.TransformerEncoder(...)
        # Implementation details
```

### 🎲 **Porting to Other Games**
Adapt the framework by implementing the game interface:
```python
class YourGame:
    def get_valid_moves(self) -> List[int]
    def make_move(self, action: int) -> 'YourGame'
    def is_game_over(self) -> bool
    def get_board_tensor(self) -> torch.Tensor
```

### 🚀 **Performance Optimizations**
- **GPU Parallelization**: Multi-GPU training support
- **Mixed Precision**: FP16 training for faster convergence
- **Distributed Training**: Cluster-based self-play generation

## 📄 License

MIT License - feel free to use, modify, and distribute!

## 🤝 Contributing

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-improvement`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-improvement`)
5. **Open** a Pull Request

## 📚 References & Inspiration

- 📖 [**AlphaGo Zero Paper**](https://www.nature.com/articles/nature24270) - Mastering the game of Go without human knowledge
- 🎯 [**Nine Men's Morris Rules**](https://en.wikipedia.org/wiki/Nine_men%27s_morris) - Traditional game mechanics
- 🌳 [**MCTS Algorithm**](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search) - Tree search methodology
- 🧠 [**Deep Reinforcement Learning**](https://arxiv.org/abs/1312.5602) - Neural network game playing

---

<div align="center">

**⭐ Star this repository if you found it helpful!**

Built with ❤️ using PyTorch and AlphaZero principles

</div>