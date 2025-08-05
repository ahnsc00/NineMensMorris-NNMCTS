# NeuralMill-MCTS 🎯

> *AlphaZero-style MCTS agent for Nine Men's Morris with parallel processing*

Advanced AI system combining deep reinforcement learning with Monte Carlo Tree Search. Features CNN and ResNet architectures with multi-core CPU and GPU optimization.

## ✨ Key Features

- **Complete Nine Men's Morris** with 3-phase gameplay (Placing → Moving → Flying)
- **AlphaZero Architecture**: MCTS + CNN/ResNet neural networks
- **🚀 Parallel Processing**: Multi-core self-play and GPU optimization
- **Mixed Precision Training**: Automatic FP16 on modern GPUs
- **Auto-Scaling**: Dynamic resource allocation based on system specs
- **Tournament Evaluation**: Parallel model comparison and performance tracking

## ⚡ Quick Start

```bash
# Install and run
git clone https://github.com/ahnsc00/NeuralMill-MCTS.git
cd NeuralMill-MCTS
pip install -r requirements.txt

# Quick demo
python main.py demo

# Train AI (auto-optimized for your system)
python main.py train --network resnet --iterations 50

# Test performance improvements
python main.py test-performance

# Play against AI
python main.py play
```

## ⚙️ Configuration

| Parameter | Default | Auto-Scaled | Description |
|-----------|---------|-------------|-------------|
| `--iterations` | 50 | - | Training cycles |
| `--simulations` | 200/400 | GPU-based | MCTS depth |
| `--games` | 25+ | CPU cores × 4 | Self-play games |
| `--batch-size` | 32/64 | GPU memory | Training batch |

```bash
# Examples
python main.py train --iterations 100 --simulations 800  # High performance
python main.py train --iterations 25 --games 10          # Fast testing
```

## 📊 Performance

| Architecture | Training Time (50 iter) | Memory | Notes |
|--------------|-------------------------|--------|-------|
| **CNN** | 30min (GPU) / 2-3h (CPU) | ~200MB | Fast, efficient |
| **ResNet** | 1h (GPU) / 4-5h (CPU) | ~400MB | Better performance |

**Parallel Processing Speedup**: 2-4x faster with multi-core CPUs

## 🔧 Advanced Features

- **Temperature Scheduling**: Exploration → Exploitation transition
- **Early Stopping**: Auto-termination on performance plateau  
- **Mixed Precision**: FP16 training on RTX GPUs
- **Resource Monitoring**: Real-time CPU/GPU/memory tracking
- **Auto-Checkpointing**: Periodic model saving

## 🛠️ Troubleshooting


# Reduce memory usage
python main.py train --batch-size 16 --games 10

# Speed up training  
python main.py train --simulations 200
```

## 📚 References

- [AlphaGo Zero Paper](https://www.nature.com/articles/nature24270) - Core algorithm
- [Nine Men's Morris Rules](https://en.wikipedia.org/wiki/Nine_men%27s_morris) - Game mechanics
- [MCTS Algorithm](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search) - Search methodology

---
**MIT License** • Built with PyTorch and AlphaZero principles
