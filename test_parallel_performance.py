#!/usr/bin/env python3
"""
Performance test script for parallel training and evaluation
"""

import time
import multiprocessing as mp
from training_loop import TrainingManager, get_default_config
from evaluation import quick_evaluation_demo
import torch
import psutil


def test_system_resources():
    """Test and display system resources"""
    print("🖥️  SYSTEM RESOURCES")
    print("=" * 50)
    
    # CPU Information
    cpu_count = mp.cpu_count()
    cpu_percent = psutil.cpu_percent(interval=1)
    print(f"CPU Cores: {cpu_count}")
    print(f"CPU Usage: {cpu_percent:.1f}%")
    
    # Memory Information
    memory = psutil.virtual_memory()
    print(f"Total RAM: {memory.total / (1024**3):.1f} GB")
    print(f"Available RAM: {memory.available / (1024**3):.1f} GB")
    print(f"Memory Usage: {memory.percent:.1f}%")
    
    # GPU Information
    if torch.cuda.is_available():
        print(f"GPU Available: Yes")
        print(f"GPU Name: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
        
        # GPU capability for mixed precision
        capability = torch.cuda.get_device_capability()
        mixed_precision_supported = capability[0] >= 7
        print(f"Mixed Precision Support: {'Yes' if mixed_precision_supported else 'No'} (Compute {capability[0]}.{capability[1]})")
    else:
        print("GPU Available: No")
    
    print()


def test_parallel_training():
    """Test parallel training performance"""
    print("🚀 PARALLEL TRAINING TEST")
    print("=" * 50)
    
    # Test both CNN and ResNet with parallel processing
    for network_type in ['cnn', 'resnet']:
        print(f"\n🔧 Testing {network_type.upper()} with parallel processing...")
        
        # Get optimized config
        config = get_default_config(network_type)
        print(f"Configuration:")
        print(f"  - Max workers: {config['max_workers']}")
        print(f"  - Self-play games: {config['self_play_games']}")
        print(f"  - Evaluation games: {config['evaluation_games']}")
        print(f"  - Batch size: {config['batch_size']}")
        print(f"  - MCTS simulations: {config['num_simulations']}")
        print(f"  - GPU optimization: {config['enable_gpu_optimization']}")
        print(f"  - Mixed precision: {config['mixed_precision']}")
        
        # Create training manager
        trainer = TrainingManager(config)
        
        # Test one iteration
        print(f"\n⏱️  Running 1 training iteration...")
        start_time = time.time()
        
        try:
            # Self-play phase
            print("Phase 1: Self-play...")
            self_play_start = time.time()
            examples_count = trainer.run_self_play()
            self_play_time = time.time() - self_play_start
            
            # Training phase
            print("Phase 2: Neural network training...")
            training_start = time.time()
            losses = trainer.train_network()
            training_time = time.time() - training_start
            
            # Evaluation phase
            print("Phase 3: Agent evaluation...")
            eval_start = time.time()
            eval_results = trainer.evaluate_agent()
            eval_time = time.time() - eval_start
            
            total_time = time.time() - start_time
            
            # Results
            print(f"\n📊 Results for {network_type.upper()}:")
            print(f"  - Total time: {total_time:.2f}s")
            print(f"  - Self-play time: {self_play_time:.2f}s ({examples_count} examples)")
            print(f"  - Training time: {training_time:.2f}s")
            print(f"  - Evaluation time: {eval_time:.2f}s")
            print(f"  - Training loss: {losses['total_loss']:.4f}")
            print(f"  - Win rate: {eval_results['win_rate']:.3f}")
            
            # Performance metrics
            examples_per_sec = examples_count / self_play_time if self_play_time > 0 else 0
            games_per_sec = config['evaluation_games'] / eval_time if eval_time > 0 else 0
            
            print(f"  - Self-play rate: {examples_per_sec:.1f} examples/sec")
            print(f"  - Evaluation rate: {games_per_sec:.1f} games/sec")
            
        except Exception as e:
            print(f"❌ Error during {network_type} test: {e}")
        
        print("-" * 50)


def test_parallel_evaluation():
    """Test parallel evaluation performance"""
    print("🎯 PARALLEL EVALUATION TEST")
    print("=" * 50)
    
    print("Running tournament evaluation with parallel processing...")
    
    start_time = time.time()
    try:
        results = quick_evaluation_demo()
        total_time = time.time() - start_time
        
        print(f"\n📊 Evaluation Results:")
        print(f"  - Total time: {total_time:.2f}s")
        print(f"  - Tournament info: {results['tournament_info']}")
        
        # Show top performers
        overall_stats = results['overall_stats']
        sorted_agents = sorted(overall_stats.items(), 
                             key=lambda x: x[1]['overall_win_rate'], 
                             reverse=True)
        
        print(f"\n🏆 Top Performers:")
        for rank, (agent_name, stats) in enumerate(sorted_agents[:3], 1):
            print(f"  {rank}. {agent_name}: {stats['overall_win_rate']:.3f} win rate")
        
    except Exception as e:
        print(f"❌ Error during evaluation test: {e}")


def test_memory_optimization():
    """Test memory optimization features"""
    print("🧠 MEMORY OPTIMIZATION TEST")
    print("=" * 50)
    
    # Memory before
    memory_before = psutil.virtual_memory()
    gpu_memory_before = 0
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gpu_memory_before = torch.cuda.memory_allocated()
    
    print(f"Memory before: {memory_before.used / (1024**3):.2f} GB")
    if torch.cuda.is_available():
        print(f"GPU memory before: {gpu_memory_before / (1024**3):.2f} GB")
    
    # Create a large batch for testing
    print("\nCreating neural network and testing memory efficiency...")
    
    try:
        from neural_networks import create_network
        
        # Test CNN
        cnn_net = create_network('cnn', learning_rate=0.001)
        print(f"CNN network created successfully")
        
        # Test ResNet  
        resnet_net = create_network('resnet', learning_rate=0.001)
        print(f"ResNet network created successfully")
        
        # Memory after
        memory_after = psutil.virtual_memory()
        gpu_memory_after = 0
        if torch.cuda.is_available():
            gpu_memory_after = torch.cuda.memory_allocated()
        
        print(f"\nMemory after: {memory_after.used / (1024**3):.2f} GB")
        memory_diff = (memory_after.used - memory_before.used) / (1024**3)
        print(f"Memory difference: {memory_diff:.2f} GB")
        
        if torch.cuda.is_available():
            print(f"GPU memory after: {gpu_memory_after / (1024**3):.2f} GB")
            gpu_memory_diff = (gpu_memory_after - gpu_memory_before) / (1024**3)
            print(f"GPU memory difference: {gpu_memory_diff:.2f} GB")
        
        # Cleanup
        del cnn_net, resnet_net
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"❌ Error during memory test: {e}")


def main():
    """Run all performance tests"""
    print("🔥 NEURALMILL-MCTS PARALLEL PERFORMANCE TEST")
    print("=" * 70)
    print(f"Testing CPU & GPU utilization for maximum performance")
    print(f"Multiprocessing workers: {mp.cpu_count() - 1}")
    print("=" * 70)
    print()
    
    # Test system resources
    test_system_resources()
    
    # Test memory optimization
    test_memory_optimization()
    print()
    
    # Test parallel training
    test_parallel_training()
    print()
    
    # Test parallel evaluation  
    test_parallel_evaluation()
    print()
    
    print("✅ All performance tests completed!")
    print("\n💡 Performance Tips:")
    print("   - Use GPU if available for faster training")
    print("   - Increase batch size on high-memory GPUs")
    print("   - Use mixed precision on modern GPUs (Compute 7.0+)")
    print("   - Scale self-play games with CPU cores")
    print("   - Monitor resource usage during training")


if __name__ == "__main__":
    main()