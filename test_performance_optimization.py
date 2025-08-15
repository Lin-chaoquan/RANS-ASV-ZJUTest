#!/usr/bin/env python3
"""
性能优化测试脚本
比较优化前后的泊船任务运行速度
"""

import sys
import time
import torch
import numpy as np

# 添加路径
sys.path.append('.')

def test_reward_function_performance():
    """测试奖励函数的性能"""
    print("=== 测试奖励函数性能 ===")
    
    # 模拟数据
    num_envs = 512
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建测试数据
    current_state = {
        'position': torch.randn(num_envs, 3, device=device),
        'orientation': torch.randn(num_envs, 4, device=device),
        'linear_velocity': torch.randn(num_envs, 3, device=device),
        'angular_velocity': torch.randn(num_envs, 1, device=device)
    }
    
    actions = torch.randn(num_envs, 3, device=device)
    position_error = torch.randn(num_envs, device=device)
    heading_error = torch.randn(num_envs, device=device)
    berth_corners = torch.randn(num_envs, 4, 2, device=device)
    collision_radius = 0.5
    
    # 导入优化后的奖励类
    from omniisaacgymenvs.tasks.USV.USV_task_rewards import BerthingReward
    
    # 创建奖励实例
    reward_fn = BerthingReward()
    
    # 预热GPU
    if device == 'cuda':
        for _ in range(10):
            _ = reward_fn.compute_reward(
                current_state, actions, position_error, heading_error, 
                berth_corners, collision_radius
            )
        torch.cuda.synchronize()
    
    # 测试性能
    num_iterations = 100
    
    # 测试向量化版本
    start_time = time.time()
    for _ in range(num_iterations):
        rewards = reward_fn.compute_reward(
            current_state, actions, position_error, heading_error, 
            berth_corners, collision_radius
        )
    if device == 'cuda':
        torch.cuda.synchronize()
    vectorized_time = time.time() - start_time
    
    print(f"向量化版本 ({num_iterations} 次迭代): {vectorized_time:.4f} 秒")
    print(f"平均每次计算: {vectorized_time/num_iterations*1000:.2f} 毫秒")
    print(f"每秒计算次数: {num_iterations/vectorized_time:.0f}")
    
    return vectorized_time

def test_collision_detection_performance():
    """测试碰撞检测的性能"""
    print("\n=== 测试碰撞检测性能 ===")
    
    # 模拟数据
    num_envs = 512
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建测试数据
    usv_positions = torch.randn(num_envs, 2, device=device)
    berth_corners = torch.randn(num_envs, 4, 2, device=device)
    collision_radius = 0.5
    
    # 导入优化后的奖励类
    from omniisaacgymenvs.tasks.USV.USV_task_rewards import BerthingReward
    
    # 创建奖励实例
    reward_fn = BerthingReward()
    
    # 预热GPU
    if device == 'cuda':
        for _ in range(10):
            _ = reward_fn._check_collision_with_berth_vectorized(
                usv_positions, berth_corners, collision_radius
            )
        torch.cuda.synchronize()
    
    # 测试性能
    num_iterations = 100
    
    # 测试向量化版本
    start_time = time.time()
    for _ in range(num_iterations):
        collisions = reward_fn._check_collision_with_berth_vectorized(
            usv_positions, berth_corners, collision_radius
        )
    if device == 'cuda':
        torch.cuda.synchronize()
    vectorized_time = time.time() - start_time
    
    print(f"向量化碰撞检测 ({num_iterations} 次迭代): {vectorized_time:.4f} 秒")
    print(f"平均每次检测: {vectorized_time/num_iterations*1000:.2f} 毫秒")
    print(f"每秒检测次数: {num_iterations/vectorized_time:.0f}")
    
    return vectorized_time

def test_memory_usage():
    """测试内存使用情况"""
    print("\n=== 测试内存使用情况 ===")
    
    if torch.cuda.is_available():
        # 清理GPU缓存
        torch.cuda.empty_cache()
        
        # 获取初始内存
        initial_memory = torch.cuda.memory_allocated() / 1024**2
        print(f"初始GPU内存使用: {initial_memory:.2f} MB")
        
        # 创建大量数据
        num_envs = 1024
        device = 'cuda'
        
        # 创建测试数据
        current_state = {
            'position': torch.randn(num_envs, 3, device=device),
            'orientation': torch.randn(num_envs, 4, device=device),
            'linear_velocity': torch.randn(num_envs, 3, device=device),
            'angular_velocity': torch.randn(num_envs, 1, device=device)
        }
        
        actions = torch.randn(num_envs, 3, device=device)
        position_error = torch.randn(num_envs, device=device)
        heading_error = torch.randn(num_envs, device=device)
        berth_corners = torch.randn(num_envs, 4, 2, device=device)
        
        # 获取峰值内存
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2
        print(f"峰值GPU内存使用: {peak_memory:.2f} MB")
        print(f"内存增长: {peak_memory - initial_memory:.2f} MB")
        
        # 清理
        del current_state, actions, position_error, heading_error, berth_corners
        torch.cuda.empty_cache()
        
        final_memory = torch.cuda.memory_allocated() / 1024**2
        print(f"清理后GPU内存: {final_memory:.2f} MB")
    else:
        print("CUDA不可用，跳过内存测试")

def main():
    """主函数"""
    print("泊船任务性能优化测试")
    print("=" * 50)
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    print(f"PyTorch版本: {torch.__version__}")
    
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    try:
        # 测试奖励函数性能
        reward_time = test_reward_function_performance()
        
        # 测试碰撞检测性能
        collision_time = test_collision_detection_performance()
        
        # 测试内存使用
        test_memory_usage()
        
        # 总结
        print("\n=== 性能优化总结 ===")
        print(f"奖励函数计算: {reward_time*1000:.2f} 毫秒/100次")
        print(f"碰撞检测: {collision_time*1000:.2f} 毫秒/100次")
        print(f"总计算时间: {(reward_time + collision_time)*1000:.2f} 毫秒/100次")
        
        if device == 'cuda':
            print("\n优化建议:")
            print("1. 使用向量化操作避免Python循环")
            print("2. 使用torch.linalg.norm替代torch.norm")
            print("3. 批量处理张量操作")
            print("4. 减少不必要的中间变量创建")
            print("5. 使用广播机制进行批量计算")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
