#!/usr/bin/env python3
"""
验证187个高度采样点的计算过程
"""

import torch

def verify_height_points():
    """验证高度采样点的计算"""
    
    # 从配置中获取的测量点
    measured_points_x = [
        -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8
    ]  # 17个X坐标点
    
    measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]  # 11个Y坐标点
    
    print(f"X坐标点数量: {len(measured_points_x)}")
    print(f"Y坐标点数量: {len(measured_points_y)}")
    print(f"总点数: {len(measured_points_x)} × {len(measured_points_y)} = {len(measured_points_x) * len(measured_points_y)}")
    
    # 创建网格
    x = torch.tensor(measured_points_x)
    y = torch.tensor(measured_points_y)
    grid_x, grid_y = torch.meshgrid(x, y)
    
    print(f"\ngrid_x形状: {grid_x.shape}")
    print(f"grid_y形状: {grid_y.shape}")
    print(f"grid_x.numel(): {grid_x.numel()}")
    
    # 验证点数
    assert grid_x.numel() == 187, f"期望187个点，实际得到{grid_x.numel()}个点"
    
    # 显示网格点
    print(f"\n网格点坐标:")
    for i in range(min(10, len(measured_points_x))):  # 只显示前10行
        for j in range(min(10, len(measured_points_y))):  # 只显示前10列
            print(f"({grid_x[i,j]:.1f}, {grid_y[i,j]:.1f})", end=" ")
        print()
    
    # 显示所有点的分布
    print(f"\n所有187个点的分布:")
    points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
    print(f"points形状: {points.shape}")
    
    # 统计不同区域的点数
    front_points = points[points[:, 0] > 0]  # 前方
    back_points = points[points[:, 0] < 0]   # 后方
    left_points = points[points[:, 1] < 0]   # 左侧
    right_points = points[points[:, 1] > 0]  # 右侧
    center_points = points[(points[:, 0] >= -0.1) & (points[:, 0] <= 0.1) & 
                          (points[:, 1] >= -0.1) & (points[:, 1] <= 0.1)]  # 中心
    
    print(f"前方点数: {len(front_points)}")
    print(f"后方点数: {len(back_points)}")
    print(f"左侧点数: {len(left_points)}")
    print(f"右侧点数: {len(right_points)}")
    print(f"中心点数: {len(center_points)}")
    
    # 验证覆盖范围
    print(f"\n覆盖范围:")
    print(f"X范围: [{points[:, 0].min():.1f}, {points[:, 0].max():.1f}]")
    print(f"Y范围: [{points[:, 1].min():.1f}, {points[:, 1].max():.1f}]")
    
    print(f"\n✓ 验证通过: 成功生成187个高度采样点")

if __name__ == "__main__":
    verify_height_points() 