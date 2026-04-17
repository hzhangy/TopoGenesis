#!/usr/bin/env python3
"""
化学键的海绵空间模拟（完全版）：H₂⁺ 的缝合机制
Sponge Space Simulation of Chemical Bonding (Complete): Stitching in H₂⁺

本版本加入：
1. 加权最短路径（缝合成本）
2. 电子维持成本（常数）
3. 电子-电子及电子-核重叠惩罚（产生左半支上升）
"""

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import dijkstra
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings('ignore')

def build_h2plus_network(distance, n_electrons=2, spread=0.25):
    """
    构建网络：两个核 + 在核间区域的电子节点。
    电子位置：对称分布在核间连线的两侧，且随距离自适应。
    """
    anchor1 = np.array([-distance/2, 0.0])
    anchor2 = np.array([ distance/2, 0.0])
    
    # 电子位置：使其能有效缝合两个核
    # 电子1偏向核1，电子2偏向核2，同时保持一定间距
    x1 = -distance/2 + distance * 0.35
    x2 =  distance/2 - distance * 0.35
    y1 =  distance * spread
    y2 = -distance * spread
    electron_pos = np.array([[x1, y1], [x2, y2]])
    
    pts = np.vstack([anchor1.reshape(1,2), anchor2.reshape(1,2), electron_pos])
    N = pts.shape[0]
    anchor1_idx, anchor2_idx = 0, 1
    electron_indices = [2, 3]
    
    # 连接半径
    r_connect = distance * 0.9
    tree = cKDTree(pts)
    pairs = tree.query_pairs(r=r_connect, output_type='ndarray')
    
    row, col, data = [], [], []
    for u, v in pairs:
        u, v = int(u), int(v)
        dist = np.linalg.norm(pts[u] - pts[v])
        row.extend([u, v])
        col.extend([v, u])
        data.extend([dist, dist])
    
    adj = sp.csr_matrix((data, (row, col)), shape=(N, N))
    return adj, anchor1_idx, anchor2_idx, electron_indices, pts

def compute_enthalpy(adj, anchor1, anchor2, electron_indices, pts, 
                     electron_cost=0.5, overlap_penalty=0.3):
    """
    总焓 = 缝合路径成本 + 电子维持成本 + 重叠惩罚。
    overlap_penalty: 模拟电子-电子及电子-核的排斥。
    """
    try:
        dist_matrix = dijkstra(adj, indices=anchor1, directed=False)
        path_cost = dist_matrix[anchor2]
    except:
        path_cost = 1e6
    
    electron_count = len(electron_indices)
    H = path_cost + electron_cost * electron_count
    
    # 重叠惩罚：电子-电子距离的倒数 + 电子-对方核距离的倒数
    e1, e2 = electron_indices[0], electron_indices[1]
    # 电子-电子
    d_ee = np.linalg.norm(pts[e1] - pts[e2])
    if d_ee > 1e-6:
        H += overlap_penalty / d_ee
    # 电子-对方核
    d_e1_a2 = np.linalg.norm(pts[e1] - pts[anchor2])
    d_e2_a1 = np.linalg.norm(pts[e2] - pts[anchor1])
    H += overlap_penalty * (1.0/d_e1_a2 + 1.0/d_e2_a1)
    
    return H

def main():
    print("="*60)
    print("海绵空间化学键模拟（完全版）：H₂⁺ 的缝合机制")
    print("="*60)
    
    distances = np.linspace(0.5, 5.0, 25)
    H_vals = []
    for d in distances:
        adj, a1, a2, e_idx, pts = build_h2plus_network(d)
        H = compute_enthalpy(adj, a1, a2, e_idx, pts)
        H_vals.append(H)
        print(f"d = {d:.3f} -> H = {H:.4f}")
    
    best_idx = np.argmin(H_vals)
    best_d = distances[best_idx]
    best_H = H_vals[best_idx]
    
    plt.figure(figsize=(8,5))
    plt.plot(distances, H_vals, 'o-', color='navy')
    plt.axvline(best_d, color='red', linestyle='--', label=f'Optimal d = {best_d:.3f}')
    plt.xlabel('Internuclear distance d')
    plt.ylabel('Total Enthalpy H')
    plt.title('Sponge Space H₂⁺ Bonding (Complete Model)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('h2plus_complete.png', dpi=150)
    plt.show()
    
    print("\n" + "="*60)
    print(f"最优缝合距离: d_opt = {best_d:.3f}")
    print(f"对应最小焓: H_min = {best_H:.4f}")
    print("物理意义：")
    print("  当核间距过小，电子-电子及电子-核重叠惩罚急剧升高；")
    print("  当核间距过大，缝合路径长度增加，信息延迟成本升高。")
    print("  最优距离是这两种竞争机制的平衡，对应化学键的平衡键长。")
    print("="*60)

if __name__ == "__main__":
    main()