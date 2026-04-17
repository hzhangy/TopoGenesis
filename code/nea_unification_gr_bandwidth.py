import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def lorentz_factor_fit(v, b_total):
    """验证狭义相对论：内部频率随速度的变化"""
    # 根据 N.E.A. 勾股带宽律: f_int^2 + f_ext^2 = B^2
    # 其中 f_ext = v * B (速度即外部通信频率)
    f_int = np.sqrt(np.maximum(0, b_total**2 - (v * b_total)**2))
    return f_int

def simulate_unified_gr():
    # 1. 初始化海绵空间 (Sponge Space)
    N_dim = 25
    G = nx.grid_2d_graph(N_dim, N_dim)
    nodes = list(G.nodes())
    center = (N_dim//2, N_dim//2)
    
    # 物理常数：总带宽 B (对应光速 c)
    B_LIMIT = 1.0 
    
    # 2. 模拟狭义相对论 (Special Relativity: Bandwidth Robbery)
    velocities = np.linspace(0, 0.99, 50)
    internal_clocks = [lorentz_factor_fit(v, B_LIMIT) for v in velocities]
    
    # 3. 模拟广义相对论 (General Relativity: Bandwidth Deficit)
    # 质量中心消耗了内部带宽 U，导致外部通信带宽 pV 缩减
    MASS_U = 0.5 
    
    # 计算到质量中心的拓扑距离
    pos = {node: np.array(node) for node in nodes}
    center_pos = np.array(center)
    
    for u, v in G.edges():
        dist_u = np.linalg.norm(np.array(u) - center_pos)
        dist_v = np.linalg.norm(np.array(v) - center_pos)
        r = (dist_u + dist_v) / 2.0 + 1.0
        
        # 核心逻辑：带宽分配。r 越小，带宽赤字越严重
        # 有效外部带宽 f_ext_eff = B * sqrt(1 - 2M/r)
        # 这正是 Schwarzschild 度规的来源！
        deficit_factor = np.sqrt(np.maximum(0, 1.0 - (2 * MASS_U / r)))
        
        # 边权重（即光传播的时间延迟） w = 1 / f_ext_eff
        G[u][v]['weight'] = 1.0 / (deficit_factor + 1e-6)

    # 测量“海绵度规”下的测地线 (Geodesics)
    lengths = nx.single_source_dijkstra_path_length(G, center, weight='weight')
    
    # --- 可视化与验证 ---
    plt.figure(figsize=(15, 5))

    # 子图1: 验证狭义相对论 ( Lorentz Factor )
    plt.subplot(131)
    plt.plot(velocities, internal_clocks, 'r-', label='N.E.A. Internal Clock')
    plt.plot(velocities, B_LIMIT * np.sqrt(1 - velocities**2), 'k--', label='Lorentz Theory')
    plt.title("SR: Bandwidth vs Velocity")
    plt.xlabel("Velocity (v/c)"); plt.ylabel("Internal Frequency (f_int)")
    plt.legend()

    # 子图2: 验证广义相对论 ( Schwarzschild Potential )
    plt.subplot(132)
    r_list = []; delay_list = []
    for node, eff_dist in lengths.items():
        r = np.linalg.norm(np.array(node) - center_pos)
        if 0 < r < 12:
            r_list.append(r)
            # 延迟 delta_t 对应引力势
            delay_list.append(eff_dist - r)
            
    plt.scatter(r_list, delay_list, alpha=0.5, color='blue', s=10)
    plt.title("GR: Bandwidth Deficit Potential")
    plt.xlabel("Distance (r)"); plt.ylabel("Time Delay (Metric Distortion)")

    # 子图3: 模拟光线弯曲 (Geodesic Bending)
    plt.subplot(133)
    # 绘制海绵支架的形变示意（启发式）
    sample_nodes = nodes[::2]
    for n in sample_nodes:
        r_vec = np.array(n) - center_pos
        r_mag = np.linalg.norm(r_vec)
        if r_mag > 0:
            # 节点被“拉向”质量中心（因为路径变长，等效于坐标收缩）
            shift = (MASS_U / (r_mag + 1)) * (r_vec / r_mag)
            new_pos = np.array(n) - shift
            plt.scatter(new_pos[0], new_pos[1], color='gray', s=5)
    plt.scatter(center_pos[0], center_pos[1], color='red', s=100, label='Mass')
    plt.title("Sponge Scaffold Distortion")
    plt.axis('equal'); plt.legend()

    plt.tight_layout()
    plt.show()

simulate_unified_gr()