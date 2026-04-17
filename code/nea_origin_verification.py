import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh
from scipy.stats import linregress

def calculate_ds_refined(G):
    """精细计算谱维度"""
    if G.number_of_edges() == 0: return 0.0
    nodes_cc = max(nx.connected_components(G), key=len)
    sub = G.subgraph(nodes_cc)
    if len(sub) < 50: return 1.0
    L = nx.laplacian_matrix(sub).astype(float)
    k = min(120, L.shape[0] - 2)
    evals = eigsh(L, k=k, which='SM', return_eigenvectors=False)
    evals = np.sort(evals[evals > 1e-8])
    log_lambda = np.log(evals)
    log_N = np.log(np.arange(1, len(evals) + 1))
    slope, _, _, _, _ = linregress(log_lambda, log_N)
    return 2 * slope

# ==========================================
# 实验 A：步长的局域涌现 (Variational Stride)
# 逻辑：力是节点为了优化“延迟/成本”比自发产生的
# ==========================================
def experiment_stride_emergence():
    print("Running Experiment A: Variational Stride Emergence...")
    N = 400
    # 基础1D链 (Weak)
    G = nx.path_graph(N)
    
    # 定义不同“密度”（可用预算）下的最优连边
    budgets = np.linspace(0.1, 1.5, 5) # 节点平均度数预算
    selected_strides = []
    
    for b in budgets:
        # 寻找能最大化提升全局效率（缩短平均路径）的跨度 stride
        efficiency_gains = []
        strides = np.arange(2, N//4)
        for s in strides:
            temp_G = G.copy()
            # 模拟在此步长下建立连接
            for i in range(0, N, s):
                temp_G.add_edge(i, (i + s) % N)
            # 效率增益 = 1 / 平均路径长度
            gain = 1.0 / nx.average_shortest_path_length(temp_G)
            efficiency_gains.append(gain)
        
        best_s = strides[np.argmax(efficiency_gains)]
        selected_strides.append(best_s)
        print(f"Budget (Density): {b:.2f} | Emergent Optimal Stride: {best_s}")
    return budgets, selected_strides

# ==========================================
# 实验 B：q 的自发塌缩 (The Origin of q)
# 逻辑：带宽赤字 -> 缝合线断裂 -> 维度塌缩 (G-Paper Bridge)
# ==========================================
def experiment_q_collapse():
    print("\nRunning Experiment B: Bandwidth-Driven Dimension Collapse...")
    N = 800
    G_base = nx.path_graph(N)
    # 模拟2D编织 (EM)
    sqrt_N = int(np.sqrt(N))
    for i in range(N): G_base.add_edge(i, (i + sqrt_N) % N)
    
    # 模拟从“高带宽中心”到“低带宽边缘”
    # 带宽预算 B(r) 随半径下降
    radii = np.linspace(1, 20, 10)
    ds_profile = []
    
    for r in radii:
        G = G_base.copy()
        # 引力缝合概率 P ~ 1/r (带宽赤字)
        stitch_prob = np.exp(-r / 5.0) 
        n_stiches = int(N * stitch_prob)
        for _ in range(n_stiches):
            u, v = np.random.choice(N, 2, replace=False)
            G.add_edge(u, v)
        
        ds = calculate_ds_refined(G)
        ds_profile.append(ds)
        print(f"Distance r: {r:.2f} | Budget-limited ds (q+1): {ds:.4f}")
    return radii, ds_profile

# ==========================================
# 实验 C：谱频率质量定标 (The Origin of Mass)
# 逻辑：验证 U = sum(sqrt(lambda)) 作为计算步数的物理等效性
# ==========================================
def experiment_mass_origin():
    print("\nRunning Experiment C: Mass-Frequency Equivalence...")
    # 对比：1D线, 2D格点, 3D海绵, 4D团簇
    topologies = {
        "1D_Line": nx.path_graph(50),
        "2D_Grid": nx.grid_2d_graph(7, 7),
        "3D_Sponge": nx.connected_watts_strogatz_graph(50, 6, 0.2),
        "4D_K4_Cluster": nx.complete_graph(10)
    }
    
    mass_results = {}
    for name, G in topologies.items():
        L = nx.laplacian_matrix(G).toarray().astype(float)
        evals = np.linalg.eigvalsh(L)
        # 物理本源：维持拓扑需要的总刷新频率
        mass_rent = np.sum(np.sqrt(np.maximum(0, evals[evals > 1e-10])))
        # 归一化每个节点的贡献
        mass_results[name] = mass_rent / G.number_of_nodes()
        print(f"Topology: {name:15s} | Unit Mass Rent: {mass_results[name]:.4f}")
    return mass_results

# --- 执行并可视化 ---
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# Plot A
b, s = experiment_stride_emergence()
axs[0].plot(b, s, 'o-g', linewidth=2)
axs[0].set_title("A: Stride Emergence vs Density")
axs[0].set_xlabel("Computational Density"); axs[0].set_ylabel("Optimal Stride")

# Plot B
r, ds = experiment_q_collapse()
axs[1].plot(r, ds, 's-r', linewidth=2)
axs[1].axhline(y=2.0, color='k', linestyle='--', label='2D Limit')
axs[1].set_title("B: Dimensional Collapse (q-scaling)")
axs[1].set_xlabel("Distance from Center (r)"); axs[1].set_ylabel("Spectral Dimension ds")
axs[1].legend()

# Plot C
m = experiment_mass_origin()
axs[2].bar(m.keys(), m.values(), color=['blue', 'cyan', 'orange', 'red'])
axs[2].set_title("C: Unit Mass Rent per Topology")
axs[2].set_ylabel("U_unit (sum sqrt(lambda)/N)")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()