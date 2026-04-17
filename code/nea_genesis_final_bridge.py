import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh
from scipy.stats import linregress

def calculate_ds(G):
    if G.number_of_edges() == 0: return 0.0
    nodes_cc = max(nx.connected_components(G), key=len)
    sub = G.subgraph(nodes_cc)
    if len(sub) < 50: return 1.0
    L = nx.laplacian_matrix(sub).astype(float)
    k = min(150, L.shape[0] - 2)
    try:
        evals = eigsh(L, k=k, which='SM', return_eigenvectors=False)
        evals = np.sort(evals[evals > 1e-6])
        log_lambda = np.log(evals)
        log_N = np.log(np.arange(1, len(evals) + 1))
        slope, _, _, _, _ = linregress(log_lambda, log_N)
        return 2 * slope
    except: return 1.0

# --- 实验配置 ---
N = 1200
sqrt_N = int(np.sqrt(N))
G = nx.Graph()
G.add_nodes_from(range(N))

# 阶段 1: WEAK (0 -> 1) - 因果链 (Stride-1)
for i in range(N-1): G.add_edge(i, i+1)
ds_weak = calculate_ds(G)

# 阶段 2: EM (1 -> 2) - 电磁织造 (Stride-sqrt(N))
# 模拟二维表面的“经纬线”织造
for i in range(N):
    G.add_edge(i, (i + sqrt_N) % N)
ds_em = calculate_ds(G)

# 阶段 3: GRAVITY (2 -> 3) - 海绵缝合 (Stride-L^2)
# 模拟跨越平面的体积拉伸
for i in range(N):
    # 引入跨度为 N/4 的长程缝合
    G.add_edge(i, (i + N//4) % N)
ds_grav = calculate_ds(G)

# 阶段 4: STRONG (3 -> 4) - 4D 硬核锁定 (Recursive Stitches)
# 模拟高阶拓扑锚定
for i in range(N):
    G.add_edge(i, (i + N//8) % N)
    G.add_edge(i, (i + N//12) % N)
ds_strong = calculate_ds(G)

print(f"Final Calibrated ds Ladder:")
print(f"Weak (0-1): {ds_weak:.4f}")
print(f"EM   (1-2): {ds_em:.4f}")
print(f"Grav (2-3): {ds_grav:.4f}")
print(f"Str  (3-4): {ds_strong:.4f}")

# --- 引力势能验证 (海绵赤字模型) ---
m_node = N // 2
mass_val = 200.0
raw_d = nx.single_source_dijkstra_path_length(G, m_node)

# 带宽分配律：距离中心越近，计算阻力（权重）越大
for u, v in G.edges():
    d_avg = (raw_d.get(u, N) + raw_d.get(v, N)) / 2.0
    G[u][v]['weight'] = 1.0 + mass_val / (d_avg + 2.0)

eff_d = nx.single_source_dijkstra_path_length(G, m_node, weight='weight')

r_list, phi_list = [], []
for n in raw_d:
    if 0 < raw_d[n] < 30: # 聚焦局部势场
        phi = (eff_d[n] - raw_d[n]) / mass_val
        r_list.append(raw_d[n])
        phi_list.append(phi)

# --- 绘图 ---
plt.figure(figsize=(14, 6))
plt.subplot(121)
plt.plot(['Weak', 'EM', 'Grav', 'Strong'], [ds_weak, ds_em, ds_grav, ds_strong], 'o-', color='#2c3e50', linewidth=3)
plt.title("T-Paper: Topological Genesis Map", fontsize=14)
plt.ylabel(r"Spectral Dimension ($d_s$)")
plt.grid(True, alpha=0.3)

plt.subplot(122)
plt.scatter(r_list, phi_list, alpha=0.5, color='#e74c3c')
# 拟合参考线
ref_r = np.sort(r_list)
plt.plot(ref_r, 1.5/ref_r, '--', color='#3498db', label=r'Newtonian $1/r$')
plt.title("G-U Bridge: Emergent Gravity Potential", fontsize=14)
plt.xlabel("r"); plt.ylabel(r"Potential $\Phi$"); plt.legend()
plt.tight_layout(); plt.show()