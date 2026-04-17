import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh

def calculate_unit_mass_rent(G):
    """
    计算单节点平均质量租金：
    U_node = (sum of sqrt(evals)) / number_of_nodes
    这代表了该拓扑环境下，每个节点必须支付的平均“时钟租金”。
    """
    if G.number_of_nodes() == 0: return 0
    L = nx.laplacian_matrix(G).astype(float)
    evals = np.linalg.eigvalsh(L.toarray())
    # 排除零模，计算内部频率总和
    total_rent = np.sum(np.sqrt(evals[evals > 1e-6]))
    return total_rent / G.number_of_nodes()

# --- 实验 1：拓扑“狠度”对比 (线 vs 环 vs 团) ---
n_nodes = 6
line = nx.path_graph(n_nodes)      # 1D 线 (Weak)
ring = nx.cycle_graph(n_nodes)     # 1D 环 (EM)
clique = nx.complete_graph(n_nodes) # K6 团簇 (Strong)

rent_line = calculate_unit_mass_rent(line)
rent_ring = calculate_unit_mass_rent(ring)
rent_clique = calculate_unit_mass_rent(clique)

print(f"--- Experiment 1: Unit Mass Rent (n={n_nodes}) ---")
print(f"1D Line (Weak)   Unit Rent: {rent_line:.4f}")
print(f"1D Ring (EM)     Unit Rent: {rent_ring:.4f}")
print(f"K-Clique (Strong) Unit Rent: {rent_clique:.4f}")
print(f"Strong/Weak Ratio: {rent_clique / rent_line:.4f}")

# --- 实验 2：质量生成的带宽挤兑 (修正Bug版) ---
N_side = 10
G_grid = nx.grid_2d_graph(N_side, N_side)
# 将坐标节点 (x,y) 映射为简单的整数，避开 numpy 的报错
G = nx.convert_node_labels_to_integers(G_grid)
nodes_list = list(G.nodes())

clique_densities = np.arange(0, 16, 2)
total_u = []
manifest_pv = []
B_TOTAL = 500.0 # 系统总带宽定额 Q

for d in clique_densities:
    temp_G = G.copy()
    for _ in range(d):
        # 随机选择4个点组成 K4 团簇（强力锚点）
        if len(nodes_list) >= 4:
            target_nodes = np.random.choice(nodes_list, 4, replace=False)
            for i in range(4):
                for j in range(i+1, 4):
                    temp_G.add_edge(target_nodes[i], target_nodes[j])
    
    # 计算总质量 U
    u_val = np.sum(np.sqrt(np.linalg.eigvalsh(nx.laplacian_matrix(temp_G).toarray().astype(float))))
    total_u.append(u_val)
    manifest_pv.append(max(0, B_TOTAL - u_val))

# --- 可视化 ---
plt.figure(figsize=(14, 5))
plt.subplot(121)
plt.bar(['Line', 'Ring', 'Clique'], [rent_line, rent_ring, rent_clique], color=['gray', 'blue', 'red'])
plt.title("Unit Mass Rent by Topology")
plt.ylabel("Internal Cost per Node")

plt.subplot(122)
plt.plot(clique_densities, total_u, 'r-o', label='Internal Mass (U)')
plt.plot(clique_densities, manifest_pv, 'g-s', label='Manifest Space (pV)')
plt.axhline(y=0, color='black', linestyle='-')
plt.title("The 'Anemia' of Space: Mass (U) vs Space (pV)")
plt.xlabel("Number of Strong Force Anchors")
plt.ylabel("Enthalpy Budget Q")
plt.legend()
plt.tight_layout()
plt.show()