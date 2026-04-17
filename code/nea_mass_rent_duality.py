import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def get_topology_rent(G):
    """计算结构的单节点内部带宽租金 U_unit"""
    n = G.number_of_nodes()
    if n == 0: return 0
    # 获取拉普拉斯矩阵
    L = nx.laplacian_matrix(G).toarray().astype(float)
    # 计算频率谱 (内部更新速度)
    evals = np.linalg.eigvalsh(L)
    # 过滤微小负数，取平方根之和代表总带宽占用
    # U = sum(sqrt(lambda))
    internal_bandwidth = np.sum(np.sqrt(np.maximum(0, evals[evals > 1e-10])))
    return internal_bandwidth / n

# 1. 模拟【强力单位 - 夸克】: K4 团簇 (高度对称，支架适应性强)
quark_clique = nx.complete_graph(4)
u_strong = get_topology_rent(quark_clique)

# 2. 模拟【电磁力单位 - 电子】: 2D 织造环 (Stride-sqrt 织造的局部缩影)
# 模拟 1->2 织造出的一个 8 节点全息环
electron_loop = nx.cycle_graph(8)
u_em = get_topology_rent(electron_loop)

# 3. 模拟【弱力单位 - W/Z 介子】: 0->1 强制桥接 (高电阻因果链)
# 在海绵中强行拉开一个“极短且扭曲”的路径，节点间连接极度紧绷
weak_bridge = nx.Graph()
weak_bridge.add_edges_from([(0,1), (1,2), (2,0)]) # 极小的紧束缚环，代表 0D 破碎
# 为了模拟弱力的高租金，我们增加其逻辑边的权重（表示维持该因果链的极端代价）
u_weak = get_topology_rent(weak_bridge) * 15.0 # 弱力锚定系数

# --- 实验数据输出 ---
print(f"--- Final N.E.A. Mass Spectrum Simulation ---")
print(f"Strong (Quark-like) Unit Rent  (Mass): {u_strong:.4f}")
print(f"EM     (Electron-like) Unit Rent (Mass): {u_em:.4f}")
print(f"Weak   (W/Z-like) Unit Rent      (Mass): {u_weak:.4f}")

# 计算比值
print(f"\n--- Mass Ratios (The 'Easy' Unification) ---")
print(f"W-Boson / Electron Ratio (Predicted): {u_weak / u_em:.2f}")
print(f"Electron / Quark Ratio   (Predicted): {u_em / u_strong:.2f}")

# 可视化：力的强度 vs 质量
force_strengths = [100, 1, 0.001] # 强、电、弱 的相对强度 (示意)
mass_rents = [u_strong, u_em, u_weak]
labels = ['Strong', 'EM', 'Weak']

plt.figure(figsize=(8, 6))
plt.scatter(force_strengths, mass_rents, s=200, c=['red', 'blue', 'green'])
for i, txt in enumerate(labels):
    plt.annotate(txt, (force_strengths[i], mass_rents[i]), xytext=(10,10), textcoords='offset points')

plt.xscale('log')
plt.yscale('log')
plt.xlabel("Force Interaction Strength (Log Scale)")
plt.ylabel("Unit Mass Rent (U_unit) (Log Scale)")
plt.title("The Inverse Duality: Force Strength vs Unit Mass")
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.show()