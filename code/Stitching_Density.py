import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree, Delaunay
from scipy.sparse.csgraph import dijkstra
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# ============================================================
# 核心物理设定：质量驱动
# ============================================================
N = 400
BOX_SIZE = 50.0
# 注入一个中心质量，它会产生“带宽挤兑”，强迫空间缝合
CENTRAL_MASS_DENSITY = 500.0 

def generate_base_sponge(N, box_size):
    pts = np.random.rand(N, 2) * box_size
    tri = Delaunay(pts)
    edges = set()
    for s in tri.simplices:
        for i in range(3):
            u, v = sorted((s[i], s[(i+1)%3]))
            edges.add((u, v))
    row, col, data = [], [], []
    for u, v in edges:
        d = np.linalg.norm(pts[u] - pts[v])
        row.extend([u, v]); col.extend([v, u]); data.extend([d, d])
    return sp.csr_matrix((data, (row, col)), shape=(N, N)), pts

# ============================================================
# 变分泛函：成本 vs 排泄压力
# ============================================================
def calculate_enthalpy(params, pts, base_adj, center_idx, r_vals, r_max):
    # 1. 生成 gamma 场 (使用简单的衰减模型减少参数干扰)
    # gamma(r) = a * exp(-r/b)
    a, b = params
    gamma_field = a * np.exp(-r_vals / (b + 1e-3))
    
    # 2. 构建缝合网络
    row, col, data = [], [], []
    adj_coo = base_adj.tocoo()
    row = adj_coo.row.tolist(); col = adj_coo.col.tolist(); data = adj_coo.data.tolist()
    
    tree = cKDTree(pts)
    stitching_cost = 0
    for i in range(len(pts)):
        r_stitch = 2.5 * (1.0 + gamma_field[i])
        indices = tree.query_ball_point(pts[i], r=r_stitch)
        for j in indices:
            if i < j:
                d = np.linalg.norm(pts[i] - pts[j])
                row.extend([i, j]); col.extend([j, i]); data.extend([d, d])
                # 缝合租金
                stitching_cost += d * 0.05 

    full_adj = sp.csr_matrix((data, (row, col)), shape=(N, N))
    
    # 3. 核心计算：信息流的“排泄压力”
    # 质量产生的压力需要通过最短路径消散到边缘
    dist = dijkstra(full_adj, indices=center_idx, directed=False)
    dist = dist[np.isfinite(dist)]
    
    # 压强能：如果路径很长，中心质量产生的“焓”就无法排散，导致系统能量激增
    # E_pressure = Mass * Average_Path_Length
    pressure_energy = CENTRAL_MASS_DENSITY * np.mean(dist)
    
    # H = 缝合租金 + 压强能
    # 系统为了最小化 H，必须通过缝合来减小 pressure_energy
    return stitching_cost + pressure_energy

def main():
    print("N.E.A. 变分实验：质量驱动的缝合场涌现...")
    base_adj, pts = generate_base_sponge(N, BOX_SIZE)
    center_idx = np.argmin(np.linalg.norm(pts - BOX_SIZE/2, axis=1))
    r_vals = np.linalg.norm(pts - pts[center_idx], axis=1)
    r_max = r_vals.max()

    # 初始参数：[强度, 衰减尺度]
    init_params = [1.0, 10.0] 
    
    print("正在寻找总焓极小值的 gamma(r) 分布...")
    res = minimize(
        calculate_enthalpy, init_params,
        args=(pts, base_adj, center_idx, r_vals, r_max),
        method='L-BFGS-B',
        bounds=[(0.0, 10.0), (1.0, 50.0)]
    )

    a_opt, b_opt = res.x
    gamma_opt = a_opt * np.exp(-r_vals / b_opt)

    # 绘图
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(np.sort(r_vals), gamma_opt[np.argsort(r_vals)], 'r-', label='Optimal $\gamma(r)$')
    plt.title("Emergent Stitching Field")
    plt.xlabel("Radius"); plt.ylabel("Intensity"); plt.legend()

    plt.subplot(1, 2, 2)
    # 模拟等效引力势 Phi = -gamma
    plt.scatter(r_vals, -gamma_opt, c='blue', s=10, alpha=0.6, label='Inferred Potential')
    plt.title("Emergent Gravitational Potential Profile")
    plt.xlabel("Radius"); plt.ylabel("Potential Phi"); plt.legend()
    
    plt.tight_layout()
    plt.show()

    print(f"最优解 -> 强度: {a_opt:.4f}, 衰减尺度: {b_opt:.4f}")
    if a_opt > 0.1:
        print("!!! 理论合龙：引力场自发涌现 !!!")
        print("原因：中心质量产生的‘排泄压力’强迫空间节点进行了高强度缝合。")
    else:
        print("失败：质量压力不足以抵消缝合租金。")

if __name__ == "__main__":
    main()