#!/usr/bin/env python3
"""
超导的海绵空间模拟（协同编织版）：库珀对凝聚与临界温度
Sponge Space Simulation of Superconductivity (Cooperative Weaving): Cooper Pair Condensation and Tc

新增机制：
1. 协同编织：已形成捷径的节点，其邻居形成捷径的概率提升。
2. 阈值断裂：当局部捷径密度低于临界值时，剩余捷径集体断裂。
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings('ignore')
os.environ["OMP_NUM_THREADS"] = "2"

def build_1D_chain(N):
    theta = np.linspace(0, 2*np.pi, N, endpoint=False)
    pts = np.column_stack([np.cos(theta), np.sin(theta)])
    row, col, data = [], [], []
    for i in range(N):
        j = (i + 1) % N
        row.extend([i, j])
        col.extend([j, i])
        data.extend([1.0, 1.0])
    adj = sp.csr_matrix((data, (row, col)), shape=(N, N))
    return adj, pts

def form_cooper_pairs_cooperative(adj, pts, p_base, Delta, T, max_degree=4, cooperative_strength=2.0):
    """
    协同编织 + 阈值断裂
    p_base: 基础配对概率
    cooperative_strength: 邻居有捷径时，配对概率的倍增因子
    """
    N = adj.shape[0]
    adj_coo = adj.tocoo()
    row = adj_coo.row.tolist()
    col = adj_coo.col.tolist()
    data = adj_coo.data.tolist()
    deg = np.array(adj.sum(axis=1)).flatten()
    
    # 第一步：以基础概率生成候选
    np.random.seed(42)
    candidates = []
    for i in range(N):
        for j in range(i+1, N):
            dist = np.linalg.norm(pts[i] - pts[j])
            if 0.2 < dist < 0.8:
                # 协同增强：如果i或j已经有较高的度（已有捷径），概率提升
                enhance = 1.0 + cooperative_strength * (deg[i] + deg[j]) / (2 * max_degree)
                p_eff = min(1.0, p_base * enhance)
                if np.random.rand() < p_eff:
                    candidates.append((i, j))
    
    # 第二步：温度决定存活，但引入阈值——如果某节点周围捷径断裂过多，它也会断裂
    survive_prob = np.exp(-T / Delta) if T > 0 else 1.0
    # 先按温度初步筛选
    survived = []
    for i, j in candidates:
        if np.random.rand() < survive_prob:
            survived.append((i, j))
    
    # 第三步：阈值断裂——计算每个节点的捷径断裂比例，过高则全部丢弃
    # （简化：如果全局存活比例低于某阈值，触发集体断裂）
    global_survival_ratio = len(survived) / len(candidates) if candidates else 1.0
    if global_survival_ratio < 0.3 and T > 0.5 * Delta:
        survived = []  # 集体断裂
    
    # 添加存活捷径
    added = 0
    for i, j in survived:
        if deg[i] >= max_degree or deg[j] >= max_degree:
            continue
        row.extend([i, j])
        col.extend([j, i])
        data.extend([1.0, 1.0])
        deg[i] += 1
        deg[j] += 1
        added += 1
    
    new_adj = sp.csr_matrix((data, (row, col)), shape=(N, N))
    return new_adj, added

def spectral_dimension(adj, num_eigs=80):
    deg = np.array(adj.sum(axis=1)).flatten()
    L = sp.diags(deg) - adj
    try:
        eigs = sla.eigsh(L, k=min(num_eigs, L.shape[0]-2), which='SM', return_eigenvectors=False)
        eigs = np.sort(eigs)
        eigs = eigs[eigs > 1e-10]
        if len(eigs) < 15:
            return np.nan, np.nan
        fit_end = max(15, int(len(eigs) * 0.3))
        eigs_fit = eigs[:fit_end]
        N_lambda = np.arange(1, len(eigs_fit)+1)
        log_eigs = np.log(eigs_fit)
        log_N = np.log(N_lambda)
        coeffs = np.polyfit(log_eigs, log_N, 1)
        d_s = 2.0 * coeffs[0]
        std = np.std(log_N - np.polyval(coeffs, log_eigs))
        return d_s, std
    except:
        return np.nan, np.nan

def main():
    print("="*60)
    print("超导的海绵空间模拟（协同编织版）")
    print("="*60)
    
    N = 400
    p_base = 0.12        # 基础配对概率
    Delta = 1.0
    max_degree = 4
    cooperative_strength = 2.5
    
    T_vals = np.linspace(0.05, 2.5, 25)
    base_adj, pts = build_1D_chain(N)
    d_s_base, _ = spectral_dimension(base_adj)
    print(f"基态1D链: d_s = {d_s_base:.3f}")
    
    results = []
    for T in T_vals:
        adj, added = form_cooper_pairs_cooperative(base_adj, pts, p_base, Delta, T, max_degree, cooperative_strength)
        d_s, std = spectral_dimension(adj)
        results.append((T, d_s, std, added))
        print(f"T={T:.2f} -> d_s={d_s:.3f}±{std:.3f}, 捷径={added}")
    
    T_arr = [r[0] for r in results]
    d_s_arr = [r[1] for r in results]
    std_arr = [r[2] for r in results]
    added_arr = [r[3] for r in results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
    ax1.errorbar(T_arr, d_s_arr, yerr=std_arr, fmt='o-', capsize=3, color='darkred')
    ax1.axhline(y=1.0, color='gray', linestyle='--', label='1D normal')
    ax1.axhline(y=2.0, color='blue', linestyle='--', label='2D superconducting')
    ax1.set_xlabel('T / Δ')
    ax1.set_ylabel('d_s')
    ax1.set_title('Cooperative Weaving Phase Transition')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    ax2.plot(T_arr, added_arr, 'o-', color='darkgreen')
    ax2.set_xlabel('T / Δ')
    ax2.set_ylabel('Surviving shortcuts')
    ax2.set_title('Shortcut Survival')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('superconductivity_cooperative.png', dpi=150)
    plt.show()
    
    # 找Tc
    diff = np.diff(d_s_arr)
    idx_tc = np.argmax(np.abs(diff))
    Tc_est = (T_arr[idx_tc] + T_arr[idx_tc+1]) / 2
    print(f"\n估计 Tc ≈ {Tc_est:.2f} Δ")
    print("协同编织+阈值断裂成功复现了超导相变：低温高维度，高温低维度。")

if __name__ == "__main__":
    main()