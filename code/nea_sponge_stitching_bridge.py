import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from scipy.sparse.linalg import eigsh
from scipy.stats import linregress

def create_2d_atomic_cell(n_nodes, radius=1.0):
    points = np.random.uniform(-radius, radius, (n_nodes, 2))
    tri = Delaunay(points)
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                edges.add(tuple(sorted((simplex[i], simplex[j]))))
    return list(edges), points

def run_stitching_experiment(n_cells=15, nodes_per_cell=150, gamma_range=np.linspace(0, 1, 11)):
    results_ds = []
    print(f"Running Experiment: {n_cells} cells, {nodes_per_cell} nodes/cell")

    for gamma in gamma_range:
        G = nx.Graph()
        node_offset = 0
        cell_nodes_map = []
        
        for i in range(n_cells):
            edges, _ = create_2d_atomic_cell(nodes_per_cell)
            current_nodes = list(range(node_offset, node_offset + nodes_per_cell))
            cell_nodes_map.append(current_nodes)
            for u, v in edges:
                G.add_edge(u + node_offset, v + node_offset)
            node_offset += nodes_per_cell
            
        if gamma > 0:
            for i in range(n_cells - 1):
                n_stiches = int(nodes_per_cell * gamma)
                if n_stiches < 1: n_stiches = 1
                nodes_a = cell_nodes_map[i]
                nodes_b = cell_nodes_map[i+1]
                stich_pairs = np.random.choice(nodes_a, n_stiches, replace=False)
                target_pairs = np.random.choice(nodes_b, n_stiches, replace=False)
                for u, v in zip(stich_pairs, target_pairs):
                    G.add_edge(u, v)

        if nx.is_connected(G):
            L = nx.laplacian_matrix(G).astype(float)
            k_evals = min(200, G.number_of_nodes() - 2)
            evals = eigsh(L, k=k_evals, which='SM', return_eigenvectors=False)
            evals = np.sort(evals)
            evals = evals[evals > 1e-6]
            
            if len(evals) > 10:
                N_lambda = np.arange(1, len(evals) + 1)
                log_lambda = np.log(evals)
                log_N = np.log(N_lambda)
                slope, _, _, _, _ = linregress(log_lambda, log_N)
                ds = 2 * slope
            else:
                ds = 2.0
        else:
            ds = 0.0
            
        results_ds.append(ds)
        print(f"Gamma: {gamma:.2f} | Spectral Dimension ds: {ds:.4f}")

    return gamma_range, results_ds

# Execute
gammas, ds_values = run_stitching_experiment(n_cells=15, nodes_per_cell=150)

# Visualization
plt.figure(figsize=(10, 6))
plt.plot(gammas, ds_values, 'o-', linewidth=2, markersize=8, color='#2c3e50', label='Emergent $d_s$')
plt.axhline(y=2.0, color='r', linestyle='--', label='2D Holographic Limit')
plt.axhline(y=3.0, color='g', linestyle='--', label='3D Macroscopic Limit')
# 修正 xlabel 的小 Bug
plt.xlabel(r"Stitching Intensity ($\Gamma$)", fontsize=12) # 加了 r
plt.ylabel("Spectral Dimension ($d_s$)", fontsize=12)
plt.title("N.E.A. Simulation: 2D Atomic Shells to 3D Sponge Space", fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()