import numpy as np
import networkx as nx

def audit_force_roi():
    # 模拟不同尺度下的“带宽市场”
    # 尺度扫描 (log10)
    scales = np.arange(-35, 1, 1) 
    
    print(f"{'Scale (10^x)':<12} | {'Weak ROI':<10} | {'Strong ROI':<10} | {'EM ROI':<10} | {'Grav ROI':<10} | {'Winner'}")
    print("-" * 80)

    for s in scales:
        r = 10**float(s)
        
        # 核心逻辑：计算在该尺度下，不同力的“利润”
        # 利润 = 路径缩短收益 - 拓扑租金成本
        
        # 1. 弱力 (0->1): 创世利润随距离指数衰减
        weak_profit = 15.0 * np.exp(-(s + 35)/5.0) - 2.0
        
        # 2. 强力 (3->4): 随密度增加利润激增，但在宏观处租金爆炸
        # 它在微观(s < -15)处利润极高
        strong_profit = 10.0 / (abs(s + 18) + 1) - 1.25
        
        # 3. 电磁力 (1->2): 2D织造，利润极其平稳
        em_profit = 5.0 - 1.5 # 长期稳定的现金流
        
        # 4. 引力 (2->3): 缝合收益随规模N增加，但在微观处路径收益太小
        # 引力在 s > -10 处收益才开始超过 1.33 的门槛
        grav_profit = 3.0 * (s + 20) / 40.0 - 1.33

        # 结果对账
        profits = [weak_profit, strong_profit, em_profit, grav_profit]
        names = ["Weak", "Strong", "EM", "Gravity"]
        
        # 只有利润 > 0 的协议才上线
        active_names = [names[i] for i, p in enumerate(profits) if p > 0]
        winner = active_names[np.argmax([profits[i] for i, p in enumerate(profits) if p > 0])] if active_names else "NONE"

        print(f"10^{s:<9} | {weak_profit:<10.2f} | {strong_profit:<10.2f} | {em_profit:<10.2f} | {grav_profit:<10.2f} | {winner}")

if __name__ == "__main__":
    audit_force_roi()