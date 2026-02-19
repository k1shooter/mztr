import os
import sys
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import imageio
import io
from tqdm import tqdm

# src 폴더를 경로에 추가하여 treedfm 패키지 로드
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

# 1. treedfm 모듈 임포트
# 사용자가 제공한 schedule.py에 새로 추가된 함수들을 임포트합니다.
try:
    from treedfm.schedule import (
        ProfiledDepthSchedule, 
        estimate_depth_profile, 
        make_profiled_sequential_schedule
    )
    from treedfm.corruption import NoiseConfig
    from treedfm.data import MazeTreeDataset
except ImportError as e:
    print(f"Error importing treedfm modules: {e}")
    print("Please ensure src/treedfm/schedule.py is updated with the provided code.")
    sys.exit(1)

# ==========================================
# 2. Graph Layout Utility
# ==========================================
def hierarchy_pos(G, root=None, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
    """NetworkX 트리를 계층형으로 예쁘게 그리기 위한 레이아웃"""
    if not nx.is_tree(G):
        return nx.spring_layout(G)

    def _hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None):
        if pos is None:
            pos = {root: (xcenter, vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)  
        if len(children) != 0:
            dx = width / len(children) 
            nextx = xcenter - width/2 - dx/2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G, child, width=dx, vert_gap=vert_gap, 
                                     vert_loc=vert_loc-vert_gap, xcenter=nextx,
                                     pos=pos, parent=root)
        return pos
    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)

# ==========================================
# 3. Dummy Data Helper (파일이 없을 경우)
# ==========================================
def create_dummy_dataset(num_samples=50, max_depth=20, k=3):
    """실제 데이터 파일이 없을 때 Profile 생성을 위한 더미 데이터셋"""
    dataset = []
    for _ in range(num_samples):
        # Random tree generation
        nodes = [[0, 0, 2, -1]] # Root
        parents = [0]
        current_idx = 1
        
        # Simple BFS-like expansion
        while parents and current_idx < 50: # Limit size
            p_idx = parents.pop(0)
            p_depth = nodes[p_idx][0]
            if p_depth >= max_depth: continue
            
            num_children = np.random.randint(0, k+1)
            if num_children == 0 and len(nodes) < 5: num_children = 1 # 최소 크기 보장
            
            for r in range(num_children):
                ntype = np.random.randint(1, 4)
                nodes.append([p_depth + 1, r, ntype, p_idx])
                parents.append(current_idx)
                current_idx += 1
        
        dataset.append(torch.tensor(nodes, dtype=torch.long))
    return dataset

# ==========================================
# 4. Visualization Logic
# ==========================================
def visualize_treedfm_profiled(pkl_filename="maze_trees_relative.pkl", output_gif="treedfm_profiled_flow.gif", steps=100):
    print(f"Initializing Profiled TreeDFM Visualization...")
    
    # Config
    config = {
        'max_depth': 20,
        'k': 3,
        'num_types': 3
    }
    
    # 1. 데이터 로드 및 Profile 생성
    dataset = None
    pkl_path = None
    
    # 경로 탐색
    if os.path.exists(pkl_filename): pkl_path = pkl_filename
    elif os.path.exists(os.path.join("..", pkl_filename)): pkl_path = os.path.join("..", pkl_filename)
    elif os.path.exists(os.path.join("src", pkl_filename)): pkl_path = os.path.join("src", pkl_filename)

    if pkl_path:
        print(f"Loading real dataset from {pkl_path}...")
        dataset = MazeTreeDataset(pkl_path, max_depth=config['max_depth'], k=config['k'])
    else:
        print(f"Dataset file '{pkl_filename}' not found. Using generated dummy dataset for profile estimation.")
        dataset = create_dummy_dataset(num_samples=100, max_depth=config['max_depth'], k=config['k'])

    # [핵심] 2. 데이터로부터 Depth Profile 추정 (estimate_depth_profile)
    print("Estimating depth profile from dataset...")
    profile = estimate_depth_profile(
        dataset, 
        max_depth=config['max_depth'], 
        count_mode="nodes", # "nodes" or "all_slots"
        include_root=False
    )
    
    # [핵심] 3. Profiled Scheduler 생성 (make_profiled_sequential_schedule)
    print("Creating ProfiledDepthSchedule...")
    scheduler = make_profiled_sequential_schedule(
        profile,
        smoothing=1.0,   # 빈도수 스무딩
        power=0.7,       # 분포를 약간 평탄화 (너무 쏠리지 않게)
        overlap=0.1      # 윈도우 간 약간의 오버랩 허용 (자연스러운 전환)
    )

    # 4. 시각화할 타겟 트리 선정 (첫 번째 샘플)
    target_tree = dataset[0]
    
    # 5. Spurious(가짜) 노드 슬롯 추가
    # 스케줄러가 '깊이' 기반이므로 Spurious 노드도 깊이에 따라 스케줄링됨
    real_nodes = target_tree.clone()
    num_real = real_nodes.size(0)
    
    num_spurious = 25
    spurious_list = []
    real_indices = np.arange(num_real)
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    for _ in range(num_spurious):
        p_idx = np.random.choice(real_indices)
        p_depth = real_nodes[p_idx, 0].item()
        if p_depth < config['max_depth']:
            # [depth, rank, type=0, parent_idx]
            spurious_list.append([p_depth + 1, 0, 0, p_idx])
            
    if spurious_list:
        spurious_tensor = torch.tensor(spurious_list, dtype=torch.long)
        full_data = torch.cat([real_nodes, spurious_tensor], dim=0)
    else:
        full_data = real_nodes

    N = full_data.size(0)
    depths = full_data[:, 0]
    target_types = full_data[:, 2] # 0: Empty/Spurious, >0: Real
    parents = full_data[:, 3]
    
    # 6. Noise & Layout 설정
    noise_config = NoiseConfig(
        p_blank_when_target_token=0.9, 
        p_blank_when_target_blank=0.95
    )
    
    # Fixed Noise
    u_choose = torch.rand(N)
    u_noise_blank = torch.rand(N)
    u_noise_type = torch.randint(1, config['num_types'] + 1, (N,))
    
    # Layout
    G_layout = nx.Graph()
    for i in range(num_real):
        G_layout.add_node(i)
        p = parents[i].item()
        if p != -1: G_layout.add_edge(p, i)
    pos = hierarchy_pos(G_layout, root=0)
    
    # Spurious Layout
    for i in range(num_real, N):
        p = parents[i].item()
        if p in pos:
            offset = np.random.uniform(-0.15, 0.15, 2)
            pos[i] = (pos[p][0] + offset[0], pos[p][1] - 0.2 + offset[1])
        else:
            pos[i] = (0.5, 0.5)

    # 7. Visualization Loop
    print(f"Generating GIF ({steps} steps)...")
    frames = []
    time_steps = np.linspace(0, 1, steps)
    
    # 스케줄 정보 (시각화 타이틀용)
    starts = scheduler.starts.cpu().numpy()
    widths = scheduler.widths.cpu().numpy()
    
    for t_val in tqdm(time_steps):
        t_tensor = torch.tensor([t_val])
        
        # --- Logic: Profiled Scheduler Kappa ---
        # scheduler.kappa()는 입력 깊이에 따른 kappa 값을 반환
        # 깊이가 얕은 노드부터 깊은 노드 순으로 kappa가 1에 도달함 (데이터 분포에 따라 속도 다름)
        kappa = scheduler.kappa(t_tensor, depths).squeeze() # [N]
        
        # --- Standard Discrete Flow Logic ---
        choose_target = u_choose < kappa
        
        is_target_real = (target_types > 0)
        p_blank = torch.where(is_target_real, 
                              torch.tensor(noise_config.p_blank_when_target_token),
                              torch.tensor(noise_config.p_blank_when_target_blank))
        
        z0_is_blank = u_noise_blank < p_blank
        z0_val = torch.where(z0_is_blank, torch.tensor(0), u_noise_type)
        
        z_t = torch.where(choose_target, target_types, z0_val)
        z_t[0] = target_types[0] # Force Root
        
        # Structural Constraint
        exist = (z_t > 0)
        exist[0] = True
        safe_parents = parents.clone(); safe_parents[safe_parents < 0] = 0
        
        for _ in range(config['max_depth'] + 2):
            parent_exist = exist[safe_parents]
            has_parent = (parents >= 0)
            can_exist = (~has_parent) | parent_exist
            new_exist = exist & can_exist
            if torch.equal(new_exist, exist): break
            exist = new_exist
            
        final_types = torch.where(exist, z_t, torch.zeros_like(z_t))
        
        # --- Plotting ---
        fig, ax = plt.subplots(figsize=(8, 9)) # 세로로 약간 길게
        
        active_indices = torch.where(final_types > 0)[0].tolist()
        draw_nodes = []
        node_colors = []
        edge_list = []
        
        for i in active_indices:
            p = parents[i].item()
            if p != -1 and final_types[p] > 0:
                edge_list.append((p, i))
            
            draw_nodes.append(i)
            current_type = final_types[i].item()
            tgt_type = target_types[i].item()
            
            if i < num_real:
                if current_type == tgt_type:
                    node_colors.append('#2ecc71') # Green (Match)
                else:
                    node_colors.append('#e74c3c') # Red (Mismatch/Noise)
            else:
                node_colors.append('#9b59b6') # Purple (Spurious)

        if draw_nodes:
            valid_draw_nodes = [n for n in draw_nodes if n in pos]
            nx.draw_networkx_edges(G_layout, pos, edgelist=edge_list, ax=ax, edge_color='#bdc3c7', alpha=0.6)
            nx.draw_networkx_nodes(G_layout, pos, nodelist=valid_draw_nodes, node_color=node_colors, node_size=180, ax=ax, edgecolors='white', linewidths=0.8)
            
        # Title with active depth info
        # 현재 시간 t에서 활성화된 깊이 구간 찾기
        active_depths = []
        for d in range(1, len(starts)): # Root 제외
            s, w = starts[d], widths[d]
            if s <= t_val < s + w:
                active_depths.append(d)
        
        depth_info = f"Active Depths: {active_depths}" if active_depths else ""
        ax.set_title(f"Profiled Tree Flow (t={t_val:.2f})\n{depth_info}", fontsize=14)
        
        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ecc71', label='Real (Target)', markersize=10),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', label='Corrupted', markersize=10),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#9b59b6', label='Spurious', markersize=10),
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        ax.axis('off')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=80, bbox_inches='tight')
        buf.seek(0)
        frames.append(imageio.imread(buf))
        plt.close(fig)
        buf.close()

    print(f"Saving GIF to {output_gif}...")
    imageio.mimsave(output_gif, frames, fps=20)
    print("Done!")

if __name__ == "__main__":
    visualize_treedfm_profiled()