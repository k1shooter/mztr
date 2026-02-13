import os
import pickle
import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
import imageio
from tqdm import tqdm
import io
import math 

# ==========================================
# 1. Scheduler (Treedfm_vec.py V8 Logic)
# ==========================================
class CascadingWaveScheduler:
    def __init__(self, max_depth):
        # self.max_depth = max_depth
        # self.delta = 1.0 / (max_depth + 1)

        # self.max_depth = max_depth
        # self.window = 0.5 
        # if max_depth > 0:
        #     self.step = (1.0 - self.window) / max_depth
        # else:
        #     self.step = 0

        self.max_depth = max_depth
        # Each layer takes time width to fully form
        self.width = 0.3 
        # Start times are staggered: t_start(d) = d * step
        # We want t_start(max) + width = 1.0
        # => max * step + width = 1.0
        # => step = (1.0 - width) / max
        if max_depth > 0:
            self.step = (1.0 - self.width) / max_depth
        else:
            self.step = 0

    def get_schedules(self, t, depths):

        # if isinstance(t, float):
        #     t = torch.tensor(t)

        # if t.dim() == 1:
        #     t = t.view(-1, 1)

        # t = t.to(depths.device)
        # d = depths.float()

        # tau = d * self.delta

        # prog = (t - tau) / self.delta
        # alpha = torch.clamp(prog, 0.0, 1.0)

        # d_alpha = torch.where(
        #     (t >= tau) & (t <= tau + self.delta),
        #     torch.full_like(t, 1.0 / self.delta),
        #     torch.zeros_like(t)
        # )

        # # Substitution: only active while layer growing
        # beta = 1.0 - alpha
        # d_beta = -d_alpha

        # # Spurious noise only during active growth window
        # gamma = 0.2 * alpha * (1 - alpha)
        # d_gamma = 0.2 * (1 - 2*alpha) * d_alpha

        # if isinstance(t, float): t = torch.tensor(t)
        # if t.dim() == 1: t = t.view(-1, 1)
        # t = t.to(depths.device); d = depths.float(); t = t.expand_as(d)
        
        # # 1. Insertion (Alpha): Pipelined Ramp
        # t_start = d * self.step
        # prog = (t - t_start) / self.window
        # alpha = torch.clamp(prog, 0.0, 1.0)
        
        # d_alpha = torch.where((prog > 0) & (prog < 1), 
        #                       torch.tensor(1.0/self.window, device=t.device),
        #                       torch.tensor(0.0, device=t.device))
        
        # # 2. Substitution (Beta): Linear Decay 1 -> 0
        # beta = 1.0 - alpha
        # d_beta = -d_alpha
        
        # # 3. Deletion (Gamma): Peaks at 0.5
        # gamma = 0.2 * 4.0 * alpha * (1.0 - alpha)
        # d_gamma = 0.8 * (1.0 - 2.0 * alpha) * d_alpha

        if isinstance(t, float): t = torch.tensor(t)
        if t.dim() == 1: t = t.view(-1, 1)
        t = t.to(depths.device); d = depths.float(); t = t.expand_as(d)
        
        # Linear Ramp: Alpha goes 0 -> 1 over [t_start, t_end]
        t_start = d * self.step
        t_end = t_start + self.width
        
        # Calculate progress in [0, 1]
        prog = (t - t_start) / self.width
        alpha = torch.clamp(prog, 0.0, 1.0)
        
        # Constant Velocity d_alpha/dt = 1 / width inside window
        d_alpha = torch.where((t >= t_start) & (t <= t_end),
                              torch.tensor(1.0/self.width, device=t.device),
                              torch.tensor(0.0, device=t.device))
        
        beta = 1.0 - alpha
        gamma = 0.2 * (1.0 - alpha)
        
        # Hazard Rate: d_alpha / (1 - alpha)
        # With linear alpha, 1-alpha = 1 - (t-start)/width = (end-t)/width
        # Rate = (1/width) / ((end-t)/width) = 1 / (end - t)
        # This explodes at t_end. We clamp it.
        
        denom = torch.clamp(1.0 - alpha, min=1e-4)
        hazard_rate = d_alpha / denom
        
        # Soft clamp to allow strong cleanup at the end of window, but prevent NaN
        hazard_rate = torch.clamp(hazard_rate, 0.0, 20.0)
        return alpha, beta, gamma
# ==========================================
# 2. Tree Utils & Layout
# ==========================================
class TreeUtils:
    @staticmethod
    def flatten_tree(root, max_depth, k):
        flat_nodes = []
        queue = [(root, (), -1)] 
        while queue:
            node, path, parent_idx = queue.pop(0)
            if len(path) > max_depth: continue
            node_type = node.get('type', 1) 
            depth = len(path); rank = path[-1] if depth > 0 else 0
            flat_nodes.append({'depth': depth, 'rank': rank, 'type': node_type, 'parent_idx': parent_idx, 'path': path})
            children = node.get('children', [])
            children.sort(key=lambda x: x.get('id', 0))
            for i, child in enumerate(children):
                if i >= k: break 
                queue.append((child, path + (i,), len(flat_nodes)-1))
        return flat_nodes

def hierarchy_pos(G, root=None, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
    """
    NetworkX용 트리 예쁘게 그리는 레이아웃 함수
    """
    if not nx.is_tree(G):
        return nx.spring_layout(G) # Fallback

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
# 3. Visualization Logic
# ==========================================
def visualize_flow(pkl_path, output_gif="tree_flow.gif", steps=500, max_depth=16, k=3):
    print(f"Loading data from {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        raw_data = pickle.load(f)
    
    # 데이터 하나 픽 (가장 복잡한 트리 중 하나 선택)
    keys = list(raw_data.keys())
    sample_key = keys[0] # 첫 번째 데이터 사용
    flat_nodes = TreeUtils.flatten_tree(raw_data[sample_key], max_depth, k)
    
    # 텐서 변환 (Batch=1)
    # [depth, rank, type, parent_idx]
    tensor_data = torch.tensor([[n['depth'], n['rank'], n['type'], n['parent_idx']] for n in flat_nodes], dtype=torch.long)
    N = tensor_data.size(0)
    
    depths = tensor_data[:, 0].unsqueeze(0) # [1, N]
    parents = tensor_data[:, 3].unsqueeze(0) # [1, N]
    x_1_types = tensor_data[:, 2].unsqueeze(0) # [1, N]
    
    scheduler = CascadingWaveScheduler(max_depth=max_depth)
    
    # --- Fix Noise for Smooth Animation ---
    # 노이즈를 미리 고정해야 t가 변할 때 깜빡거리지 않음
    u_ins = torch.rand(1, N)
    u_sub = torch.rand(1, N)
    u_spur = torch.rand(1, N)
    
    # Random types for corruption
    # (실제론 num_types=4 등으로 가정)
    num_types = 4
    random_types = torch.randint(1, num_types + 1, (1, N))
    
    print(f"Generating GIF with {steps} steps...")
    frames = []
    
    # 전체 구조 그래프 미리 빌드 (위치 고정을 위해)
    G_full = nx.Graph()
    for i in range(N):
        G_full.add_node(i)
        p = parents[0, i].item()
        if p != -1:
            G_full.add_edge(p, i)
            
    # 레이아웃 고정
    pos = hierarchy_pos(G_full, root=0)
    
    # Time Loop
    time_steps = np.linspace(0, 1, steps)
    
    for t_val in tqdm(time_steps):
        t_tensor = torch.tensor([t_val]).float()
        
        # 1. Get Schedule Values
        alpha, beta, gamma = scheduler.get_schedules(t_tensor, depths)
        
        # 2. Logic (construct_xt logic copied)
        
        # A. Insertion (Masking)
        # alpha가 u_ins보다 크면 "보임(Inserted)"
        is_inserted = u_ins < alpha
        is_inserted[:, 0] = True # Root always
        
        # Structural Connectivity Check
        # 부모가 안 보이면 자식도 안 보여야 함
        safe_parents = parents.clone(); safe_parents[safe_parents < 0] = 0
        current_mask = is_inserted.clone()
        
        for _ in range(max_depth):
            parent_inserted = current_mask.gather(1, safe_parents)
            has_parent = parents >= 0
            condition = (~has_parent) | parent_inserted
            new_mask = current_mask & condition
            if torch.equal(new_mask, current_mask): break
            current_mask = new_mask
            
        is_visible = current_mask
        
        # B. Substitution (Corruption)
        # Real Node인데 beta가 u_sub보다 크면 "Corrupted"
        should_corrupt = (u_sub < beta) & is_visible
        
        # C. Deletion (Spurious)
        # Empty Slot인데 gamma가 u_spur보다 크면 "Spurious"
        # Spurious 조건: 현재 안 보임(Empty) + 부모는 보임
        parent_visible = is_visible.gather(1, safe_parents)
        is_empty_slot = (~is_visible) & parent_visible & (parents >= 0)
        should_spawn_spur = (u_spur < gamma) & is_empty_slot
        
        # 3. Plotting
        plt.figure(figsize=(8, 6))
        ax = plt.gca()
        
        # Draw Edges (Only if both nodes are 'Active' in some form)
        # Active = Visible OR Spurious
        is_active = is_visible | should_spawn_spur
        
        active_indices = torch.where(is_active[0])[0].tolist()
        
        # Draw Nodes
        node_colors = []
        draw_nodes = []
        
        for i in active_indices:
            # 부모가 active해야 연결선 그림
            p = parents[0, i].item()
            if p != -1 and is_active[0, p].item():
                nx.draw_networkx_edges(G_full, pos, edgelist=[(p, i)], ax=ax, edge_color='gray', alpha=0.5)
            
            draw_nodes.append(i)
            
            # Color Logic
            if should_spawn_spur[0, i]:
                # Spurious (Should be deleted) -> Purple
                node_colors.append('#9b59b6') 
            elif should_corrupt[0, i]:
                # Corrupted (Should be fixed) -> Red
                node_colors.append('#e74c3c')
            else:
                # Correct (Real) -> Green (or based on Type)
                # 실제 타입에 따라 색상 다르게 (파랑, 초록 등)
                real_type = x_1_types[0, i].item()
                type_colors = ['#ecf0f1', '#3498db', '#2ecc71', '#f1c40f', '#e67e22'] # 0, 1, 2, 3, 4
                node_colors.append(type_colors[real_type % len(type_colors)])

        if draw_nodes:
            nx.draw_networkx_nodes(G_full, pos, nodelist=draw_nodes, node_color=node_colors, node_size=200, ax=ax)
            # nx.draw_networkx_labels(G_full, pos, labels={n:str(n) for n in draw_nodes}, font_size=8)

        plt.title(f"Discrete Flow Matching Process (t={t_val:.2f})")
        
        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ecc71', label='Real (Correct)', markersize=10),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', label='Corrupted (Sub)', markersize=10),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#9b59b6', label='Spurious (Del)', markersize=10),
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.axis('off')
        
        # Save frame to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        frames.append(imageio.imread(buf))
        plt.close()
        buf.close()

    print(f"Saving GIF to {output_gif}...")
    imageio.mimsave(output_gif, frames, fps=30)
    print("Done!")

if __name__ == "__main__":
    # 실행 설정
    pkl_file = "maze_trees_relative_44.pkl" # 데이터셋 경로 확인 필요
    
    if os.path.exists(pkl_file):
        visualize_flow(pkl_file, "tree_flow_process.gif")
    else:
        print(f"Error: {pkl_file} not found. Please check the path.")