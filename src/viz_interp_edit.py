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
from treedfm.schedule import DepthStratifiedSchedule
from treedfm.corruption import NoiseConfig
from treedfm.data import MazeTreeDataset

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
# 3. Visualization Logic
# ==========================================
def visualize_treedfm_edit_real(pkl_filename="maze_trees_relative_44.pkl", output_gif="treedfm_edit_real.gif", steps=100):
    print(f"Initializing TreeEditDFM Visualization for {pkl_filename}...")
    
    # 4x4 미로 데이터셋에 맞춘 Config 설정
    # 4x4 미로의 최대 경로 길이는 보통 16 이하
    config = {
        'max_depth': 20,      # 넉넉하게 20 설정
        'k': 3,               # Left, Forward, Right (3분기)
        'num_types': 3,       # 1:L, 2:F, 3:R
        'schedule_width': 0.5 # 스케줄 너비
    }
    
    # 1. 데이터 로드 (MazeTreeDataset 활용)
    # pkl 경로가 현재 폴더 혹은 상위 폴더에 있다고 가정
    if os.path.exists(pkl_filename):
        pkl_path = pkl_filename
    elif os.path.exists(os.path.join("..", pkl_filename)):
        pkl_path = os.path.join("..", pkl_filename)
    else:
        # 파일이 없으면 더미 데이터 생성 대신 에러 메시지 출력 (User 요청: "데이터를 뽑아온다고 가정")
        print(f"Warning: {pkl_filename} not found. Using a dummy tree for demonstration.")
        pkl_path = None

    if pkl_path:
        ds = MazeTreeDataset(pkl_path, max_depth=config['max_depth'], k=config['k'])
        # 첫 번째 샘플 가져오기 (가장 복잡한 걸 원하면 길이가 긴 것을 선택 가능)
        # 여기선 편의상 0번 인덱스 사용
        # tensor shape: [N, 4] -> (depth, rank, type, parent_idx)
        tree_tensor = ds[0]
    else:
        # Fallback: Dummy Data
        tree_tensor = torch.tensor([[0,0,2,-1], [1,0,2,0], [2,0,1,1], [2,1,3,1]], dtype=torch.long)

    # 2. Spurious(가짜) 노드 슬롯 추가
    # 실제 데이터는 '정답 트리'만 가지고 있음.
    # Flow Matching 과정에서는 빈 공간(Padding)에서 가짜 노드들이 생겨났다 사라지는 과정을 보여줘야 함.
    # 따라서, 실제 트리의 노드 외에 '잠재적 빈 공간'을 임의로 추가하여 시각화 데이터 구성.
    
    real_nodes = tree_tensor.clone()
    num_real = real_nodes.size(0)
    
    # Spurious 노드를 위한 빈 슬롯 생성 (실제 부모 노드들 밑에 임의로 부착)
    num_spurious = 20 # 시각화용 가짜 슬롯 개수
    spurious_list = []
    
    # 실제 존재하는 부모 인덱스들
    real_indices = torch.arange(num_real)
    
    # 랜덤 시드 고정 (재현성)
    torch.manual_seed(42)
    np.random.seed(42)
    
    for _ in range(num_spurious):
        # 임의의 부모 선택
        p_idx = np.random.choice(real_indices.numpy())
        p_depth = real_nodes[p_idx, 0].item()
        
        # 실제 데이터와 겹치지 않게 하기 위해 k값이나 로직을 정교하게 해야 하지만,
        # 시각화 목적상 단순히 type=0 (Empty Target)인 노드를 추가
        # [depth, rank, type=0, parent_idx]
        spurious_list.append([p_depth + 1, 0, 0, p_idx])
        
    spurious_tensor = torch.tensor(spurious_list, dtype=torch.long)
    
    # 전체 시각화용 데이터: [Real Nodes ... | Spurious Slots ...]
    full_data = torch.cat([real_nodes, spurious_tensor], dim=0)
    N = full_data.size(0)
    
    depths = full_data[:, 0]
    target_types = full_data[:, 2] # 0이면 Empty(Spurious Target), >0이면 Real
    parents = full_data[:, 3]
    
    # 3. 스케줄러 & 노이즈 설정
    scheduler = DepthStratifiedSchedule(max_depth=config['max_depth'], width=config['schedule_width'])
    noise_config = NoiseConfig(
        p_blank_when_target_token=0.9, # Target이 있을 때 Noise가 Blank일 확률 (생성 효과)
        p_blank_when_target_blank=0.95 # Target이 없을 때 Noise가 Blank일 확률 (Spurious 억제)
    )
    
    # 4. 노이즈 고정 (Smooth Animation)
    # 시간 t에 따라 확률적으로 변하는 게 아니라, 미리 뽑아둔 노이즈 $u$와 임계값 $kappa(t)$를 비교
    u_choose = torch.rand(N) 
    
    # Noise State($z_0$) 결정 (시간과 무관한 기저 노이즈)
    u_noise_blank = torch.rand(N)
    u_noise_type = torch.randint(1, config['num_types'] + 1, (N,))
    
    # 5. 그래프 레이아웃 고정
    # Real Node 기준으로 레이아웃을 잡고, Spurious는 근처에 배치
    G_layout = nx.Graph()
    for i in range(num_real):
        G_layout.add_node(i)
        p = parents[i].item()
        if p != -1: G_layout.add_edge(p, i)
    
    pos = hierarchy_pos(G_layout, root=0)
    
    # Spurious 노드 위치 할당 (부모 근처 랜덤 오프셋)
    for i in range(num_real, N):
        p = parents[i].item()
        if p in pos:
            offset = np.random.uniform(-0.15, 0.15, 2)
            # y축은 아래로 내려가야 하므로 -0.1 ~ -0.3
            pos[i] = (pos[p][0] + offset[0], pos[p][1] - 0.2 + offset[1])
        else:
            pos[i] = (0.5, 0.5)

    print(f"Generating GIF frames ({steps} steps)...")
    frames = []
    time_steps = np.linspace(0, 1, steps)
    
    for t_val in tqdm(time_steps):
        t_tensor = torch.tensor([t_val])
        
        # --- [Logic] corrupt_batch_tree 로직 재구성 ---
        
        # 1. Kappa 계산 (Schedule)
        kappa = scheduler.kappa(t_tensor, depths).squeeze() # [N]
        
        # 2. 상태 결정 ($z_t$)
        # Target을 선택할 것인가? (Interpolation)
        choose_target = u_choose < kappa
        
        # 3. Noise State ($z_0$) 정의
        # Target이 Real(>0)인 경우와 Empty(0)인 경우 p_blank 확률이 다름
        is_target_real = (target_types > 0)
        
        p_blank = torch.where(is_target_real, 
                              torch.tensor(noise_config.p_blank_when_target_token),
                              torch.tensor(noise_config.p_blank_when_target_blank))
        
        z0_is_blank = u_noise_blank < p_blank
        z0_val = torch.where(z0_is_blank, torch.tensor(0), u_noise_type)
        
        # 4. $z_t$ 합성 (Target vs Noise)
        z_t = torch.where(choose_target, target_types, z0_val)
        
        # Root는 항상 존재 (옵션)
        z_t[0] = target_types[0]
        if z_t[0] == 0: z_t[0] = 2 # 혹시 모를 안전장치
        
        # 5. 구조적 일관성 (Structural Integrity)
        # corruption.py의 반복적 마스킹 로직 적용: "부모가 없으면 자식도 사라짐"
        exist = (z_t > 0)
        exist[0] = True
        
        # 인덱싱을 위한 안전한 부모 배열
        safe_parents = parents.clone()
        safe_parents[safe_parents < 0] = 0
        
        for _ in range(config['max_depth'] + 2):
            parent_exist = exist[safe_parents]
            has_parent_idx = (parents >= 0)
            # (부모가 없거나 Root) OR (부모가 살아있음)
            can_exist = (~has_parent_idx) | parent_exist
            new_exist = exist & can_exist
            if torch.equal(new_exist, exist): break
            exist = new_exist
            
        final_types = torch.where(exist, z_t, torch.zeros_like(z_t))
        
        # --- [Plotting] ---
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # 그릴 노드 선별 (Type > 0 인 것만)
        active_indices = torch.where(final_types > 0)[0].tolist()
        
        draw_nodes = []
        node_colors = []
        edge_list = []
        
        for i in active_indices:
            p = parents[i].item()
            # 엣지: 부모가 있고 부모도 활성이면 그리기
            if p != -1 and final_types[p] > 0:
                edge_list.append((p, i))
            
            draw_nodes.append(i)
            
            # 색상 로직 (상태 판별)
            current_type = final_types[i].item()
            tgt_type = target_types[i].item()
            
            if i < num_real:
                # Real Node Slot
                if current_type == tgt_type:
                    # Target과 일치 -> 정상 생성됨 (Green)
                    node_colors.append('#2ecc71') 
                else:
                    # Target과 불일치 -> Noise 상태인데 Random Type이 뜬 경우 (Red/Orange)
                    node_colors.append('#e74c3c') 
            else:
                # Spurious Slot (Target은 0이어야 함)
                # 현재 값이 존재하므로 Spurious (Purple)
                node_colors.append('#9b59b6') 

        # Draw
        if draw_nodes:
            # Layout에서 현재 활성 노드의 위치만 가져옴
            active_pos = {n: pos[n] for n in draw_nodes if n in pos}
            # 혹시 pos에 없는 노드가 있다면 제외 (Spurious 등)
            valid_draw_nodes = [n for n in draw_nodes if n in pos]
            
            nx.draw_networkx_edges(G_layout, pos, edgelist=edge_list, ax=ax, edge_color='#bdc3c7', width=1.5, alpha=0.7)
            nx.draw_networkx_nodes(G_layout, pos, nodelist=valid_draw_nodes, node_color=node_colors, node_size=200, ax=ax, edgecolors='white', linewidths=1.0)
        
        ax.set_title(f"TreeEdit Interpolant Process (t={t_val:.2f})", fontsize=14)
        
        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ecc71', label='Real (Target Match)', markersize=10),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', label='Corrupted (Mismatch)', markersize=10),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#9b59b6', label='Spurious (To Delete)', markersize=10),
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        ax.axis('off')
        
        # Save frame
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
    visualize_treedfm_edit_real()