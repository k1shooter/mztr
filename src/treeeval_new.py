import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx 
from collections import defaultdict
import argparse

# ==============================================================================
# 1. 모델 클래스 임포트
# ==============================================================================
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

try:
    # dfm.py에서 TreeEditDFM (순수 PyTorch 버전) 로드
    from treedfm.dfm import TreeEditDFM
    print("Successfully imported TreeEditDFM from treedfm.dfm")
except ImportError as e:
    print(f"Error: Could not import treedfm.dfm. {e}")
    exit(1)

# ==========================================
# 2. Maze Grid Logic & Analyzer
# ==========================================
# (기존 로직 유지)
class MazeGridSystem:
    def __init__(self):
        self.directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    
    def get_next_state(self, current_pos, current_dir_idx, node_type):
        r, c = current_pos
        next_dir_idx = current_dir_idx
        if node_type == 1:   # Left
            next_dir_idx = (current_dir_idx - 1) % 4
        elif node_type == 2: # Forward
            next_dir_idx = current_dir_idx
        elif node_type == 3: # Right
            next_dir_idx = (current_dir_idx + 1) % 4
        
        dr, dc = self.directions[next_dir_idx]
        next_pos = (r + dr, c + dc)
        return next_pos, next_dir_idx

class MazeAnalyzer:
    def __init__(self, boundary_radius=20):
        self.grid_system = MazeGridSystem()
        self.boundary_sq = boundary_radius**2
        self.reset()

    def reset(self):
        self.node_states = {} 
        self.occupied_positions = {} 
        self.faults = {'collision': [], 'oob': [], 'duplicate_type': []}
        self.metrics = {}

    def analyze_tree(self, flat_tree):
        self.reset()
        if not flat_tree: return {}, self.faults, {}

        self.node_states[0] = ((0, 0), 1) 
        self.occupied_positions[(0, 0)] = 0
        
        parent_child_types = defaultdict(set)
        max_depth = 0

        for i, node in enumerate(flat_tree):
            if i == 0: continue 
            if isinstance(node, torch.Tensor): node = node.tolist()
            
            try:
                if isinstance(node, dict):
                    depth, rank, n_type, parent_idx = node['depth'], node['rank'], node['type'], node['parent_idx']
                else:
                    depth, rank, n_type, parent_idx = map(int, node)
            except Exception: continue

            max_depth = max(max_depth, depth)
            
            if parent_idx not in self.node_states: continue

            if n_type in parent_child_types[parent_idx]:
                self.faults['duplicate_type'].append(i)
            parent_child_types[parent_idx].add(n_type)

            parent_pos, parent_dir_idx = self.node_states[parent_idx]
            new_pos, new_dir_idx = self.grid_system.get_next_state(parent_pos, parent_dir_idx, n_type)
            self.node_states[i] = (new_pos, new_dir_idx)

            if new_pos[0]**2 + new_pos[1]**2 > self.boundary_sq:
                self.faults['oob'].append(i)
            
            if new_pos in self.occupied_positions:
                self.faults['collision'].append(i)
            else:
                self.occupied_positions[new_pos] = i

        tree_size = len(flat_tree)
        total_faults = len(self.faults['collision']) + len(self.faults['oob']) + len(self.faults['duplicate_type'])
        
        non_leaf_nodes = len(parent_child_types)
        avg_branching = (tree_size - 1) / non_leaf_nodes if non_leaf_nodes > 0 else 0

        self.metrics = {
            'tree_size': tree_size,
            'max_depth': max_depth,
            'total_faults': total_faults,
            'validity_rate': (tree_size - total_faults) / tree_size if tree_size > 0 else 0,
            'avg_branching_factor': avg_branching
        }
        return self.metrics, self.faults, self.node_states

# ==========================================
# 3. Visualization
# ==========================================
class MazeVisualizer:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def plot_maze(self, flat_tree, node_states, faults, metrics, filename):
        positions = [state[0] for state in node_states.values()]
        if not positions: return

        rows, cols = zip(*positions)
        min_r, max_r = min(rows), max(rows)
        min_c, max_c = min(cols), max(cols)
        
        margin = 3
        height, width = (max_r - min_r + 1 + margin * 2), (max_c - min_c + 1 + margin * 2)
        grid_h, grid_w = int(height * 2 + 1), int(width * 2 + 1)
        maze_grid = np.zeros((grid_h, grid_w, 3))

        def to_grid(r, c):
            return int((r - min_r + margin) * 2 + 1), int((c - min_c + margin) * 2 + 1)

        collision_set = set(faults['collision'])
        oob_set = set(faults['oob'])
        dup_set = set(faults['duplicate_type'])

        for i, node in enumerate(flat_tree):
            if i not in node_states: continue
            curr_pos = node_states[i][0]
            curr_gr, curr_gc = to_grid(*curr_pos)
            
            if isinstance(node, torch.Tensor): node = node.tolist()
            if isinstance(node, dict): parent_idx = node['parent_idx']
            else: parent_idx = int(node[3])

            color = [1.0, 1.0, 1.0]
            if i == 0: color = [0.0, 1.0, 0.0]
            elif i in collision_set: color = [1.0, 0.0, 0.0]
            elif i in oob_set: color = [1.0, 0.6, 0.0]
            elif i in dup_set: color = [0.0, 1.0, 1.0]

            maze_grid[curr_gr, curr_gc] = color

            if parent_idx != -1 and parent_idx in node_states:
                parent_pos = node_states[parent_idx][0]
                parent_gr, parent_gc = to_grid(*parent_pos)
                maze_grid[(curr_gr + parent_gr) // 2, (curr_gc + parent_gc) // 2] = color

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(maze_grid)
        ax.axis('off')
        
        center_gr, center_gc = to_grid(0, 0)
        boundary_radius_px = int(np.sqrt(MazeAnalyzer().boundary_sq) * 2)
        ax.add_patch(patches.Circle((center_gc, center_gr), boundary_radius_px, fill=False, color='yellow', linestyle='--', alpha=0.5))

        metrics_str = "\n".join([f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items()])
        ax.text(0.98, 0.98, metrics_str, transform=ax.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), color='black')
        
        plt.title(f"Maze Map (Size: {metrics['tree_size']})")
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename), dpi=150)
        plt.close()

    def plot_tree_structure(self, flat_tree, faults, filename):
        G = nx.Graph()
        for i, node in enumerate(flat_tree):
            if isinstance(node, torch.Tensor): node = node.tolist()
            if isinstance(node, dict): n_type, parent_idx = node['type'], node['parent_idx']
            else: n_type, parent_idx = int(node[2]), int(node[3])
            G.add_node(i, type=n_type)
            if parent_idx != -1: G.add_edge(parent_idx, i)

        try: pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
        except: pos = nx.kamada_kawai_layout(G)

        colors = ['lime' if n==0 else 'red' if n in faults['collision'] else {1:'#ffcc00', 2:'white', 3:'cyan'}.get(G.nodes[n]['type'], 'grey') for n in G.nodes()]
        
        plt.figure(figsize=(12, 8))
        plt.gca().set_facecolor('#222222')
        plt.gcf().set_facecolor('#222222')
        nx.draw(G, pos, node_color=colors, with_labels=False, node_size=100, edge_color='gray', width=0.5)
        plt.title(f"Abstract Tree Structure (Size: {len(flat_tree)})", color='white')
        plt.savefig(os.path.join(self.save_dir, filename), facecolor='#222222')
        plt.close()

# ==========================================
# 4. Main Pipeline (Config Auto-Detect)
# ==========================================

def main():
    # 경로 설정
    checkpoint_path = "/home/dmsdmswns/mazetree/runs_1010_edit_profiled_bfs/last.pt"
    output_dir = "generation_results_edit1010"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint from: {checkpoint_path}")

    try:
        # 1. 체크포인트 파일 자체를 로드
        ckpt = torch.load(checkpoint_path, map_location=device)
        
        # 2. 학습 당시 설정(args) 복원 시도
        if 'args' in ckpt:
            print("[Info] Found training arguments in checkpoint! Using them to initialize model.")
            train_args = ckpt['args']
            
            # args가 Namespace 객체일 수도 있고 dict일 수도 있음
            if isinstance(train_args, argparse.Namespace):
                train_args = vars(train_args)
            
            # 필요한 파라미터 추출 (treedfm_edit.py의 main 함수 참조)
            # 기본값 설정 (fallback)
            config = {
                'num_types': train_args.get('num_types', 3),
                'k': train_args.get('k', 3),
                'max_depth': train_args.get('max_depth', 100),
                'max_nodes': train_args.get('max_nodes', 256),
                'd_model': train_args.get('d_model', 384),
                'n_heads': train_args.get('n_heads', 8),
                'n_layers': train_args.get('n_layers', 8),
                'dropout': train_args.get('dropout', 0.1),
                'schedule_width': train_args.get('schedule_width', 0.5),
                'schedule_max_psi': train_args.get('schedule_max_psi', 200.0),
                'permutation_invariant': not train_args.get('no_perm_invariant', False)
            }
            
            print(f"[Config] Detected: d_model={config['d_model']}, num_types={config['num_types']}, max_depth={config['max_depth']}")
            
        else:
            print("[Warning] No 'args' found in checkpoint. Using ERROR-BASED FALLBACK defaults.")
            # 에러 로그 기반 추정치 (이전 에러 메시지 참고)
            config = {
                'num_types': 5,      # shape[6, 256] -> 5 types
                'k': 3,              # shape[4, 256] -> k=3
                'max_depth': 100,    # shape[102, 256] -> depth=100
                'max_nodes': 256,
                'd_model': 256,      # shape[..., 256] -> d_model=256
                'n_heads': 8,
                'n_layers': 8,
                'permutation_invariant': False # 유저 요청 사항 반영
            }
            print(f"[Config] Fallback: d_model={config['d_model']}, num_types={config['num_types']}, max_depth={config['max_depth']}")

        # 3. 모델 초기화 (treedfm.dfm.TreeEditDFM 사용)
        model = TreeEditDFM(**config)

        # 4. State Dict 추출 및 키 매핑 (Lit -> Raw)
        if 'model_state' in ckpt:
            state_dict = ckpt['model_state']
        elif 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt

        new_state_dict = {}
        for k, v in state_dict.items():
            # 'model.' 접두사가 있으면 제거하고 'net.'으로 변경 (lit.py의 self.model -> dfm.py의 self.net)
            if k.startswith("model."):
                new_key = k.replace("model.", "net.", 1)
                new_state_dict[new_key] = v
            # 만약 이미 net.으로 시작하거나 접두사가 없다면 상황에 맞춰 처리
            elif not k.startswith("net.") and "encoder" in k: # 순수 모델 가중치일 경우
                 new_state_dict[f"net.{k}"] = v
            else:
                new_state_dict[k] = v
        
        # 5. 가중치 로드
        msg = model.load_state_dict(new_state_dict, strict=False)
        print(f"Load status: {msg}")

    except Exception as e:
        print(f"Fatal Error during loading: {e}")
        import traceback
        traceback.print_exc()
        return

    # 모델 설정
    model.to(device)
    model.eval()
    for p in model.parameters(): p.requires_grad = False

    print(f"Generating trees...")
    
    # 6. 샘플링
    with torch.no_grad():
        # sample 함수 시그니처 확인 (dfm.py)
        # def sample(self, num_samples, steps, max_nodes=None, temperature=1.0)
        generated_trees = model.sample(
            num_samples=10, 
            steps=500, 
            max_nodes=config.get('max_nodes', 256), 
            temperature=1.0
        )

    # 7. 분석 및 시각화
    analyzer = MazeAnalyzer(boundary_radius=30)
    visualizer = MazeVisualizer(save_dir=output_dir)
    
    print("\n--- Analysis & Visualization ---")
    for i, tree in enumerate(generated_trees):
        if len(tree) <= 1:
            print(f"Tree {i}: Failed (Only Root)")
            continue
            
        metrics, faults, node_states = analyzer.analyze_tree(tree)
        print(f"Tree {i}: Size={metrics['tree_size']}, Faults={metrics['total_faults']}, Validity={metrics['validity_rate']:.2f}")
        
        visualizer.plot_maze(tree, node_states, faults, metrics, filename=f"maze_{i}.png")
        visualizer.plot_tree_structure(tree, faults, filename=f"tree_{i}.png")
        
    print(f"\nResults saved to directory: {output_dir}")

if __name__ == "__main__":
    main()