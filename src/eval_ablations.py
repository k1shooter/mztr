import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx 
from collections import defaultdict
import argparse
import pandas as pd
import glob

# treedfm 모듈 임포트를 위한 경로 설정
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

try:
    from treedfm.data import MazeTreeDataset
except ImportError as e:
    print(f"Error: Could not import treedfm modules. {e}")
    sys.exit(1)

# ==========================================
# 1. Maze Logic & Analyzer
# ==========================================
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
        return (r + dr, c + dc), next_dir_idx

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
        if not flat_tree or len(flat_tree) <= 1: 
            return {'tree_size': 1, 'n_collisions': 0, 'n_duplicate_types': 0, 'n_oob': 0, 'total_faults': 0, 'validity_rate': 0}, self.faults, {}

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

            # Implicit Rule 1: 중복 자식 검사
            if n_type in parent_child_types[parent_idx]:
                self.faults['duplicate_type'].append(i)
            parent_child_types[parent_idx].add(n_type)

            parent_pos, parent_dir_idx = self.node_states[parent_idx]
            new_pos, new_dir_idx = self.grid_system.get_next_state(parent_pos, parent_dir_idx, n_type)
            self.node_states[i] = (new_pos, new_dir_idx)

            # Implicit Rule 2: 경계 이탈(OOB) 검사
            if new_pos[0]**2 + new_pos[1]**2 > self.boundary_sq:
                self.faults['oob'].append(i)
            
            # Implicit Rule 3: 브랜치 충돌 검사
            if new_pos in self.occupied_positions:
                self.faults['collision'].append(i)
            else:
                self.occupied_positions[new_pos] = i

        tree_size = len(flat_tree)
        n_coll = len(self.faults['collision'])
        n_dup = len(self.faults['duplicate_type'])
        n_oob = len(self.faults['oob'])
        total_faults = n_coll + n_dup + n_oob
        
        self.metrics = {
            'tree_size': tree_size,
            'max_depth': max_depth,
            'n_collisions': n_coll,
            'n_duplicate_types': n_dup,
            'n_oob': n_oob,
            'total_faults': total_faults,
            'validity_rate': max(0.0, (tree_size - total_faults) / tree_size) if tree_size > 0 else 0
        }
        return self.metrics, self.faults, self.node_states

# ==========================================
# 2. Visualization
# ==========================================
class MazeVisualizer:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def plot_both(self, flat_tree, node_states, faults, metrics, base_filename):
        self._plot_maze(flat_tree, node_states, faults, metrics, f"maze_{base_filename}.png")
        self._plot_tree(flat_tree, faults, f"tree_{base_filename}.png")

    def _plot_maze(self, flat_tree, node_states, faults, metrics, filename):
        positions = [state[0] for state in node_states.values()]
        if not positions: return

        rows, cols = zip(*positions)
        min_r, max_r = min(rows), max(rows)
        min_c, max_c = min(cols), max(cols)
        
        margin = 3
        height, width = (max_r - min_r + 1 + margin * 2), (max_c - min_c + 1 + margin * 2)
        grid_h, grid_w = int(height * 2 + 1), int(width * 2 + 1)
        maze_grid = np.zeros((grid_h, grid_w, 3))

        def to_grid(r, c): return int((r - min_r + margin) * 2 + 1), int((c - min_c + margin) * 2 + 1)

        col_set, oob_set, dup_set = set(faults['collision']), set(faults['oob']), set(faults['duplicate_type'])

        for i, node in enumerate(flat_tree):
            if i not in node_states: continue
            curr_pos = node_states[i][0]
            curr_gr, curr_gc = to_grid(*curr_pos)
            
            if isinstance(node, torch.Tensor): node = node.tolist()
            parent_idx = int(node[3]) if not isinstance(node, dict) else node['parent_idx']

            color = [1.0, 1.0, 1.0]
            if i == 0: color = [0.0, 1.0, 0.0]
            elif i in col_set: color = [1.0, 0.0, 0.0]
            elif i in oob_set: color = [1.0, 0.6, 0.0]
            elif i in dup_set: color = [0.0, 1.0, 1.0]

            maze_grid[curr_gr, curr_gc] = color
            if parent_idx != -1 and parent_idx in node_states:
                parent_gr, parent_gc = to_grid(*node_states[parent_idx][0])
                maze_grid[(curr_gr + parent_gr) // 2, (curr_gc + parent_gc) // 2] = color

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(maze_grid)
        ax.axis('off')
        
        metrics_str = f"Size: {metrics['tree_size']} | Faults: {metrics['total_faults']}\nVal: {metrics['validity_rate']:.2f}"
        ax.text(0.98, 0.98, metrics_str, transform=ax.transAxes, verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_tree(self, flat_tree, faults, filename):
        G = nx.Graph()
        for i, node in enumerate(flat_tree):
            if isinstance(node, torch.Tensor): node = node.tolist()
            n_type = int(node[2]) if not isinstance(node, dict) else node['type']
            parent_idx = int(node[3]) if not isinstance(node, dict) else node['parent_idx']
            G.add_node(i, type=n_type)
            if parent_idx != -1: G.add_edge(parent_idx, i)

        try: pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
        except: pos = nx.kamada_kawai_layout(G)

        colors = ['lime' if n==0 else 'red' if n in faults['collision'] else {1:'#ffcc00', 2:'white', 3:'cyan'}.get(G.nodes[n]['type'], 'grey') for n in G.nodes()]
        plt.figure(figsize=(10, 6))
        plt.gca().set_facecolor('#222222'); plt.gcf().set_facecolor('#222222')
        nx.draw(G, pos, node_color=colors, node_size=80, edge_color='gray', width=0.5)
        plt.savefig(os.path.join(self.save_dir, filename), facecolor='#222222', bbox_inches='tight')
        plt.close()


# ==========================================
# 3. Ablation-Aware Dynamic Loader
# ==========================================
def load_ablation_model(ckpt_path, ablation_name, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    
    # 1. 학습 당시 args 복원
    train_args = ckpt.get('args', {})
    if isinstance(train_args, argparse.Namespace): 
        train_args = vars(train_args)

    config = {
        'num_types': train_args.get('num_types', 3),
        'k': train_args.get('k', 3),
        'max_depth': train_args.get('max_depth', 16),
        'max_nodes': train_args.get('max_nodes', 128),
        'd_model': train_args.get('d_model', 384),
        'n_heads': train_args.get('n_heads', 8),
        'n_layers': train_args.get('n_layers', 8),
        'permutation_invariant': not train_args.get('no_perm_invariant', False)
    }

    # 2. Ablation에 따른 명시적 모델 맵핑
    if "insert_only" in ablation_name:
        from treedfm.dfm import TreeInsertOnlyDFM
        print(f"  -> [Ablation 1] Assigned TreeInsertOnlyDFM")
        model = TreeInsertOnlyDFM(**config)

    elif "depth_agnostic" in ablation_name:
        from treedfm.dfm import TreeEditDFM
        from treedfm.schedule import TimeOnlySchedule
        print(f"  -> [Ablation 2] Assigned TreeEditDFM with TimeOnlySchedule")
        
        # Depth Agnostic을 위한 TimeOnlySchedule 생성
        scheduler = TimeOnlySchedule(
            max_depth=config['max_depth'], 
            width=train_args.get('schedule_width', 0.5), 
            max_psi=train_args.get('schedule_max_psi', 200.0)
        )
        model = TreeEditDFM(**config, scheduler=scheduler)

    elif "baseline_edit" in ablation_name:
        from treedfm.dfm import TreeEditDFM
        print(f"  -> [Baseline] Assigned TreeEditDFM (Default/Profiled Schedule)")
        # baseline은 treeeval_new.py 의 방식 그대로 처리
        model = TreeEditDFM(**config)
        
    else:
        # 그 외의 경우 기본 EditDFM 사용
        from treedfm.dfm import TreeEditDFM
        print(f"  -> [Unknown Ablation] Fallback to TreeEditDFM")
        model = TreeEditDFM(**config)

    # 3. State Dict 로드 (treedfm_edit.py 로 학습했으므로 nn.Module 포맷 그대로 사용)
    state_dict = ckpt.get('model_state', ckpt)
    model.load_state_dict(state_dict, strict=False)
    
    model.to(device)
    model.eval()
    for p in model.parameters(): p.requires_grad = False
    return model


# ==========================================
# 4. Main Evaluation Pipeline
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, default="maze_trees_relative.pkl")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--out_dir", type=str, default="ablation_eval_results")
    args = parser.parse_args()

    # 정확한 폴더 3개 명시
    TARGET_FOLDERS = [
        "ablation1_insert_only",
        "ablation2_depth_agnostic",
        "baseline_edit"
    ]
    
    ckpt_files = []
    base_dir = os.path.join("runs", "ablations")
    
    for folder in TARGET_FOLDERS:
        path = os.path.join(base_dir, folder, "last.pt")
        if os.path.exists(path):
            ckpt_files.append(path)
        else:
            # 루트 디렉토리에 그냥 풀어놓았을 경우를 대비한 Fallback
            path_alt = os.path.join(folder, "last.pt")
            if os.path.exists(path_alt):
                ckpt_files.append(path_alt)
            else:
                print(f"[Warning] Checkpoint not found for {folder}")

    if not ckpt_files:
        print("No checkpoints found to evaluate. Exiting.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # Train 데이터 사이즈 분포 로드 (Truncation 방지)
    train_sizes = []
    if os.path.exists(args.train_data):
        print(f"\nLoading reference training data: {args.train_data}")
        # max_depth를 충분히 크게 주어 (예: 999) 트리가 잘리지 않도록 함
        ds = MazeTreeDataset(args.train_data, max_depth=999, k=3)
        train_sizes = [int(x.size(0)) for x in ds.data]
        print(f"Loaded {len(train_sizes)} training trees. Max size: {max(train_sizes)}")
    else:
        print(f"\n[Warning] Training data not found at '{args.train_data}'. Histogram will only show generated sizes.")

    all_results = []
    size_distributions = {'Train': train_sizes} if train_sizes else {}
    analyzer = MazeAnalyzer(boundary_radius=30)

    # 각 실험(Ablation) 순회
    for ckpt_path in ckpt_files:
        ablation_name = os.path.basename(os.path.dirname(ckpt_path))
        print(f"\n[{ablation_name}] Loading and Sampling {args.num_samples} trees...")
        
        # 앞서 구현한 Ablation-Aware 모델 로더 호출
        model = load_ablation_model(ckpt_path, ablation_name, device)
        
        vis_dir = os.path.join(args.out_dir, ablation_name)
        visualizer = MazeVisualizer(vis_dir)

        # OOM 방지 배치 샘플링
        generated_trees = []
        batch_size = 25
        num_batches = (args.num_samples + batch_size - 1) // batch_size
        
        with torch.no_grad():
            for b in range(num_batches):
                b_size = min(batch_size, args.num_samples - len(generated_trees))
                trees = model.sample(num_samples=b_size, steps=300, max_nodes=130, temperature=1.0)
                generated_trees.extend(trees)
                print(f"    Sampled {len(generated_trees)}/{args.num_samples}")

        # Implicit Rules 분석
        metrics_list = []
        sizes = []
        for i, tree in enumerate(generated_trees):
            metrics, faults, node_states = analyzer.analyze_tree(tree)
            metrics_list.append(metrics)
            sizes.append(metrics['tree_size'])

            # 처음 5개 시각화 
            if i < 5: 
                visualizer.plot_both(tree, node_states, faults, metrics, f"sample_{i}")

        size_distributions[ablation_name] = sizes
        
        # =====================================================================
        # [핵심 수정] 통계 산출 (모든 지표를 크기가 50 이상인 트리에서만 계산)
        # =====================================================================
        df_m = pd.DataFrame(metrics_list)
        
        # 크기가 50 이상인 트리만 필터링
        df_m_large = df_m[df_m['tree_size'] >= 50]
        
        if len(df_m_large) > 0:
            avg_size_large = df_m_large['tree_size'].mean()
            perfect_trees_rate = (df_m_large['total_faults'] == 0).mean() * 100
            validity_rate_large = df_m_large['validity_rate'].mean() * 100
            avg_collisions_large = df_m_large['n_collisions'].mean()
            avg_duplicates_large = df_m_large['n_duplicate_types'].mean()
            avg_oob_large = df_m_large['n_oob'].mean()
        else:
            # 50 이상인 트리가 없으면 모두 NaN 처리
            avg_size_large = float('nan')
            perfect_trees_rate = float('nan')
            validity_rate_large = float('nan')
            avg_collisions_large = float('nan')
            avg_duplicates_large = float('nan')
            avg_oob_large = float('nan')
            print(f"[Warning] No trees with size >= 50 generated in {ablation_name}. Metrics will be NaN.")

        result_dict = {
            'Ablation Type': ablation_name,
            'Total Generated': len(generated_trees),
            'Trees >= 50 Nodes': len(df_m_large),
            'Avg Size (>=50)': avg_size_large,
            'Perfect Rate (>=50) (%)': perfect_trees_rate,
            'Node-level Validity (>=50) (%)': validity_rate_large,
            'Avg Collisions (>=50)': avg_collisions_large,
            'Avg Duplicates (>=50)': avg_duplicates_large,
            'Avg OOB (>=50)': avg_oob_large
        }
        all_results.append(result_dict)

    # ==========================================
    # 5. 최종 결과 테이블 & 시각화
    # ==========================================
    df_results = pd.DataFrame(all_results).round(2)
    csv_path = os.path.join(args.out_dir, "ablation_metrics_table.csv")
    df_results.to_csv(csv_path, index=False)
    
    print("\n" + "="*90)
    print(" Ablation Evaluation Table (Metrics Filtered by Tree Size >= 50) ")
    print("="*90)
    print(df_results.to_markdown(index=False))
    print("="*90)

    # Size Distribution 히스토그램 (히스토그램은 전체 분포를 보기 위해 필터링하지 않음)
    plt.figure(figsize=(12, 7))
    for name, sizes in size_distributions.items():
        if not sizes: continue
        # Train 데이터와 겹치게 표시 (bins 조정을 통해 전체 사이즈 커버)
        bins = range(0, max(max(sizes), 140) + 10, 5) 
        
        if name == 'Train':
            plt.hist(sizes, bins=bins, density=True, alpha=0.3, color='grey', label=f"{name} (N={len(sizes)})")
        else:
            plt.hist(sizes, bins=bins, density=True, histtype='step', linewidth=2.5, label=f"{name} (Avg={np.mean(sizes):.1f})")

    plt.title("Tree Size Distribution: Train vs Ablations (All Generated Trees)")
    plt.xlabel("Tree Size (Nodes)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(os.path.join(args.out_dir, "size_distribution_comparison.png"), dpi=150)
    plt.close()
    
    print(f"\nAll evaluations complete! Check the '{args.out_dir}' folder.")

if __name__ == "__main__":
    main()