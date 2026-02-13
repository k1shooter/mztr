import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pytorch_lightning as pl
import networkx as nx 
from collections import defaultdict

# ==============================================================================
# 모델 클래스 임포트
# ==============================================================================
# try:
#     from dynamic_tree_dfm4 import FlexTreeDFM, TreeTransformer, StratifiedScheduler
#     print("Successfully imported FlexTreeDFM from dynamic_tree_dfm4.py")
# except ImportError:
#     print("Error: Could not import from dynamic_tree_dfm4.py.")
#     exit(1)

from treedfm_vec import FlexTreeDFM, TreeTransformer, StratifiedScheduler
#from treedfm_mask_vec import DualStratifiedScheduler, TreeTransformer, FlexTreeDFM
# ==========================================
# 2. Maze Grid Logic & Analyzer
# ==========================================

class MazeGridSystem:
    def __init__(self):
        # Directions: 0:North, 1:East, 2:South, 3:West
        # (row_change, col_change) -> Image coord: Row increases Down, Col increases Right
        self.directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    
    def get_next_state(self, current_pos, current_dir_idx, node_type):
        r, c = current_pos
        next_dir_idx = current_dir_idx
        
        # Type 1: Left, Type 2: Forward, Type 3: Right
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

        # [수정됨] Root 초기화
        # 위치: (0, 0)
        # 방향: 1 (East/Right) -> 처음 시작 시 오른쪽을 바라봄
        self.node_states[0] = ((0, 0), 1) 
        self.occupied_positions[(0, 0)] = 0
        
        parent_child_types = defaultdict(set)
        max_depth = 0

        for i, node in enumerate(flat_tree):
            if i == 0: continue 
            
            depth, rank, n_type, parent_idx = node
            max_depth = max(max_depth, depth)
            
            if parent_idx not in self.node_states: continue

            # 1. Duplicate Check
            if n_type in parent_child_types[parent_idx]:
                self.faults['duplicate_type'].append(i)
            parent_child_types[parent_idx].add(n_type)

            # Position Calculation
            parent_pos, parent_dir_idx = self.node_states[parent_idx]
            new_pos, new_dir_idx = self.grid_system.get_next_state(parent_pos, parent_dir_idx, n_type)
            self.node_states[i] = (new_pos, new_dir_idx)

            # 2. OOB Check
            if new_pos[0]**2 + new_pos[1]**2 > self.boundary_sq:
                self.faults['oob'].append(i)
            
            # 3. Collision Check
            if new_pos in self.occupied_positions:
                self.faults['collision'].append(i)
            else:
                self.occupied_positions[new_pos] = i

        tree_size = len(flat_tree)
        n_coll = len(self.faults['collision'])
        n_oob = len(self.faults['oob'])
        n_dup = len(self.faults['duplicate_type'])
        total_faults = n_coll + n_oob + n_dup
        
        non_leaf_nodes = len(parent_child_types)
        avg_branching = (tree_size - 1) / non_leaf_nodes if non_leaf_nodes > 0 else 0

        self.metrics = {
            'tree_size': tree_size,
            'max_depth': max_depth,
            'n_collisions': n_coll,
            'n_oob': n_oob,
            'n_duplicate_types': n_dup,
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
        """
        [Maze View] 2D Grid Pixel Art Style
        """
        positions = [state[0] for state in node_states.values()]
        if not positions: return

        rows, cols = zip(*positions)
        min_r, max_r = min(rows), max(rows)
        min_c, max_c = min(cols), max(cols)
        
        # [Visual Adjustment] Margin을 조절하여 시작점이 너무 구석에 박히지 않게 함
        margin = 3
        height = (max_r - min_r + 1 + margin * 2)
        width = (max_c - min_c + 1 + margin * 2)
        
        grid_h = height * 2 + 1
        grid_w = width * 2 + 1
        
        maze_grid = np.zeros((grid_h, grid_w, 3))

        def to_grid(r, c):
            # 좌표계 변환: (min_r, min_c)가 (margin, margin) 위치에 오도록
            gr = (r - min_r + margin) * 2 + 1
            gc = (c - min_c + margin) * 2 + 1
            return int(gr), int(gc)

        collision_set = set(faults['collision'])
        oob_set = set(faults['oob'])
        dup_set = set(faults['duplicate_type'])

        for i, node in enumerate(flat_tree):
            if i not in node_states: continue
            
            curr_pos = node_states[i][0]
            curr_gr, curr_gc = to_grid(*curr_pos)
            parent_idx = node[3]

            color = [1.0, 1.0, 1.0] # White Path

            if i == 0: color = [0.0, 1.0, 0.0] # Root: Green
            elif i in collision_set: color = [1.0, 0.0, 0.0] # Red
            elif i in oob_set: color = [1.0, 0.6, 0.0] # Orange
            elif i in dup_set: color = [0.0, 1.0, 1.0] # Cyan

            maze_grid[curr_gr, curr_gc] = color

            if parent_idx != -1 and parent_idx in node_states:
                parent_pos = node_states[parent_idx][0]
                parent_gr, parent_gc = to_grid(*parent_pos)
                
                mid_gr = (curr_gr + parent_gr) // 2
                mid_gc = (curr_gc + parent_gc) // 2
                maze_grid[mid_gr, mid_gc] = color

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(maze_grid)
        ax.set_xticks([]); ax.set_yticks([])
        
        # Boundary Circle
        center_gr, center_gc = to_grid(0, 0)
        boundary_radius_px = int(np.sqrt(MazeAnalyzer().boundary_sq) * 2) 
        boundary = patches.Circle((center_gc, center_gr), boundary_radius_px, 
                                  fill=False, color='yellow', linestyle='--', alpha=0.5, linewidth=1)
        ax.add_patch(boundary)

        metrics_str = "\n".join([f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items()])
        plt.title(f"Maze Map (Size: {metrics['tree_size']})", color='black', fontsize=14)
        
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        ax.text(0.98, 0.98, metrics_str, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right', bbox=props, color='black')

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename), dpi=150)
        plt.close()

    def plot_tree_structure(self, flat_tree, faults, filename):
        """
        [Tree View] NetworkX Hierarchical Layout
        """
        G = nx.Graph()
        for i, node in enumerate(flat_tree):
            n_type = node[2]
            parent_idx = node[3]
            G.add_node(i, type=n_type)
            if parent_idx != -1:
                G.add_edge(parent_idx, i)

        try: pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
        except: 
            print("Graphviz not found, using kamada_kawai_layout")
            pos = nx.kamada_kawai_layout(G)

        node_colors = []
        collision_set = set(faults['collision'])
        
        for node in G.nodes():
            n_type = G.nodes[node]['type']
            if node == 0: c = 'lime' 
            elif node in collision_set: c = 'red'
            elif n_type == 1: c = '#ffcc00' # Left
            elif n_type == 2: c = 'white'   # Forward
            elif n_type == 3: c = 'cyan'    # Right
            else: c = 'grey'
            node_colors.append(c)

        plt.figure(figsize=(12, 8))
        plt.gca().set_facecolor('#222222')
        plt.gcf().set_facecolor('#222222')
        
        nx.draw(G, pos, node_color=node_colors, with_labels=False, node_size=100, edge_color='gray', width=0.5)
        
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', label='Start', markerfacecolor='lime', markersize=8),
            plt.Line2D([0], [0], marker='o', color='w', label='Left', markerfacecolor='#ffcc00', markersize=8),
            plt.Line2D([0], [0], marker='o', color='w', label='Forward', markerfacecolor='white', markersize=8),
            plt.Line2D([0], [0], marker='o', color='w', label='Right', markerfacecolor='cyan', markersize=8),
            plt.Line2D([0], [0], marker='o', color='w', label='Collision', markerfacecolor='red', markersize=8),
        ]
        plt.legend(handles=legend_elements, loc='upper right', facecolor='#333333', labelcolor='white')
        
        plt.title(f"Abstract Tree Structure (Size: {len(flat_tree)})", color='white')
        plt.savefig(os.path.join(self.save_dir, filename), facecolor='#222222')
        plt.close()

# ==========================================
# 4. Main Pipeline
# ==========================================

def main():
    # 본인 경로로 수정
    checkpoint_path = "/home/dmsdmswns/mazetree/src/ckpt/ins_mask_1010.ckpt"
    output_dir = "generation_results_mask1010"
    
    # pl.seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        model = FlexTreeDFM.load_from_checkpoint(checkpoint_path)
        print(f"Successfully loaded model from {checkpoint_path}")
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return

    model.to(device)
    model.eval()

    print(f"Generating trees...")
    try:
        # 충분히 큰 미로를 위해 max_nodes 2000
        generated_trees = model.sample(num_samples=10, steps=500, max_nodes=130)
    except TypeError:
        generated_trees = model.sample(num_samples=10, steps=500)

    analyzer = MazeAnalyzer(boundary_radius=30)
    visualizer = MazeVisualizer(save_dir=output_dir)
    
    print("\n--- Analysis & Visualization ---")
    for i, tree in enumerate(generated_trees):
        if len(tree) <= 1:
            print(f"Tree {i}: Failed (Only Root)")
            continue
            
        metrics, faults, node_states = analyzer.analyze_tree(tree)
        print(f"Tree {i}: Size={metrics['tree_size']}, Faults={metrics['total_faults']}")
        
        # Save Maze
        visualizer.plot_maze(tree, node_states, faults, metrics, filename=f"maze_{i}.png")
        # Save Tree Structure
        visualizer.plot_tree_structure(tree, faults, filename=f"tree_{i}.png")
        
    print(f"\nResults saved to directory: {output_dir}")

if __name__ == "__main__":
    main()