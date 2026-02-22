import pickle
import random
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def create_node(node_type):
    return {"type": node_type, "children": []}

def shuffle_tree(node):
    if node.get("children"):
        random.shuffle(node["children"])
        for child in node["children"]:
            shuffle_tree(child)

# =====================================================================
# 시각화 헬퍼 함수
# =====================================================================
def save_tree_visualizations(trees, dataset_name, output_dir="synthetic_datasets/visualizations", num_samples=3):
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(min(num_samples, len(trees))):
        G = nx.DiGraph()
        labels = {}
        node_colors = []
        cmap = plt.get_cmap('Set3')
        
        queue = [(trees[i], 0)]
        current_id = 0
        
        while queue:
            curr_node, curr_id = queue.pop(0)
            n_type = curr_node["type"]
            
            G.add_node(curr_id, type=n_type)
            labels[curr_id] = f"{curr_id}\n(T:{n_type})"
            node_colors.append(cmap(n_type % 12))
            
            for child in curr_node.get("children", []):
                current_id += 1
                G.add_edge(curr_id, current_id)
                queue.append((child, current_id))

        plt.figure(figsize=(10, 8))
        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
        except:
            pos = nx.kamada_kawai_layout(G)

        nx.draw(G, pos, with_labels=True, labels=labels, node_color=node_colors,
                node_size=800, font_size=8, edge_color='gray', arrows=True)

        plt.title(f"{dataset_name} - Sample {i+1} (Size: {len(G.nodes)})")
        filepath = os.path.join(output_dir, f"{dataset_name}_sample_{i+1}.png")
        plt.savefig(filepath, bbox_inches='tight')
        plt.close()

def save_tree_metrics_histograms(trees, dataset_name, output_dir="synthetic_datasets/visualizations"):
    os.makedirs(output_dir, exist_ok=True)
    sizes, depths = [], []
    
    def get_size_and_depth(node):
        if not node.get("children"): return 1, 0
        sz, max_d = 1, 0
        for child in node["children"]:
            c_sz, c_d = get_size_and_depth(child)
            sz += c_sz
            max_d = max(max_d, c_d + 1)
        return sz, max_d
        
    for tree in trees:
        sz, d = get_size_and_depth(tree)
        sizes.append(sz)
        depths.append(d)
        
    # Size Histogram
    plt.figure(figsize=(8, 5))
    max_size = max(sizes) if sizes else 1
    bins_size = range(0, max_size + 5, max(1, max_size // 20))
    plt.hist(sizes, bins=bins_size, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title(f"{dataset_name} Tree Size Distribution\nMean: {np.mean(sizes):.1f}, Max: {max_size}")
    plt.xlabel("Number of Nodes")
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_size_dist.png"), bbox_inches='tight')
    plt.close()
    
    # Depth Histogram
    plt.figure(figsize=(8, 5))
    max_depth_val = max(depths) if depths else 1
    bins_depth = np.arange(-0.5, max_depth_val + 1.5, 1)
    plt.hist(depths, bins=bins_depth, color='lightcoral', edgecolor='black', alpha=0.7)
    plt.title(f"{dataset_name} Max Depth Distribution\nMean: {np.mean(depths):.1f}, Max: {max_depth_val}")
    plt.xlabel("Max Depth")
    plt.ylabel("Frequency")
    plt.xticks(range(0, max_depth_val + 1))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_depth_dist.png"), bbox_inches='tight')
    plt.close()

# =====================================================================
# 1. 자원 보존 트리 (Resource Flow)
# - 루트 값(Target)은 항상 60으로 엄격히 고정
# - Types: 1 ~ 60 (총 60개 고정)
# =====================================================================
def generate_resource_tree(num_samples=10000, target_res=60, k=3, output_path="synthetic_datasets/resource_flow.pkl"):
    print(f"\nGenerating Resource Flow Trees -> {output_path}")
    data = []
    
    def build_flow(resource, current_depth, max_depth):
        node = create_node(resource)
        if resource <= 1 or current_depth >= max_depth or (current_depth > 0 and random.random() < 0.1):
            return node
        
        num_children = random.randint(2, min(k, resource))
        splits = []
        rem = resource
        for _ in range(num_children - 1):
            val = 1 if rem - (num_children - len(splits) - 1) <= 1 else random.randint(1, rem - (num_children - len(splits) - 1))
            splits.append(val)
            rem -= val
        splits.append(rem)
        
        for val in splits:
            node["children"].append(build_flow(val, current_depth + 1, max_depth))
        return node

    for _ in range(num_samples):
        max_d = random.randint(4, 12)
        # [수정] 룰 학습을 위해 시작 자원량(60) 절대 고정
        tree = build_flow(target_res, 0, max_depth=max_d)
        shuffle_tree(tree)
        data.append(tree)
        
    with open(output_path, "wb") as f: pickle.dump(data, f)
    save_tree_visualizations(data, "Resource_Flow")
    save_tree_metrics_histograms(data, "Resource_Flow")

# =====================================================================
# 2. 수식 평가 트리 (Arithmetic Expression)
# - 타겟 값(Sum)은 항상 30으로 엄격히 고정
# - Types: 1~10 (리프/값), 11 (ADD 연산자) -> (총 11개 고정)
# =====================================================================
def generate_expr_tree(num_samples=10000, target_val=30, output_path="synthetic_datasets/arithmetic_expr.pkl"):
    print(f"\nGenerating Arithmetic Expr Trees -> {output_path}")
    data = []
    ADD_TYPE = 11
    
    def build_expr(target, current_depth, max_depth):
        # [수정] 값이 1이면 더 이상 쪼갤 수 없으므로 확률/깊이 무시하고 무조건 종료!
        # 그 외에 타겟이 10 이하일 때는 기존처럼 깊이나 확률에 따라 조기 종료.
        if target == 1 or (target <= 10 and (current_depth >= max_depth or random.random() < 0.2)):
            return create_node(target)
            
        node = create_node(ADD_TYPE)
        
        # 합이 target이 되도록 분할 (target이 무조건 2 이상이므로 randint(1, target-1)은 안전함)
        left_val = random.randint(1, target - 1)
        right_val = target - left_val
        
        node["children"].append(build_expr(left_val, current_depth + 1, max_depth))
        node["children"].append(build_expr(right_val, current_depth + 1, max_depth))
        return node

    for _ in range(num_samples):
        max_d = random.randint(4, 10)
        tree = build_expr(target_val, 0, max_depth=max_d)
        shuffle_tree(tree)
        data.append(tree)
        
    with open(output_path, "wb") as f: pickle.dump(data, f)
    save_tree_visualizations(data, "Arithmetic_Expr")
    save_tree_metrics_histograms(data, "Arithmetic_Expr")

# =====================================================================
# 3. 스코프 의존성 트리 (Scope Dependency)
# =====================================================================
# =====================================================================
# 3. 스코프 의존성 트리 (Scope Dependency)
# - Types: 1(BLOCK), 2~4(DECL), 5~7(USE)
# - [수정] 트리 생성 시 총 노드 수를 추적하여 최대 크기(MAX_NODES)를 넘지 않도록 안전장치 추가
# =====================================================================
def generate_scope_tree(num_samples=10000, k=3, output_path="synthetic_datasets/scope_dependency.pkl"):
    print(f"\nGenerating Scope Dependency Trees -> {output_path}")
    data = []
    MAX_NODES = 200
    
    def build_scope(current_depth, max_depth, declared_vars, is_root, current_size):
        if current_depth >= max_depth or current_size[0] >= MAX_NODES:
            current_size[0] += 1
            # 종료 시 최대한 의미 있는 USE 노드로 끝맺음 (85% 확률)
            if declared_vars and random.random() < 0.85:
                return create_node(random.choice(list(declared_vars)) + 3)
            return create_node(1) # 부득이한 경우만 BLOCK
            
        node_type = 1 
        new_declared = set(declared_vars)
        
        if not is_root:
            size_ratio = min(1.0, current_size[0] / 140.0) 
            
            # [핵심 로직] 확률 밸런스 완전 개편
            decl_prob = 0.45  # 변수 선언 확률을 45%로 고정 (다채로움 확보)
            use_prob = 0.15 + 0.7 * size_ratio # 15%에서 시작해 최대 85%까지 상승
            
            # rand_val이 decl_prob(45%) 안에 들면 선언
            # decl_prob ~ (decl_prob+use_prob) 안에 들면 사용
            # 그 밖으로 튕겨나가면(나머지 확률) BLOCK이 됨
            rand_val = random.random()
            if rand_val < decl_prob:
                var_type = random.choice([2, 3, 4]) 
                node_type = var_type
                new_declared.add(var_type)
            elif rand_val < decl_prob + use_prob and new_declared:
                current_size[0] += 1
                return create_node(random.choice(list(new_declared)) + 3)
            
        node = create_node(node_type)
        current_size[0] += 1
        
        if is_root:
            min_child, max_child = 2, 3
        elif current_depth == 1:
            min_child, max_child = 1, 3
        else:
            min_child, max_child = 0, 2
            
        if current_size[0] > 140:
            max_child = min(1, max_child)
            
        if min_child > max_child:
            max_child = min_child
            
        num_children = random.randint(min_child, max_child)
        
        for _ in range(num_children):
            if current_size[0] >= MAX_NODES:
                break
            node["children"].append(build_scope(current_depth + 1, max_depth, new_declared, False, current_size))
            
        return node

    for _ in range(num_samples):
        max_d = random.randint(5, 10) 
        current_size_tracker = [0] 
        tree = build_scope(0, max_depth=max_d, declared_vars=set(), is_root=True, current_size=current_size_tracker)
        shuffle_tree(tree)
        data.append(tree)
        
    with open(output_path, "wb") as f: pickle.dump(data, f)
    save_tree_visualizations(data, "Scope_Dependency")
    save_tree_metrics_histograms(data, "Scope_Dependency")

# =====================================================================
# 4. 구조적 대칭 트리 (Symmetric Tree)
# =====================================================================
def generate_symmetric_tree(num_samples=10000, output_path="synthetic_datasets/symmetric_cfg.pkl"):
    print(f"\nGenerating Symmetric Trees -> {output_path}")
    data = []
    
    def build_random_subtree(current_depth, max_depth):
        node = create_node(random.randint(1, 5))
        if current_depth >= max_depth or (current_depth > 0 and random.random() < 0.2):
            return node
        num_children = random.randint(1, 2)
        for _ in range(num_children):
            node["children"].append(build_random_subtree(current_depth + 1, max_depth))
        return node
        
    def mirror_tree(node):
        new_node = create_node(node["type"])
        for child in reversed(node["children"]):
            new_node["children"].append(mirror_tree(child))
        return new_node

    for _ in range(num_samples):
        # 대칭 구조 유지를 위해 루트 타입 항상 1로 고정
        root = create_node(1) 
        max_d = random.randint(3, 7)
        left_subtree = build_random_subtree(0, max_depth=max_d)
        right_subtree = mirror_tree(left_subtree)
        root["children"] = [left_subtree, right_subtree]
        
        shuffle_tree(root)
        data.append(root)
        
    with open(output_path, "wb") as f: pickle.dump(data, f)
    save_tree_visualizations(data, "Symmetric_Tree")
    save_tree_metrics_histograms(data, "Symmetric_Tree")

if __name__ == "__main__":
    os.makedirs("synthetic_datasets", exist_ok=True)
    generate_resource_tree()
    generate_expr_tree()
    generate_scope_tree()
    generate_symmetric_tree()
    print("\nAll datasets generated and sampled successfully!")