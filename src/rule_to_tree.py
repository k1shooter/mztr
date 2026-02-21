import pickle
import random
import argparse
import os

def create_node(node_type):
    return {"type": node_type, "children": []}

# =====================================================================
# 1. 자원 보존 트리 (Resource Flow / Conservation Trees)
# - 부모 노드의 Type == 자식 노드들의 Type 합
# - [수정] 루트 노드의 자원량(Type)을 max_resource(50)로 완전히 고정
# =====================================================================
def generate_resource_tree(num_samples=5000, max_resource=50, k=3, output_path="dataset_resource.pkl"):
    print(f"Generating Resource Flow Trees -> {output_path}")
    data = []
    
    def build_flow(resource, current_depth, max_depth):
        node = create_node(resource)
        if resource <= 1 or current_depth >= max_depth or random.random() < 0.2:
            return node
        
        num_children = random.randint(2, min(k, resource))
        splits = []
        rem = resource
        for _ in range(num_children - 1):
            if rem - (num_children - len(splits) - 1) <= 1:
                val = 1
            else:
                val = random.randint(1, rem - (num_children - len(splits) - 1))
            splits.append(val)
            rem -= val
        splits.append(rem)
        
        for val in splits:
            node["children"].append(build_flow(val, current_depth + 1, max_depth))
        return node

    for _ in range(num_samples):
        # [고정됨] 루트의 자원량을 항상 max_resource(예: 50)로 고정
        data.append(build_flow(max_resource, 0, max_depth=6))
        
    with open(output_path, "wb") as f:
        pickle.dump(data, f)

# =====================================================================
# 2. 수식 평가 트리 (Arithmetic Expression Trees)
# - 타겟값을 만들기 위한 덧셈 연산.
# - [고정됨] 루트는 항상 덧셈 연산자(ADD_TYPE, 11)입니다.
# =====================================================================
def generate_expr_tree(num_samples=5000, target_val=10, output_path="dataset_expr.pkl"):
    print(f"Generating Arithmetic Expr Trees -> {output_path}")
    data = []
    ADD_TYPE = 11
    
    def build_expr(target, current_depth, max_depth):
        if target == 1 or current_depth >= max_depth or random.random() < 0.3:
            return create_node(target)
            
        node = create_node(ADD_TYPE)
        left_val = random.randint(1, target - 1)
        right_val = target - left_val
        
        node["children"].append(build_expr(left_val, current_depth + 1, max_depth))
        node["children"].append(build_expr(right_val, current_depth + 1, max_depth))
        return node

    for _ in range(num_samples):
        # target_val이 1보다 크기 때문에(기본값 10) 항상 ADD_TYPE(11)으로 시작함
        data.append(build_expr(target_val, 0, max_depth=5))
        
    with open(output_path, "wb") as f:
        pickle.dump(data, f)

# =====================================================================
# 3. 스코프 의존성 트리 (Variable Scoping / Dependency Trees)
# - 조상 중 선언된(Decl) 변수만 자손이 사용(Use) 가능
# - [고정됨] 루트는 항상 블록 시작(BLOCK, 1)입니다.
# =====================================================================
def generate_scope_tree(num_samples=5000, k=3, output_path="dataset_scope.pkl"):
    print(f"Generating Scope Dependency Trees -> {output_path}")
    data = []
    
    BLOCK = 1
    DECL_A, DECL_B, DECL_C = 2, 3, 4
    USE_A, USE_B, USE_C = 5, 6, 7
    
    def build_scope(current_depth, max_depth, declared_vars, is_root=False):
        if current_depth >= max_depth:
            return create_node(BLOCK)
            
        node_type = BLOCK
        new_declared = set(declared_vars)
        
        # 루트 노드는 무조건 BLOCK(1)으로 유지되도록 확률 스킵
        if not is_root:
            rand_val = random.random()
            if rand_val < 0.3:
                var_type = random.choice([DECL_A, DECL_B, DECL_C])
                node_type = var_type
                new_declared.add(var_type)
            elif rand_val < 0.6 and new_declared:
                decl_type = random.choice(list(new_declared))
                use_type = decl_type + 3
                return create_node(use_type)
            
        node = create_node(node_type)
        num_children = random.randint(1, k)
        for _ in range(num_children):
            node["children"].append(build_scope(current_depth + 1, max_depth, new_declared, is_root=False))
        return node

    for _ in range(num_samples):
        # is_root=True를 주어 루트 노드의 타입이 항상 1(BLOCK)이 되도록 보장
        data.append(build_scope(0, max_depth=6, declared_vars=set(), is_root=True))
        
    with open(output_path, "wb") as f:
        pickle.dump(data, f)

# =====================================================================
# 4. 구조적 대칭 트리 (Structurally Symmetric Trees)
# - 좌측/우측 브랜치가 서로 완벽한 거울상을 이룸
# - [수정] 루트 노드의 색상(Type)을 1로 완전히 고정
# =====================================================================
def generate_symmetric_tree(num_samples=5000, output_path="dataset_symmetric.pkl"):
    print(f"Generating Symmetric Trees -> {output_path}")
    data = []
    
    def build_random_subtree(current_depth, max_depth):
        node = create_node(random.randint(1, 5))
        if current_depth >= max_depth or random.random() < 0.3:
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
        # [고정됨] 루트 타입을 기존 랜덤에서 1로 고정
        root = create_node(1) 
        left_subtree = build_random_subtree(0, max_depth=4)
        right_subtree = mirror_tree(left_subtree)
        root["children"] = [left_subtree, right_subtree]
        data.append(root)
        
    with open(output_path, "wb") as f:
        pickle.dump(data, f)

if __name__ == "__main__":
    os.makedirs("synthetic_datasets", exist_ok=True)
    generate_resource_tree(output_path="synthetic_datasets/resource_flow.pkl")
    generate_expr_tree(output_path="synthetic_datasets/arithmetic_expr.pkl")
    generate_scope_tree(output_path="synthetic_datasets/scope_dependency.pkl")
    generate_symmetric_tree(output_path="synthetic_datasets/symmetric_cfg.pkl")
    print("All datasets generated successfully!")