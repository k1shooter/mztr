import pickle
import numpy as np
import random
import os
import argparse

def create_node(node_type, node_id):
    """트리 노드 생성 헬퍼 함수"""
    return {
        "type": node_type,
        "children": [],
        "id": node_id
    }

def generate_dataset_1(num_samples=10000, mean_size=50, std_size=10, output_path="dataset_descending.pkl"):
    """
    조건 1: 타입 51개, k<=4, Child Type < Parent Type
    """
    print(f"Generating Dataset 1 (Descending Types) -> {output_path} ...")
    data = []
    
    for _ in range(num_samples):
        # 1. 목표 크기 샘플링
        target_size = int(np.random.normal(mean_size, std_size))
        target_size = max(1, target_size)
        
        # 2. 루트 생성
        # 자식 타입이 부모보다 작아야 하므로, 트리가 깊어지려면 루트 타입이 커야 함
        # 여기서는 트리의 다양성을 위해 20~51 사이에서 랜덤 시작하거나, 항상 51로 시작할 수 있음
        root_type = 51 
        root = create_node(root_type, 0)
        
        nodes = [root]
        # 확장이 가능한 노드 후보군 (type > 1 이고 자식이 4개 미만인 노드)
        growable = [root]
        
        while len(nodes) < target_size and growable:
            # 확장할 부모 노드 랜덤 선택
            parent = random.choice(growable)
            
            # 자식 타입 결정 (1 ~ parent.type - 1)
            if parent["type"] <= 1:
                growable.remove(parent)
                continue
                
            child_type = random.randint(1, parent["type"] - 1)
            child = create_node(child_type, len(nodes))
            
            # 연결
            parent["children"].append(child)
            nodes.append(child)
            
            # 자식 노드도 확장 가능한지 확인 후 추가
            if child["type"] > 1:
                growable.append(child)
            
            # 부모 노드가 꽉 찼으면(k=4) 후보군에서 제거
            if len(parent["children"]) >= 4:
                if parent in growable:
                    growable.remove(parent)
        
        data.append(root)
        
    with open(output_path, "wb") as f:
        pickle.dump(data, f)
    print(f"Done! {len(data)} trees saved.")

def generate_dataset_2(num_samples=10000, mean_size=50, std_size=10, output_path="dataset_even_branch.pkl"):
    """
    조건 2: 타입 3개, k<=4, 모든 노드의 자식 수는 짝수 (0, 2, 4)
    """
    print(f"Generating Dataset 2 (Even Branching) -> {output_path} ...")
    data = []
    
    for _ in range(num_samples):
        # 1. 목표 크기 샘플링 (구조상 항상 홀수 개가 됨)
        target_size = int(np.random.normal(mean_size, std_size))
        target_size = max(1, target_size)
        
        # 2. 루트 생성
        root = create_node(random.randint(1, 3), 0)
        nodes = [root]
        
        # 확장은 항상 리프 노드(자식 0개)에서 일어나야 자식 수 규칙(0->2 or 0->4)을 유지하기 쉬움
        leaves = [root]
        
        while len(nodes) < target_size and leaves:
            # 확장할 리프 노드 선택
            parent = random.choice(leaves)
            
            # 남은 용량에 따라 2개 혹은 4개 추가
            remaining_space = target_size - len(nodes)
            
            # 공간이 4개 이상 남았으면 2 또는 4개 추가, 2개 남았으면 2개, 아니면 중단
            options = []
            if remaining_space >= 2: options.append(2)
            if remaining_space >= 4: options.append(4)
            
            if not options:
                break
                
            num_children = random.choice(options)
            
            # 리프 노드였던 parent는 이제 내부 노드가 되므로 leaves에서 제거
            if parent in leaves:
                leaves.remove(parent)
                
            # 자식 생성 및 연결
            for _ in range(num_children):
                child_type = random.randint(1, 3) # 타입은 1~3 자유
                child = create_node(child_type, len(nodes))
                parent["children"].append(child)
                nodes.append(child)
                leaves.append(child) # 새 자식들은 리프 노드가 됨
        
        data.append(root)

    with open(output_path, "wb") as f:
        pickle.dump(data, f)
    print(f"Done! {len(data)} trees saved.")

if __name__ == "__main__":
    # 설정값 조절 가능
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=10000, help="Number of trees per dataset")
    parser.add_argument("--mean", type=int, default=50, help="Mean tree size")
    parser.add_argument("--std", type=int, default=10, help="Std dev of tree size")
    args = parser.parse_args()

    # 데이터셋 1 생성
    generate_dataset_1(
        num_samples=args.samples, 
        mean_size=args.mean, 
        std_size=args.std,
        output_path="dataset_descending.pkl"
    )
    
    # 데이터셋 2 생성
    generate_dataset_2(
        num_samples=args.samples, 
        mean_size=args.mean, 
        std_size=args.std,
        output_path="dataset_even_branch.pkl"
    )