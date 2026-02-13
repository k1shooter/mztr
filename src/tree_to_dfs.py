# scripts/preprocess_maze.py
import pickle
import random
import os

# 설정 (파일 이름 확인 필요: 아까 저장한 파일명과 일치시켜 주세요)
INPUT_FILE = 'maze_trees_relative_44.pkl' 
OUTPUT_FILE = 'maze_sequences_44.pkl'

BACK_TOKEN = 4
EOS_TOKEN = 5
VOCAB_SIZE = 6 # 0~5

# 토큰 의미 (출력 확인용)
TOKEN_MAP = {
    1: "Left",
    2: "Straight",
    3: "Right",
    4: "Back",
    5: "EOS"
}

def dfs_traverse(node, sequence):
    """
    트리를 DFS로 순회하며 시퀀스를 생성합니다.
    자식 노드 방문 순서는 랜덤입니다.
    """
    # 자식 노드 리스트 복사 및 셔플 (랜덤 방문)
    children = node.get('children', [])[:]
    random.shuffle(children)
    
    for child in children:
        # 1. 자식으로 이동 (Move Token 추가)
        sequence.append(child['type'])
        
        # 2. 재귀 호출 (자식의 서브트리 순회)
        dfs_traverse(child, sequence)
        
        # 3. 부모로 복귀 (Back Token 추가)
        sequence.append(BACK_TOKEN)

def convert_trees_to_sequences():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. (파일 이름을 확인해주세요)")
        return

    with open(INPUT_FILE, 'rb') as f:
        tree_data = pickle.load(f)

    processed_sequences = []
    max_len = 0

    print(f"Processing {len(tree_data)} trees...")

    for i, (idx, root_node) in enumerate(tree_data.items()):
        seq = []
        # DFS 시작
        dfs_traverse(root_node, seq)
        
        # EOS 추가
        seq.append(EOS_TOKEN)
        
        processed_sequences.append(seq)
        max_len = max(max_len, len(seq))

        # [테스트용] 첫 번째 시퀀스만 출력해서 확인
        if i == 0:
            print(f"\n[Sample Sequence Check (ID: {idx})]")
            print(f"Raw Tokens: {seq}")
            
            # 읽기 편하게 변환해서 출력
            readable_seq = [TOKEN_MAP.get(t, str(t)) for t in seq]
            print(f"Readable: {readable_seq}")
            print(f"Length: {len(seq)}")
            print("-" * 50 + "\n")

    print(f"Max sequence length found: {max_len}")
    
    # 데이터 저장
    data_to_save = {
        'sequences': processed_sequences,
        'vocab_size': VOCAB_SIZE,
        'max_len': max_len
    }
    
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(data_to_save, f)
    
    print(f"Saved {len(processed_sequences)} sequences to {OUTPUT_FILE}")

if __name__ == "__main__":
    convert_trees_to_sequences()