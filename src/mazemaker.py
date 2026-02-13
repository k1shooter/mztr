import random

def get_neighbors(node, width, height):
    """현재 노드의 상하좌우 이웃 노드 인덱스 반환"""
    r, c = divmod(node, width)
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)] # 상, 하, 좌, 우
    neighbors = []
    for dr, dc in moves:
        nr, nc = r + dr, c + dc
        if 0 <= nr < height and 0 <= nc < width:
            neighbors.append(nr * width + nc)
    return neighbors

def generate_maze_edges(width, height):
    """DFS(Recursive Backtracker)를 이용한 미로 엣지 생성"""
    start_node = 0
    stack = [start_node]
    visited = {start_node}
    edges = []

    while stack:
        current = stack[-1]
        neighbors = get_neighbors(current, width, height)
        # 방문하지 않은 이웃만 선택
        unvisited_neighbors = [n for n in neighbors if n not in visited]

        if unvisited_neighbors:
            # 랜덤하게 이웃 선택 (이동)
            next_node = random.choice(unvisited_neighbors)
            visited.add(next_node)
            stack.append(next_node)
            
            # 엣지 추가 (항상 작은 숫자가 앞에 오도록 정렬하여 저장)
            u, v = sorted((current, next_node))
            edges.append((u, v))
        else:
            # 갈 곳이 없으면 백트래킹
            stack.pop()
            
    # 보기 좋게 엣지 리스트 정렬
    edges.sort()
    return edges

# 1000개 생성 및 파일 저장
output_filename = "maze_4x4_1000.txt"
with open(output_filename, "w") as f:
    for i in range(1000):
        edges = generate_maze_edges(4, 4)
        # 형식: "Index: [(u, v), ...]"
        f.write(f"{i}: {edges}\n")

print(f"Generated {output_filename}")