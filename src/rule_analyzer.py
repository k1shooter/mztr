import torch

class ResourceAnalyzer:
    """자원 보존 트리의 규칙 평가: 부모 노드의 Type == 자식 노드들의 Type 합"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.faults = {'conservation_error': []}
        self.metrics = {}

    def analyze_tree(self, flat_tree):
        self.reset()
        if not flat_tree or len(flat_tree) <= 1:
            return {'tree_size': len(flat_tree), 'total_faults': 0, 'validity_rate': 0}, self.faults

        # 부모-자식 관계 그룹핑
        children_map = {}
        type_map = {}
        
        for i, node in enumerate(flat_tree):
            if isinstance(node, torch.Tensor): node = node.tolist()
            if isinstance(node, dict): n_type, p_idx = node['type'], node['parent_idx']
            else: n_type, p_idx = int(node[2]), int(node[3])
            
            type_map[i] = n_type
            if p_idx != -1:
                if p_idx not in children_map: children_map[p_idx] = []
                children_map[p_idx].append(i)

        # 보존 법칙 검사
        for parent_idx, children in children_map.items():
            if parent_idx not in type_map: continue
            parent_val = type_map[parent_idx]
            children_sum = sum([type_map[c] for c in children])
            
            if parent_val != children_sum:
                self.faults['conservation_error'].append(parent_idx)

        tree_size = len(flat_tree)
        total_faults = len(self.faults['conservation_error'])
        
        self.metrics = {
            'tree_size': tree_size,
            'total_faults': total_faults,
            'validity_rate': max(0.0, (len(children_map) - total_faults) / len(children_map)) if children_map else 1.0
        }
        return self.metrics, self.faults

class ExprAnalyzer:
    """수식 평가 트리의 규칙 평가: 올바른 구조(연산자는 2자식, 숫자는 0자식) 및 타겟값 도달 여부"""
    def __init__(self, target_val=10):
        self.target_val = target_val
        self.ADD_TYPE = 11
        self.reset()

    def reset(self):
        self.faults = {'syntax_error': [], 'eval_error': []}
        self.metrics = {}

    def analyze_tree(self, flat_tree):
        self.reset()
        tree_size = len(flat_tree)
        if tree_size <= 1:
            return {'tree_size': tree_size, 'total_faults': 0, 'validity_rate': 0}, self.faults

        children_map = {}
        type_map = {}
        for i, node in enumerate(flat_tree):
            if isinstance(node, torch.Tensor): node = node.tolist()
            if isinstance(node, dict): n_type, p_idx = node['type'], node['parent_idx']
            else: n_type, p_idx = int(node[2]), int(node[3])
            type_map[i] = n_type
            if p_idx != -1:
                if p_idx not in children_map: children_map[p_idx] = []
                children_map[p_idx].append(i)

        # 1. Syntax Error Check
        for i, n_type in type_map.items():
            num_children = len(children_map.get(i, []))
            if n_type == self.ADD_TYPE and num_children != 2: # 연산자는 자식이 2개여야 함
                self.faults['syntax_error'].append(i)
            elif n_type < self.ADD_TYPE and num_children > 0: # 숫자는 리프여야 함
                self.faults['syntax_error'].append(i)

        # 2. Eval Error Check (Post-order evaluation)
        def evaluate(idx):
            if type_map[idx] < self.ADD_TYPE: return type_map[idx] # 숫자 반환
            children = children_map.get(idx, [])
            if len(children) != 2: return -9999 # 문법 오류가 있는 경우 계산 불가
            return evaluate(children[0]) + evaluate(children[1])

        if len(self.faults['syntax_error']) == 0:
            final_val = evaluate(0)
            if final_val != self.target_val:
                self.faults['eval_error'].append(0) # 루트에서 평가 에러 발생

        total_faults = len(self.faults['syntax_error']) + len(self.faults['eval_error'])
        self.metrics = {
            'tree_size': tree_size,
            'total_faults': total_faults,
            'validity_rate': max(0.0, (tree_size - total_faults) / tree_size)
        }
        return self.metrics, self.faults

class ScopeAnalyzer:
    """의존성 트리의 규칙 평가: USE 노드는 반드시 조상 경로에 해당 DECL 노드를 가져야 함"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.faults = {'undefined_use': []}
        self.metrics = {}

    def analyze_tree(self, flat_tree):
        self.reset()
        tree_size = len(flat_tree)
        if tree_size <= 1:
            return {'tree_size': tree_size, 'total_faults': 0, 'validity_rate': 0}, self.faults

        parent_map = {}
        type_map = {}
        for i, node in enumerate(flat_tree):
            if isinstance(node, torch.Tensor): node = node.tolist()
            if isinstance(node, dict): n_type, p_idx = node['type'], node['parent_idx']
            else: n_type, p_idx = int(node[2]), int(node[3])
            type_map[i] = n_type
            parent_map[i] = p_idx

        # USE 노드(5,6,7)를 찾아서 조상 거슬러 올라가기
        USE_TYPES = [5, 6, 7]
        for i, n_type in type_map.items():
            if n_type in USE_TYPES:
                target_decl = n_type - 3 # 5->2, 6->3, 7->4
                
                # 조상 추적
                curr = parent_map[i]
                found = False
                while curr != -1:
                    if type_map[curr] == target_decl:
                        found = True
                        break
                    curr = parent_map[curr]
                    
                if not found:
                    self.faults['undefined_use'].append(i)

        total_faults = len(self.faults['undefined_use'])
        self.metrics = {
            'tree_size': tree_size,
            'total_faults': total_faults,
            'validity_rate': max(0.0, (tree_size - total_faults) / tree_size)
        }
        return self.metrics, self.faults

class SymmetricAnalyzer:
    """구조적 대칭 트리 평가: 왼쪽 자식 서브트리와 오른쪽 자식 서브트리가 대칭이어야 함"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.faults = {'asymmetry_error': []}
        self.metrics = {}

    def analyze_tree(self, flat_tree):
        self.reset()
        tree_size = len(flat_tree)
        if tree_size <= 1:
            return {'tree_size': tree_size, 'total_faults': 0, 'validity_rate': 0}, self.faults

        # 1. 플랫 트리를 재귀 딕셔너리로 재구성
        nodes = {}
        root = None
        for i, node in enumerate(flat_tree):
            if isinstance(node, torch.Tensor): node = node.tolist()
            if isinstance(node, dict): n_type, p_idx, rank = node['type'], node['parent_idx'], node.get('rank', 0)
            else: n_type, p_idx, rank = int(node[2]), int(node[3]), int(node[1])
            
            nodes[i] = {"type": n_type, "children": {}}
            if p_idx == -1: root = nodes[i]
            else:
                if p_idx in nodes: nodes[p_idx]["children"][rank] = nodes[i]

        def check_symmetric(node_L, node_R):
            """두 노드가 루트인 서브트리가 서로 완벽히 대칭(거울상)인지 검사"""
            if node_L is None and node_R is None: return True
            if node_L is None or node_R is None: return False
            if node_L["type"] != node_R["type"]: return False
            
            # K=2 라고 가정할 때 0번(Left)과 1번(Right)이 서로 교차 대칭이어야 함
            child_L0 = node_L["children"].get(0, None)
            child_L1 = node_L["children"].get(1, None)
            child_R0 = node_R["children"].get(0, None)
            child_R1 = node_R["children"].get(1, None)
            
            # Left의 왼쪽 자식 == Right의 오른쪽 자식
            return check_symmetric(child_L0, child_R1) and check_symmetric(child_L1, child_R0)

        # 루트의 자식이 2개인지 확인
        if root and len(root["children"]) == 2:
            is_sym = check_symmetric(root["children"].get(0), root["children"].get(1))
            if not is_sym:
                self.faults['asymmetry_error'].append(0) # 트리가 비대칭임
        else:
            self.faults['asymmetry_error'].append(0) # 구조 위반

        total_faults = len(self.faults['asymmetry_error'])
        self.metrics = {
            'tree_size': tree_size,
            'total_faults': total_faults,
            'validity_rate': 1.0 if total_faults == 0 else 0.0 # 대칭은 트리 전체의 O/X 문제에 가까움
        }
        return self.metrics, self.faults