import os
import math
import argparse
import random
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import networkx as nx
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import Callback
import itertools

class SetMatchingLoss(nn.Module):
    def __init__(self, k, num_types, loss_weights=None):
        super().__init__()
        self.k = k
        self.num_types = num_types
        self.weights = loss_weights
        
        # K개 슬롯의 모든 순열(Permutations)을 미리 계산해둠 (GPU)
        # 예: K=2 -> [[0,1], [1,0]]
        # 예: K=3 -> [[0,1,2], [0,2,1], ... ] (총 6개)
        perms = list(itertools.permutations(range(k)))
        self.register_buffer("permutations", torch.tensor(perms, dtype=torch.long))

    def forward(self, logits, target, weight=None):
        """
        logits: [M, K, NumTypes+1] (M: Active Nodes)
        target: [M, K] (Indices)
        weight: [NumTypes+1] (Class Weights)
        """
        # M: 배치 * 시퀀스 중 유효한 부모 노드의 개수
        M = logits.shape[0] 
        K = self.k
        P = self.permutations.size(0)
        
        # 1. 차원 확장
        # logits: [M, K, C] -> [M, 1, K, C]
        logits_exp = logits.unsqueeze(1)
        
        # 2. 타겟 순열 생성
        # target: [M, K]
        # target_perms: [M, P, K]
        # target[..., self.permutations]는 [M, P, K]가 됨
        target_perms = target[:, self.permutations] 
        
        # 3. Logits 확장
        # [M, 1, K, C] -> [M, P, K, C]
        logits_rep = logits_exp.expand(-1, P, -1, -1)
        
        # 4. Cross Entropy 계산 (Reduction='none')
        # Input: [M*P*K, C]
        # Target: [M*P*K]
        loss_flat = F.cross_entropy(
            logits_rep.reshape(-1, self.num_types+1),
            target_perms.reshape(-1),
            weight=weight if weight is not None else self.weights,
            reduction='none'
        )
        
        # 5. Loss 재조합
        # [M*P*K] -> [M, P, K]
        loss_tensor = loss_flat.view(M, P, K)
        
        # 6. 슬롯(K)에 대해 합산 -> 각 순열(P)별 총 Cost
        # [M, P]
        cost_per_perm = loss_tensor.sum(dim=-1)
        
        # 7. 최적 순열 선택 (Min over P)
        # [M]
        min_loss_val, _ = torch.min(cost_per_perm, dim=-1)
        
        # 8. 최종 평균
        return min_loss_val.mean()

class StratifiedScheduler:
    def __init__(self, max_depth, width=0.5):
        self.max_depth = max_depth
        self.width = width

    def get_kappa_and_psi(self, t, depths):
        """
        Returns kappa (interpolant) and psi (rate scalar) strictly derived.
        kappa(t, d) = (t - start) / width, clamped [0, 1]
        psi(t, d) = d_kappa/dt / (1 - kappa)
        """
        # t: (B,) -> (B, 1) for broadcasting
        if isinstance(t, float): t = torch.tensor(t)
        t = t.view(-1, 1).to(depths.device)
        
        # depths: (B, N). No need to reshape to (1, -1)!
        # Just ensure it's float for division
        d = depths.float()
        
        # Start time per depth
        t_start = d / self.max_depth
        
        # Linear ramp progress
        # (B, 1) - (B, N) -> (B, N) Broadcasting works correctly here
        numer = t - t_start
        progress = numer / self.width
        
        kappa = torch.clamp(progress, 0.0, 1.0)
        
        # Derivative of kappa w.r.t t is 1/width where 0 < kappa < 1
        d_kappa = torch.where((progress > 0) & (progress < 1), 
                              torch.tensor(1.0/self.width, device=t.device), 
                              torch.tensor(0.0, device=t.device))
        
        # Psi = d_kappa / (1 - kappa)
        denom = 1.0 - kappa
        denom = torch.clamp(denom, min=1e-6)
        
        psi = d_kappa / denom
        
        # Explicitly zero out psi where kappa=1 (Already target) or kappa=0 (Not started)
        mask_active = (progress > 0) & (progress < 1.0)
        psi = psi * mask_active.float()
        
        return kappa, psi

# --- 2. Data & Utils (Unchanged) ---
def pad_sequence(batch, padding_value=0):
    batch = [item for item in batch]
    max_len = max([x.size(0) for x in batch])
    padded_batch = []
    masks = []
    for x in batch:
        mask = torch.ones(max_len, dtype=torch.bool)
        mask[:x.size(0)] = False 
        if x.size(0) < max_len:
            pad = torch.full((max_len - x.size(0), x.size(1)), padding_value, dtype=x.dtype)
            x_padded = torch.cat([x, pad], dim=0)
        else:
            x_padded = x
        padded_batch.append(x_padded)
        masks.append(mask)
    return torch.stack(padded_batch), torch.stack(masks)

class TreeUtils:
    @staticmethod
    def flatten_tree(root, max_depth, k):
        flat_nodes = []
        queue = [(root, (), -1)] 
        while queue:
            node, path, parent_idx = queue.pop(0)
            if len(path) > max_depth: continue
            node_type = node.get('type', 1) 
            depth = len(path); rank = path[-1] if depth > 0 else 0
            flat_nodes.append({'depth': depth, 'rank': rank, 'type': node_type, 'parent_idx': parent_idx, 'path': path})
            children = node.get('children', [])
            children.sort(key=lambda x: x.get('id', 0))
            for i, child in enumerate(children):
                if i >= k: break 
                queue.append((child, path + (i,), len(flat_nodes)-1))
        return flat_nodes

    @staticmethod
    def save_tree_plot(nodes_list, filename, title=""):
        G = nx.Graph()
        for i, n in enumerate(nodes_list):
            if isinstance(n, torch.Tensor): n = n.tolist()
            ntype = n[2]; parent = n[3]
            G.add_node(i, type=ntype, depth=n[0])
            if parent != -1: G.add_edge(parent, i)
        plt.figure(figsize=(10, 8))
        try: pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
        except: pos = nx.spring_layout(G)
        colors = ['#eeeeee', '#ff9999', '#99ccff', '#99ff99']
        nc = [colors[G.nodes[i].get('type', 0) % 4] for i in G.nodes()]
        nx.draw(G, pos, node_color=nc, with_labels=True, node_size=300, cmap=plt.cm.Set3)
        plt.title(title); plt.savefig(filename); plt.close()

# --- 3. Dataset (Unchanged) ---
class MazeTreeDataset(Dataset):
    def __init__(self, pkl_path, max_depth=100, k=3):
        self.max_depth = max_depth; self.k = k
        with open(pkl_path, 'rb') as f: self.raw_data = pickle.load(f)
        self.data = [TreeUtils.flatten_tree(r, max_depth, k) for r in self.raw_data.values()]
        print(f"[Dataset] Loaded {len(self.data)} trees.")
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

def distinct_collate_fn(batch):
    feats = [torch.tensor([[n['depth'], n['rank'], n['type'], n['parent_idx']] for n in nodes], dtype=torch.long) for nodes in batch]
    return pad_sequence(feats, padding_value=0)

# --- 4. Training Visualizer (Unchanged) ---
class TrainingVisualizer(Callback):
    def __init__(self, save_dir="training_logs", sample_interval=5):
        super().__init__()
        self.save_dir = save_dir; self.sample_interval = sample_interval
        self.loss_history = []; self.avg_size_history = []
        os.makedirs(save_dir, exist_ok=True)
    def on_train_epoch_end(self, trainer, pl_module):
        current_epoch = trainer.current_epoch + 1
        if "train_loss" in trainer.callback_metrics: self.loss_history.append(trainer.callback_metrics["train_loss"].item())
        if current_epoch % self.sample_interval == 0:
            print(f"\n[Visualizer] Sampling at epoch {current_epoch}...")
            trees = pl_module.sample(num_samples=3, steps=500) 
            avg_size = sum(len(t) for t in trees) / len(trees)
            self.avg_size_history.append((current_epoch, avg_size))
            epoch_dir = os.path.join(self.save_dir, f"epoch_{current_epoch}")
            os.makedirs(epoch_dir, exist_ok=True)
            for idx, nodes in enumerate(trees):
                TreeUtils.save_tree_plot(nodes, os.path.join(epoch_dir, f"sample_{idx}.png"), title=f"Sz {len(nodes)}")
            self._plot_metrics()
    def _plot_metrics(self):
        if not self.loss_history: return
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(self.loss_history, color='blue'); ax1.set_ylabel('Loss', color='blue')
        if self.avg_size_history:
            ax2 = ax1.twinx(); ep, sz = zip(*self.avg_size_history)
            ax2.plot(ep, sz, color='red', marker='o'); ax2.set_ylabel('Size', color='red')
        plt.savefig(os.path.join(self.save_dir, "loss_size_plot.png")); plt.close()

# --- 5. Model (Unchanged) ---
class TreeTransformer(nn.Module):
    def __init__(self, num_types, k, d_model=256, n_heads=4, n_layers=6, max_depth=100):
        super().__init__()
        self.d_model = d_model; self.k = k
        self.type_emb = nn.Embedding(num_types + 1, d_model, padding_idx=0)
        self.depth_emb = nn.Embedding(max_depth + 2, d_model) # +2 for safety
        self.rank_emb = nn.Embedding(k + 1, d_model)
        self.time_mlp = nn.Sequential(nn.Linear(1, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model*2, dropout=0.1, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.growth_head = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, k * (num_types + 1)))
    def forward(self, x, mask, t):
        h = self.type_emb(x[:,:,2]) + self.depth_emb(x[:,:,0].clamp(max=150)) + self.rank_emb(x[:,:,1].clamp(max=self.k))
        h = h + self.time_mlp(t.view(-1, 1)).unsqueeze(1)
        out = self.transformer(h, src_key_padding_mask=mask)
        return self.growth_head(out).view(out.size(0), out.size(1), self.k, -1)

# --- 6. Lightning Module (Rigorous DeFoG Implementation) ---
class FlexTreeDFM(pl.LightningModule):
    def __init__(self, num_types=4, k=3, max_depth=150, lr=1e-3, pos_weight=50.0): # Added pos_weight
        super().__init__()
        self.save_hyperparameters()
        self.model = TreeTransformer(num_types, k, max_depth=max_depth)
        self.k = k; self.num_types = num_types
        
        # Rigorous Stratified Scheduler
        self.scheduler = StratifiedScheduler(max_depth, width=0.5)
        
        
        # Loss weighting to fix sparsity issue (empty vs real)
        self.loss_weights = torch.ones(num_types + 1)
        self.loss_weights[0] = 1.0 # Empty class
        self.loss_weights[1:] = pos_weight # Real classes weighted higher
        self.matcher_loss = SetMatchingLoss(self.k, self.num_types, self.loss_weights)

    def get_stratified_mask(self, batch_size, seq_len, parents, depths, t):
        """
        Rigorous Noising: Selects a subset x_t based on depth-dependent kappa.
        """
        # Get Kappa(t, d) for all nodes
        kappa, _ = self.scheduler.get_kappa_and_psi(t, depths) # (B, N)
        
        # Sample Bernoulli(kappa)
        u = torch.rand_like(kappa)
        keep_mask = u < kappa
        keep_mask[:, 0] = True # Root always exists
        
        # Enforce Structural Connectivity (Parent must be kept if child is kept)
        safe_parents = parents.clone()
        safe_parents[safe_parents < 0] = 0
        
        for _ in range(self.scheduler.max_depth):
            parent_kept = keep_mask.gather(1, safe_parents)
            
            # Condition: If parent is valid (>=0) and parent is NOT kept, child must be NOT kept.
            # Equivalently: keep_mask = keep_mask AND (parent_kept OR parent < 0)
            has_parent = parents >= 0
            condition = (~has_parent) | parent_kept
            
            new_mask = keep_mask & condition
            if torch.equal(new_mask, keep_mask):
                break
            keep_mask = new_mask
        
        return keep_mask

    def training_step(self, batch, batch_idx):
        x_1, padding_mask = batch
        bs, seq_len, _ = x_1.size()
        parents = x_1[:, :, 3].clone(); parents[padding_mask] = -100
        depths = x_1[:, :, 0]
        
        t = torch.rand(bs, device=self.device)
        
        # 1. Rigorous Stratified Noising
        keep_mask = self.get_stratified_mask(bs, seq_len, parents, depths, t)
        
        # 2. Model Forward
        model_input_mask = ~keep_mask | padding_mask
        # Model predicts s_theta(x_t) which approximates p(x_1 | x_t)
        pred_logits = self.model(x_1, model_input_mask, t) 
        
        # 3. Target Construction
        target_types = torch.zeros(bs, seq_len, self.k, dtype=torch.long, device=self.device)
        # Indices
        batch_idx = torch.arange(bs, device=self.device).view(-1, 1).expand(-1, seq_len)
        node_idx = torch.arange(seq_len, device=self.device).view(1, -1).expand(bs, -1)
        
        p_idx = parents
        r_idx = x_1[:, :, 1] # Rank
        type_val = x_1[:, :, 2] # Type
        
        # Condition:
        # 1. Not padding
        # 2. Parent is valid (>=0)
        # 3. Parent is kept (active in x_t)
        # 4. Rank is valid (< k)
        
        safe_p = p_idx.clone(); safe_p[safe_p < 0] = 0
        parent_kept = keep_mask.gather(1, safe_p)
        
        valid_condition = (~padding_mask) & (p_idx >= 0) & parent_kept & (r_idx < self.k)
        
        b_sel = batch_idx[valid_condition]
        p_sel = p_idx[valid_condition]
        r_sel = r_idx[valid_condition]
        val_sel = type_val[valid_condition]
        
        # Scatter targets
        target_types.index_put_((b_sel, p_sel, r_sel), val_sel)

        # 4. Loss Calculation
        active_mask = keep_mask & (~padding_mask)
        if active_mask.sum() == 0: 
            return torch.tensor(0.0, device=self.device, requires_grad=True)
            
        # Weighted CrossEntropy handles the sparsity/class imbalance
        # loss = F.cross_entropy(
        #     pred_logits[active_mask].view(-1, self.num_types+1), 
        #     target_types[active_mask].view(-1),
        #     weight=self.loss_weights.to(self.device)
        # )
        active_logits = pred_logits[active_mask] 
        active_targets = target_types[active_mask]
        loss = self.matcher_loss(active_logits, active_targets, weight=self.loss_weights.to(self.device))
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self): 
        return torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr)

    @torch.no_grad()
    def sample(self, num_samples=1, steps=500, max_nodes=200): # max_nodes 제한 추가
        self.eval(); device = self.device
        generated_trees = [[ [0, 0, 2, -1] ] for _ in range(num_samples)]
        dt = 1.0 / steps
        
        for s in range(steps):
            t_val = s * dt
            t_tensor = torch.full((num_samples,), t_val, device=device)
            
            # --- Safe Guard: 트리가 너무 커지면 OOM 방지를 위해 중단 ---
            current_max_len = max(len(t) for t in generated_trees)
            if current_max_len > max_nodes:
                print(f"[Warning] Tree size exceeded limit ({max_nodes}). Stopping generation to prevent OOM.")
                break

            features_list = [torch.tensor(n, dtype=torch.long) for n in generated_trees]
            x_curr, mask = pad_sequence(features_list, padding_value=0)
            
            # GPU 메모리 효율을 위해 필요한 부분만 GPU로
            x_curr = x_curr.to(device); mask = mask.to(device)
            
            logits = self.model(x_curr, mask, t_tensor)
            probs = F.softmax(logits, dim=-1)
            
            current_depths = x_curr[:, :, 0]
            child_depths = current_depths + 1
            _, psi = self.scheduler.get_kappa_and_psi(t_tensor, child_depths)
            
            for b in range(num_samples):
                # 개별 트리가 제한을 넘으면 해당 트리는 성장을 멈춤
                if len(generated_trees[b]) >= max_nodes: continue

                curr_nodes = generated_trees[b]; n_curr = len(curr_nodes)
                existing_children = set((n[3], n[1]) for n in curr_nodes)
                
                for i in range(n_curr):
                    parent_type = curr_nodes[i][2]
                    if parent_type == 0: continue
                    
                    rate_scalar = psi[b, i].item()
                    if rate_scalar < 1e-6: continue
                    
                    for k in range(self.k):
                        if (i, k) in existing_children: continue
                        
                        slot_probs = probs[b, i, k]
                        p_empty = slot_probs[0].item()
                        p_exists = 1.0 - p_empty
                        
                        spawn_prob = rate_scalar * p_exists * dt
                        
                        # --- Safety: 확률이 비정상적으로 높으면 클리핑 ---
                        spawn_prob = min(spawn_prob, 1.0)

                        if random.random() < spawn_prob:
                            type_dist = slot_probs[1:]
                            if type_dist.sum() > 0:
                                type_idx = torch.multinomial(type_dist, 1).item() + 1
                                generated_trees[b].append([curr_nodes[i][0]+1, k, type_idx, i])
                                existing_children.add((i, k))
        return generated_trees

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl_path", type=str, default="maze_trees_relative.pkl")
    parser.add_argument("--save_dir", type=str, default="dfm_logs_v3_1010_vec_hung")
    parser.add_argument("--max_depth", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-4)
    # Added argument for sparsity control
    parser.add_argument("--pos_weight", type=float, default=1.0, help="Weight for real nodes vs empty")
    args = parser.parse_args()
    
    pl.seed_everything(42)
    if not os.path.exists(args.pkl_path): return
    
    ds = MazeTreeDataset(args.pkl_path, max_depth=args.max_depth)
    dl = DataLoader(ds, batch_size=args.batch_size, collate_fn=distinct_collate_fn, shuffle=True, num_workers=8)
    
    # Initialize with new parameters
    model = FlexTreeDFM(max_depth=args.max_depth, lr=args.lr, pos_weight=args.pos_weight)
    vis_cb = TrainingVisualizer(save_dir=args.save_dir)
    
    trainer = pl.Trainer(max_epochs=args.epochs, accelerator="auto", callbacks=[vis_cb], log_every_n_steps=10)
    trainer.fit(model, dl)

if __name__ == "__main__":
    main()