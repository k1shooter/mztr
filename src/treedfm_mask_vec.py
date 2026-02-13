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

# ==========================================
# 1. Dual Scheduler (Topology & Semantics)
# ==========================================
class DualStratifiedScheduler:
    """
    Manages two coupled schedules for Mask-First Generation.
    1. Alpha(t): Insertion Probability (Empty -> Mask)
    2. Beta(t):  Unmasking Probability (Mask -> Real)
    """
    def __init__(self, max_depth, width=0.5, lag=0.005):
        self.max_depth = max_depth
        self.width = width
        self.lag = lag

    # def get_coeffs_and_rates(self, t, depths):
    #     if isinstance(t, float): t = torch.tensor(t)
    #     t = t.to(depths.device)
    #     d = depths.float()
        
    #     t_start = d / self.max_depth
        
    #     # 1. Insertion Flow (Alpha)
    #     prog_ins = (t - t_start) / self.width
    #     alpha = torch.clamp(prog_ins, 0.0, 1.0)
        
    #     d_alpha = torch.where((prog_ins > 0) & (prog_ins < 1),
    #                           torch.tensor(1.0/self.width, device=t.device),
    #                           torch.tensor(0.0, device=t.device))

    #     # 2. Unmasking Flow (Beta)
    #     prog_unmask = (t - (t_start + self.lag)) / self.width
    #     beta = torch.clamp(prog_unmask, 0.0, 1.0)
        
    #     d_beta = torch.where((prog_unmask > 0) & (prog_unmask < 1),
    #                          torch.tensor(1.0/self.width, device=t.device),
    #                          torch.tensor(0.0, device=t.device))
        
    #     # Constraint
    #     beta = torch.min(alpha, beta)
        
    #     # 3. Calculate Rates (Psi)
    #     psi_ins = d_alpha / torch.clamp(1.0 - alpha, min=1e-6)
    #     psi_ins = psi_ins * ((prog_ins > 0) & (prog_ins < 1.0)).float()
        
    #     denom_rev = alpha - beta
    #     psi_rev = d_beta / torch.clamp(denom_rev, min=1e-6)
    #     psi_rev = psi_rev * ((prog_unmask > 0) & (prog_unmask < 1.0)).float()
        
    #     return alpha, beta, psi_ins, psi_rev

    def get_coeffs_and_rates(self, t, depths):
        if isinstance(t, float): t = torch.tensor(t)
        t = t.to(depths.device)
        d = depths.float()
        
        # [Strict Time Window]
        # t_start must end before 1.0 to ensure completion
        t_start = (d / self.max_depth) 
        
        # --- 1. Insertion Flow (Alpha) ---
        # Linear Ramp: 0 -> 1 over [t_start, t_start + width]
        numer_ins = t - t_start
        prog_ins = numer_ins / self.width
        alpha = torch.clamp(prog_ins, 0.0, 1.0)
        
        # d_alpha/dt (Exact Derivative)
        d_alpha = torch.where((prog_ins > 0) & (prog_ins < 1),
                              torch.tensor(1.0/self.width, device=t.device),
                              torch.tensor(0.0, device=t.device))

        # --- 2. Unmasking Flow (Beta) ---
        # Delayed Ramp: 0 -> 1 over [t_start + lag, t_start + lag + width]
        numer_unmask = t - (t_start + self.lag)
        prog_unmask = numer_unmask / self.width
        beta_raw = torch.clamp(prog_unmask, 0.0, 1.0)
        
        d_beta_raw = torch.where((prog_unmask > 0) & (prog_unmask < 1),
                                 torch.tensor(1.0/self.width, device=t.device),
                                 torch.tensor(0.0, device=t.device))
        
        # Constraint: Beta cannot exceed Alpha
        beta = torch.min(alpha, beta_raw)
        
        # --- 3. Rigorous Rate Calculation ---
        # Rate = d(State)/dt / (1 - P(State))
        
        # Psi_ins (Insertion Rate): Conditional on Empty(1-alpha)
        # alpha increases from 0 to 1. 
        # singularity at alpha=1 handled by safe division or clamping rate.
        denom_ins = torch.clamp(1.0 - alpha, min=1e-5)
        psi_ins = d_alpha / denom_ins
        
        # Psi_rev (Unmasking Rate): Conditional on Mask(alpha-beta)
        # This drives Mask -> Real.
        # If beta_raw > alpha, it means we SHOULD be real, but alpha is limiting.
        # Ideally this case implies instantaneous unmasking (Rate -> Inf).
        
        denom_rev = alpha - beta
        
        # Case A: Normal flow (beta_raw < alpha)
        # Rate is governed by d_beta_raw
        
        # Case B: Clamped (beta_raw >= alpha)
        # Rate is governed by d_alpha (immediate pass-through)
        # But mathematically, if beta is stuck at alpha, the "Mask" mass is 0.
        # Transition rate is effectively infinite for any newly created mask.
        
        # We calculate rate based on beta_raw's target trajectory
        denom_rev_safe = torch.clamp(denom_rev, min=1e-5)
        
        # If we are in the active unmasking window, we want a rate.
        # Using d_beta_raw gives the "planned" unmasking speed.
        psi_rev = d_beta_raw / denom_rev_safe
        
        # Zero out rates outside active regions to prevent numerical noise
        psi_ins = torch.where(prog_ins < 1.0, psi_ins, torch.zeros_like(psi_ins))
        psi_rev = torch.where(prog_unmask < 1.0, psi_rev, torch.zeros_like(psi_rev))
        
        return alpha, beta, psi_ins, psi_rev

# ==========================================
# 2. Data & Utils (Restored)
# ==========================================
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

# [Visualizer Restored]
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
            # Sample using the new vectorized method
            try:
                trees = pl_module.sample(num_samples=3, steps=500, max_nodes=120) 
                avg_size = sum(len(t) for t in trees) / len(trees)
                self.avg_size_history.append((current_epoch, avg_size))
                epoch_dir = os.path.join(self.save_dir, f"epoch_{current_epoch}")
                os.makedirs(epoch_dir, exist_ok=True)
                for idx, nodes in enumerate(trees):
                    TreeUtils.save_tree_plot(nodes, os.path.join(epoch_dir, f"sample_{idx}.png"), title=f"Sz {len(nodes)}")
                self._plot_metrics()
            except Exception as e:
                print(f"[Visualizer] Error during sampling: {e}")

    def _plot_metrics(self):
        if not self.loss_history: return
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(self.loss_history, color='blue', label='Train Loss')
        ax1.set_ylabel('Loss', color='blue')
        ax1.set_xlabel('Epoch')
        if self.avg_size_history:
            ax2 = ax1.twinx()
            ep, sz = zip(*self.avg_size_history)
            ax2.plot(ep, sz, color='red', marker='o', label='Avg Tree Size')
            ax2.set_ylabel('Size', color='red')
        plt.title("Training Metrics")
        plt.savefig(os.path.join(self.save_dir, "loss_size_plot.png")); plt.close()

# ==========================================
# 3. Model Architecture
# ==========================================
class TreeTransformer(nn.Module):
    def __init__(self, num_types, k, d_model=256, n_heads=4, n_layers=6, max_depth=120):
        super().__init__()
        self.k = k
        self.num_types = num_types
        self.mask_token_id = num_types + 1 # ID for Mask
        
        self.type_emb = nn.Embedding(num_types + 2, d_model, padding_idx=0)
        self.depth_emb = nn.Embedding(max_depth + 2, d_model)
        self.rank_emb = nn.Embedding(k + 1, d_model)
        self.time_mlp = nn.Sequential(nn.Linear(1, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model*2, dropout=0.1, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=n_layers)
        
        # Head predicts 0..C (Not Mask)
        self.head = nn.Linear(d_model, k * (num_types + 1))

    def forward(self, x, mask, t):
        h = self.type_emb(x[:,:,2]) + self.depth_emb(x[:,:,0].clamp(max=150)) + self.rank_emb(x[:,:,1].clamp(max=self.k))
        if t.dim() == 1: t_emb = self.time_mlp(t.view(-1, 1)).unsqueeze(1)
        else: t_emb = self.time_mlp(t.unsqueeze(-1))
        h = h + t_emb
        out = self.transformer(h, src_key_padding_mask=mask)
        return self.head(out).view(out.size(0), out.size(1), self.k, -1)

# ==========================================
# 4. Vectorized Mask-First DFM (Updated)
# ==========================================
class FlexTreeDFM(pl.LightningModule):
    def __init__(self, num_types=4, k=3, max_depth=120, lr=1e-3, pos_weight=50.0):
        super().__init__()
        self.save_hyperparameters()
        self.model = TreeTransformer(num_types, k, max_depth=max_depth)
        self.scheduler = DualStratifiedScheduler(max_depth)
        self.mask_token_id = num_types + 1
        
        self.loss_weights = torch.ones(num_types + 1)
        self.loss_weights[0] = 1.0
        self.loss_weights[1:] = pos_weight

    def training_step(self, batch, batch_idx):
        x_1, padding_mask = batch
        bs, seq_len, _ = x_1.size()
        parents = x_1[:, :, 3].clone(); parents[padding_mask] = -100
        depths = x_1[:, :, 0]
        
        # Time Jittering
        t_global = torch.rand(bs, device=self.device).view(-1, 1)
        t_jitter = torch.randn(bs, seq_len, device=self.device) * 0.1 
        t_node = torch.clamp(t_global + t_jitter, 0.0, 1.0)
        
        # Get Flows
        alpha, beta, _, _ = self.scheduler.get_coeffs_and_rates(t_node, depths)
        
        # Construct x_t
        u = torch.rand_like(alpha)
        x_t_types = torch.zeros_like(x_1[:,:,2])
        
        is_real = u < beta
        is_mask = (u >= beta) & (u < alpha)
        
        x_t_types[is_real] = x_1[:,:,2][is_real]
        x_t_types[is_mask] = self.mask_token_id
        
        # Structural Constraint
        exists_mask = x_t_types > 0
        exists_mask[:, 0] = True # Root always exists
        
        # 부모 인덱스 정리 (-100이나 -1은 0번 인덱스로 매핑해서 에러 방지)
        safe_parents = parents.clone()
        safe_parents[safe_parents < 0] = 0
        
        # 트리의 깊이만큼만 반복하면 모든 연결성 확인 가능 (보통 20~50회)
        # N=2000번 도는 것보다 훨씬 빠름
        for _ in range(self.hparams.max_depth):
            # 부모가 살아있는지 확인 (Batch Gather)
            parent_exists = exists_mask.gather(1, safe_parents)
            
            # 조건: "부모가 없는데(False) 부모가 존재해야 하는 노드(parents>=0)라면 죽인다"
            # 즉, (부모가 없거나 Root임) OR (부모가 살아있음)
            has_parent = parents >= 0
            condition = (~has_parent) | parent_exists
            
            new_mask = exists_mask & condition
            
            # 더 이상 변하는 게 없으면 조기 종료
            if torch.equal(new_mask, exists_mask):
                break
            exists_mask = new_mask
            
        x_t_types[~exists_mask] = 0 # 마스크 반영
        
        # Forward
        x_t = x_1.clone()
        x_t[:, :, 2] = x_t_types
        model_input_mask = ~exists_mask | padding_mask
        
        logits = self.model(x_t, model_input_mask, t_node)
        
        # Target
        target = torch.zeros(bs, seq_len, self.hparams.k, dtype=torch.long, device=self.device)
        
        # 모든 인덱스 준비 (B, N)
        batch_idx = torch.arange(bs, device=self.device).view(-1, 1).expand(-1, seq_len)
        node_idx = torch.arange(seq_len, device=self.device).view(1, -1).expand(bs, -1)
        
        p_idx = parents
        r_idx = x_1[:, :, 1] # Rank
        type_val = x_1[:, :, 2] # Type
        
        # 유효한 타겟 조건 (for문 안의 if 조건들을 텐서로 변환)
        # 1. 패딩 아님
        # 2. 부모 인덱스 유효 (>=0)
        # 3. 부모가 exists_mask에 살아있음
        # 4. Rank 유효 (< k)
        
        safe_p = p_idx.clone(); safe_p[safe_p < 0] = 0
        parent_exists_in_mask = exists_mask.gather(1, safe_p)
        
        valid_condition = (~padding_mask) & (p_idx >= 0) & parent_exists_in_mask & (r_idx < self.hparams.k)
        
        # 조건에 맞는 인덱스만 골라냄 (Filtering)
        b_sel = batch_idx[valid_condition]
        p_sel = p_idx[valid_condition]
        r_sel = r_idx[valid_condition]
        val_sel = type_val[valid_condition]
        
        # 한 번에 할당 (Scatter)
        target.index_put_((b_sel, p_sel, r_sel), val_sel)
                        
        active_parent_mask = exists_mask & (~padding_mask)
        
        # (B, N, K, C+1) -> (M, C+1)
        # active_parent_mask인 노드들의 logits만 가져옴
        relevant_logits = logits[active_parent_mask] # (M, K, C+1)
        relevant_target = target[active_parent_mask]
        
        loss = F.cross_entropy(
            relevant_logits.view(-1, self.hparams.num_types + 1),
            relevant_target.view(-1),
            weight=self.loss_weights.to(self.device)
        )
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self): 
        return torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr)

    @torch.no_grad()
    def sample(self, num_samples=1, steps=500, max_nodes=120):
        self.eval(); device = self.device
        
        # Pre-allocate (Vectorized)
        generated = torch.zeros((num_samples, max_nodes, 4), dtype=torch.long, device=device)
        
        # Init Root
        generated[:, 0, 0] = 0; generated[:, 0, 1] = 0
        generated[:, 0, 2] = 2; generated[:, 0, 3] = -1

        filled_mask = torch.zeros((num_samples, max_nodes, self.hparams.k), dtype=torch.bool, device=device)
        
        counts = torch.ones(num_samples, dtype=torch.long, device=device)
        dt = 1.0 / steps
        
        for s in range(steps):
            t_val = s * dt
            if counts.max() >= max_nodes - self.hparams.k: break

            max_len = counts.max().item()
            seq_range = torch.arange(max_len, device=device).unsqueeze(0)
            padding_mask = seq_range >= counts.unsqueeze(1)
            
            x_curr = generated[:, :max_len]
            t_tensor = torch.full((num_samples, max_len), t_val, device=device)
            
            logits = self.model(x_curr, padding_mask, t_tensor)
            probs = F.softmax(logits, dim=-1)
            
            p_depths = x_curr[:, :, 0]
            c_depths = p_depths + 1
            _, _, psi_ins, psi_rev = self.scheduler.get_coeffs_and_rates(t_tensor, c_depths)
            
            psi_ins = psi_ins.unsqueeze(-1).expand(-1, -1, self.hparams.k)
            psi_rev = psi_rev.unsqueeze(-1).expand(-1, -1, self.hparams.k)
            
            # 1. Insertion Flow
            p_exist = 1.0 - probs[..., 0]
            spawn_probs = psi_ins * p_exist * dt

            current_filled = filled_mask[:, :max_len, :]
            spawn_probs[current_filled] = 0
            spawn_probs[padding_mask] = 0 
            
            should_insert = torch.rand_like(spawn_probs) < spawn_probs
            insert_indices = torch.nonzero(should_insert) 
            
            # 2. Unmasking Flow
            mask_indices = torch.nonzero(x_curr[:, :, 2] == self.mask_token_id)
            if len(mask_indices) > 0:
                b_m = mask_indices[:, 0]; c_m = mask_indices[:, 1]
                p_m = generated[b_m, c_m, 3]; k_m = generated[b_m, c_m, 1]
                
                rates = psi_rev[b_m, p_m, k_m]
                rel_probs = probs[b_m, p_m, k_m]
                p_real_sum = 1.0 - rel_probs[:, 0]
                
                reveal_prob = rates * p_real_sum * dt
                should_reveal = torch.rand_like(reveal_prob) < reveal_prob
                
                if should_reveal.any():
                    final_b = b_m[should_reveal]
                    final_c = c_m[should_reveal]
                    final_probs = rel_probs[should_reveal, 1:]
                    
                    if final_probs.sum() > 0:
                        final_types = torch.multinomial(final_probs, 1).squeeze(-1) + 1
                        generated[final_b, final_c, 2] = final_types

            # 3. Apply Insertion
            if insert_indices.shape[0] > 0:
                b_i = insert_indices[:, 0]; p_i = insert_indices[:, 1]; k_i = insert_indices[:, 2]
                
                unique_b, counts_b = torch.unique(b_i, return_counts=True)
                sort_order = torch.argsort(b_i)
                b_i = b_i[sort_order]; p_i = p_i[sort_order]; k_i = k_i[sort_order]
                
                start_ptr = 0
                for i, b in enumerate(unique_b):
                    
                    cnt = counts_b[i].item()
                    curr_cnt = counts[b].item()
                    
                    if curr_cnt + cnt > max_nodes: cnt = max_nodes - curr_cnt
                    if cnt <= 0: start_ptr += counts_b[i].item(); continue
                    
                    target_indices = torch.arange(curr_cnt, curr_cnt + cnt, device=device)
                    
                    real_p_i = p_i[start_ptr:start_ptr+cnt]
                    real_k_i = k_i[start_ptr:start_ptr+cnt]
                    
                    # [FIX] 실제로 추가된 노드들만 filled_mask 업데이트
                    filled_mask[b, real_p_i, real_k_i] = True

                    # Insert as MASK
                    generated[b, target_indices, 0] = x_curr[b, p_i[start_ptr], 0] + 1
                    generated[b, target_indices, 1] = k_i[start_ptr:start_ptr+cnt]
                    generated[b, target_indices, 2] = self.mask_token_id 
                    generated[b, target_indices, 3] = p_i[start_ptr:start_ptr+cnt]
                    
                    counts[b] += cnt
                    start_ptr += counts_b[i].item()

        # Convert to list
        final_trees = []
        for b in range(num_samples):
            valid_len = counts[b].item()
            final_trees.append(generated[b, :valid_len].tolist())
        return final_trees

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl_path", type=str, default="maze_trees_relative.pkl")
    parser.add_argument("--save_dir", type=str, default="dfm_logs_mask_vectttt")
    parser.add_argument("--max_depth", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--pos_weight", type=float, default=1.0)
    args = parser.parse_args()
    
    pl.seed_everything(42)
    if not os.path.exists(args.pkl_path): return
    
    ds = MazeTreeDataset(args.pkl_path, max_depth=args.max_depth)
    dl = DataLoader(ds, batch_size=args.batch_size, collate_fn=distinct_collate_fn, shuffle=True, num_workers=8)
    
    model = FlexTreeDFM(max_depth=args.max_depth, lr=args.lr, pos_weight=args.pos_weight)
    vis_cb = TrainingVisualizer(save_dir=args.save_dir)
    
    trainer = pl.Trainer(max_epochs=args.epochs, accelerator="auto", callbacks=[vis_cb], log_every_n_steps=10)
    trainer.fit(model, dl)

if __name__ == "__main__":
    main()