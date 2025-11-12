#!/usr/bin/env python3
"""
Decision Transformer for Indoor Localization - 올바른 구현
각 timestep의 위치를 예측
"""
import numpy as np
import pickle
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math

print("="*70)
print("🚀 Decision Transformer 학습 (Proper Implementation)")
print("="*70)

# ============================================================================
# Configuration
# ============================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n사용 디바이스: {DEVICE}")

CONFIG = {
    'context_length': 20,      # K: 최근 20 timesteps만 사용
    'n_layer': 3,
    'n_head': 4,
    'n_embd': 128,
    'dropout': 0.1,
    'batch_size': 64,
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'epochs': 50,
    'warmup_steps': 5000,
}

# ============================================================================
# Dataset
# ============================================================================
class DTDataset(Dataset):
    """
    Decision Transformer용 데이터셋

    각 샘플: 100 timesteps → 마지막 K timesteps만 사용
    """
    def __init__(self, states, trajectories, rtg, context_length=20):
        """
        Args:
            states: (N, 100, 6)
            trajectories: (N, 100, 2)
            rtg: (N, 100, 1)
        """
        self.states = torch.FloatTensor(states)
        self.trajectories = torch.FloatTensor(trajectories)
        self.rtg = torch.FloatTensor(rtg)
        self.K = context_length

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        # 마지막 K timesteps만 사용
        states = self.states[idx, -self.K:]       # (K, 6)
        actions = self.trajectories[idx, -self.K:]  # (K, 2)
        rtg = self.rtg[idx, -self.K:]             # (K, 1)
        timesteps = torch.arange(self.K, dtype=torch.long)

        return {
            'states': states,
            'actions': actions,
            'rtg': rtg,
            'timesteps': timesteps,
        }

# ============================================================================
# Model (동일한 구조)
# ============================================================================

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config['n_embd'] % config['n_head'] == 0

        self.c_attn = nn.Linear(config['n_embd'], 3 * config['n_embd'])
        self.c_proj = nn.Linear(config['n_embd'], config['n_embd'])
        self.attn_dropout = nn.Dropout(config['dropout'])
        self.resid_dropout = nn.Dropout(config['dropout'])
        self.n_head = config['n_head']
        self.n_embd = config['n_embd']

        self.register_buffer("bias", torch.tril(torch.ones(1024, 1024))
                                     .view(1, 1, 1024, 1024))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.resid_dropout(self.c_proj(y))
        return y

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config['n_embd'])
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config['n_embd'])
        self.mlp = nn.Sequential(
            nn.Linear(config['n_embd'], 4 * config['n_embd']),
            nn.GELU(),
            nn.Linear(4 * config['n_embd'], config['n_embd']),
            nn.Dropout(config['dropout']),
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class DecisionTransformer(nn.Module):
    def __init__(self, state_dim, action_dim, config):
        super().__init__()
        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Token embeddings
        self.embed_timestep = nn.Embedding(1024, config['n_embd'])
        self.embed_return = nn.Linear(1, config['n_embd'])
        self.embed_state = nn.Linear(state_dim, config['n_embd'])
        self.embed_action = nn.Linear(action_dim, config['n_embd'])

        self.embed_ln = nn.LayerNorm(config['n_embd'])

        # Transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config['n_layer'])])

        # Prediction head
        self.predict_action = nn.Sequential(
            nn.Linear(config['n_embd'], config['n_embd']),
            nn.ReLU(),
            nn.Linear(config['n_embd'], action_dim)
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, states, actions, returns_to_go, timesteps):
        """
        Args:
            states: (B, K, state_dim)
            actions: (B, K, action_dim)
            returns_to_go: (B, K, 1)
            timesteps: (B, K)

        Returns:
            action_preds: (B, K, action_dim)
        """
        B, K, _ = states.shape

        # Embed
        time_embeddings = self.embed_timestep(timesteps)
        state_embeddings = self.embed_state(states) + time_embeddings
        action_embeddings = self.embed_action(actions) + time_embeddings
        returns_embeddings = self.embed_return(returns_to_go) + time_embeddings

        # Stack: (R, s, a, R, s, a, ...)
        stacked_inputs = torch.stack(
            [returns_embeddings, state_embeddings, action_embeddings],
            dim=2
        ).reshape(B, 3*K, self.config['n_embd'])

        stacked_inputs = self.embed_ln(stacked_inputs)

        # Transformer
        x = self.blocks(stacked_inputs)

        # Predict actions from state tokens
        x = x.reshape(B, K, 3, self.config['n_embd'])[:, :, 1]
        action_preds = self.predict_action(x)

        return action_preds

# ============================================================================
# Training
# ============================================================================

def train():
    print("\n[1/5] 데이터 로드...")

    data_dir = Path(__file__).parent.parent / 'processed_data_dt'

    states_train = np.load(data_dir / 'states_train.npy')
    traj_train = np.load(data_dir / 'trajectories_train.npy')
    rtg_train = np.load(data_dir / 'rtg_train.npy')

    states_val = np.load(data_dir / 'states_val.npy')
    traj_val = np.load(data_dir / 'trajectories_val.npy')
    rtg_val = np.load(data_dir / 'rtg_val.npy')

    print(f"  Train: {states_train.shape}")
    print(f"  Val:   {states_val.shape}")

    print("\n[2/5] 데이터셋 생성...")
    train_dataset = DTDataset(states_train, traj_train, rtg_train, CONFIG['context_length'])
    val_dataset = DTDataset(states_val, traj_val, rtg_val, CONFIG['context_length'])

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'],
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'],
                            shuffle=False, num_workers=0)

    print("\n[3/5] 모델 초기화...")
    model = DecisionTransformer(
        state_dim=6,
        action_dim=2,
        config=CONFIG
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  총 파라미터: {n_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )

    # Scheduler
    def lr_lambda(step):
        if step < CONFIG['warmup_steps']:
            return step / CONFIG['warmup_steps']
        else:
            return 0.5 * (1 + math.cos(math.pi * (step - CONFIG['warmup_steps']) /
                                       (CONFIG['epochs'] * len(train_loader) - CONFIG['warmup_steps'])))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print("\n[4/5] 학습 시작...")
    print(f"  Epochs: {CONFIG['epochs']}")
    print(f"  Batch size: {CONFIG['batch_size']}")
    print(f"  Train batches: {len(train_loader)}")

    best_val_loss = float('inf')

    for epoch in range(CONFIG['epochs']):
        # Training
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            states = batch['states'].to(DEVICE)
            actions = batch['actions'].to(DEVICE)
            rtg = batch['rtg'].to(DEVICE)
            timesteps = batch['timesteps'].to(DEVICE)

            # Forward
            action_preds = model(states, actions, rtg, timesteps)

            # Loss: 모든 timestep의 action 예측
            loss = F.mse_loss(action_preds, actions)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                states = batch['states'].to(DEVICE)
                actions = batch['actions'].to(DEVICE)
                rtg = batch['rtg'].to(DEVICE)
                timesteps = batch['timesteps'].to(DEVICE)

                action_preds = model(states, actions, rtg, timesteps)
                loss = F.mse_loss(action_preds, actions)

                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_dir = Path(__file__).parent.parent / 'models'
            model_dir.mkdir(exist_ok=True)

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'config': CONFIG,
            }, model_dir / 'dt_proper_best.pt')

    print("\n[5/5] 학습 완료!")
    print(f"  최고 Val Loss: {best_val_loss:.4f}")
    print(f"  모델 저장: models/dt_proper_best.pt")

if __name__ == '__main__':
    train()
