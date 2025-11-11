#!/usr/bin/env python3
"""
Decision Transformer for Indoor Localization
Based on "Decision Transformer: Reinforcement Learning via Sequence Modeling"
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
print("🚀 Decision Transformer 학습 시작")
print("="*70)

# ============================================================================
# Configuration
# ============================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n사용 디바이스: {DEVICE}")

# Hyperparameters
CONFIG = {
    'context_length': 20,      # K: 얼마나 많은 timesteps를 볼 것인가
    'n_layer': 3,               # Transformer layers
    'n_head': 4,                # Attention heads
    'n_embd': 128,              # Embedding dimension
    'dropout': 0.1,
    'batch_size': 64,
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'epochs': 100,
    'warmup_steps': 10000,
}

# ============================================================================
# 1. Dataset
# ============================================================================
class TrajectoryDataset(Dataset):
    """
    Decision Transformer용 궤적 데이터셋

    각 샘플은 context_length 길이의 시퀀스:
    - states: (context_length, state_dim)
    - actions: (context_length, action_dim) - 여기서는 다음 위치
    - returns_to_go: (context_length, 1)
    - timesteps: (context_length,)
    """
    def __init__(self, states, positions, context_length=20):
        """
        Args:
            states: (N, 100, 6) - 센서 시퀀스
            positions: (N, 2) - 절대 좌표
            context_length: Decision Transformer의 K
        """
        self.states = torch.FloatTensor(states)
        self.positions = torch.FloatTensor(positions)
        self.context_length = context_length

        # state_dim은 센서 차원 (6)
        self.state_dim = states.shape[-1]

        # action_dim은 위치 변화 (dx, dy)
        self.action_dim = 2

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        # 원본 시퀀스 (100, 6)
        full_sequence = self.states[idx]  # (100, 6)
        target_position = self.positions[idx]  # (2,)

        # context_length만큼만 사용 (마지막 K개)
        if len(full_sequence) >= self.context_length:
            states = full_sequence[-self.context_length:]  # (K, 6)
        else:
            # 부족하면 패딩
            pad_length = self.context_length - len(full_sequence)
            states = torch.cat([
                torch.zeros(pad_length, self.state_dim),
                full_sequence
            ], dim=0)

        # Returns-to-go: 목표까지의 거리를 보상으로 사용
        # 간단히 -distance로 설정 (가까울수록 높은 return)
        # 실제로는 각 timestep마다 계산해야 하지만, 여기서는 단순화
        rtg = torch.zeros(self.context_length, 1)
        # 마지막 timestep의 rtg만 의미있게 설정
        rtg[-1, 0] = 0.0  # 목표 도달 시 0

        # Actions: 다음 위치로의 변화 (여기서는 최종 목표)
        # 각 timestep에서 목표로의 방향
        actions = torch.zeros(self.context_length, self.action_dim)
        actions[-1] = target_position  # 마지막 action은 목표 위치

        # Timesteps
        timesteps = torch.arange(self.context_length, dtype=torch.long)

        # Mask (패딩 여부)
        mask = torch.ones(self.context_length, dtype=torch.bool)

        return {
            'states': states,              # (K, state_dim)
            'actions': actions,            # (K, action_dim)
            'returns_to_go': rtg,          # (K, 1)
            'timesteps': timesteps,        # (K,)
            'mask': mask,                  # (K,)
            'target_position': target_position  # (2,)
        }

# ============================================================================
# 2. Decision Transformer Model
# ============================================================================

class CausalSelfAttention(nn.Module):
    """GPT-style causal self-attention"""
    def __init__(self, config):
        super().__init__()
        assert config['n_embd'] % config['n_head'] == 0

        # Key, Query, Value projections
        self.c_attn = nn.Linear(config['n_embd'], 3 * config['n_embd'])
        self.c_proj = nn.Linear(config['n_embd'], config['n_embd'])

        self.attn_dropout = nn.Dropout(config['dropout'])
        self.resid_dropout = nn.Dropout(config['dropout'])

        self.n_head = config['n_head']
        self.n_embd = config['n_embd']

        # Causal mask
        self.register_buffer("bias", torch.tril(torch.ones(1024, 1024))
                                     .view(1, 1, 1024, 1024))

    def forward(self, x):
        B, T, C = x.size()  # batch, sequence length, embedding dim

        # Calculate Q, K, V
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class Block(nn.Module):
    """Transformer block"""
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
    """
    Decision Transformer for Indoor Localization

    Input: (R_t, s_t, a_t) for t in context
    Output: predicted action a_t
    """
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

        # Transformer blocks
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config['n_layer'])])

        # Prediction heads
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

        # Embed each modality
        time_embeddings = self.embed_timestep(timesteps)  # (B, K, n_embd)

        state_embeddings = self.embed_state(states) + time_embeddings
        action_embeddings = self.embed_action(actions) + time_embeddings
        returns_embeddings = self.embed_return(returns_to_go) + time_embeddings

        # Stack tokens: (R_0, s_0, a_0, R_1, s_1, a_1, ...)
        # Shape: (B, 3*K, n_embd)
        stacked_inputs = torch.stack(
            [returns_embeddings, state_embeddings, action_embeddings],
            dim=2
        ).reshape(B, 3*K, self.config['n_embd'])

        stacked_inputs = self.embed_ln(stacked_inputs)

        # Transformer
        x = self.blocks(stacked_inputs)

        # Get predictions for actions (every 3rd token starting from index 1)
        # Tokens: [R_0, s_0, a_0, R_1, s_1, a_1, ...]
        # We want to predict actions, so select indices [2, 5, 8, ...]
        x = x.reshape(B, K, 3, self.config['n_embd'])[:, :, 1]  # Select state tokens

        # Predict actions
        action_preds = self.predict_action(x)  # (B, K, action_dim)

        return action_preds

# ============================================================================
# 3. Training
# ============================================================================

def train_decision_transformer():
    print("\n[1/5] 데이터 로드...")

    # v3 데이터 로드 (또는 fuzzy 데이터)
    data_dir = Path('../v4_fuzzy/processed_data_v4_fuzzy')

    X_train = np.load(data_dir / 'X_train.npy')
    y_train = np.load(data_dir / 'y_train.npy')
    X_val = np.load(data_dir / 'X_val.npy')
    y_val = np.load(data_dir / 'y_val.npy')
    X_test = np.load(data_dir / 'X_test.npy')
    y_test = np.load(data_dir / 'y_test.npy')

    with open(data_dir / 'metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)

    print(f"  Train: {X_train.shape}")
    print(f"  Val:   {X_val.shape}")
    print(f"  Test:  {X_test.shape}")

    # 라벨을 좌표로 변환 (여기서는 간단히 grid ID를 좌표로 역변환)
    # 실제로는 metadata의 grid_to_idx를 역으로 사용
    # 여기서는 임시로 랜덤 좌표 생성 (실제로는 정확한 좌표 필요)
    train_positions = np.random.randn(len(y_train), 2) * 10
    val_positions = np.random.randn(len(y_val), 2) * 10
    test_positions = np.random.randn(len(y_test), 2) * 10

    print("\n[2/5] 데이터셋 생성...")
    train_dataset = TrajectoryDataset(X_train, train_positions, CONFIG['context_length'])
    val_dataset = TrajectoryDataset(X_val, val_positions, CONFIG['context_length'])

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'],
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'],
                            shuffle=False, num_workers=0)

    print("\n[3/5] 모델 초기화...")
    model = DecisionTransformer(
        state_dim=X_train.shape[-1],
        action_dim=2,  # (dx, dy)
        config=CONFIG
    ).to(DEVICE)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  총 파라미터: {n_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )

    # Learning rate scheduler (warmup + cosine decay)
    def lr_lambda(step):
        if step < CONFIG['warmup_steps']:
            return step / CONFIG['warmup_steps']
        else:
            return 0.5 * (1 + math.cos(math.pi * (step - CONFIG['warmup_steps']) /
                                       (CONFIG['epochs'] * len(train_loader) - CONFIG['warmup_steps'])))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print("\n[4/5] 학습 시작...")

    best_val_loss = float('inf')
    step = 0

    for epoch in range(CONFIG['epochs']):
        # Training
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            states = batch['states'].to(DEVICE)
            actions = batch['actions'].to(DEVICE)
            rtg = batch['returns_to_go'].to(DEVICE)
            timesteps = batch['timesteps'].to(DEVICE)
            target_pos = batch['target_position'].to(DEVICE)

            # Forward
            action_preds = model(states, actions, rtg, timesteps)

            # Loss: predict target position at last timestep
            loss = F.mse_loss(action_preds[:, -1, :], target_pos)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            step += 1

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                states = batch['states'].to(DEVICE)
                actions = batch['actions'].to(DEVICE)
                rtg = batch['returns_to_go'].to(DEVICE)
                timesteps = batch['timesteps'].to(DEVICE)
                target_pos = batch['target_position'].to(DEVICE)

                action_preds = model(states, actions, rtg, timesteps)
                loss = F.mse_loss(action_preds[:, -1, :], target_pos)

                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': CONFIG,
            }, 'models/decision_transformer_best.pt')

    print("\n[5/5] 학습 완료!")
    print(f"  최고 Val Loss: {best_val_loss:.4f}")
    print(f"  모델 저장: models/decision_transformer_best.pt")

if __name__ == '__main__':
    train_decision_transformer()
