import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config

class ActionEmbedding(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.action_emb = nn.Embedding(config.action_vocab_size, config.action_emb_dim)
        self.bert_proj = nn.Linear(768, config.desc_emb_dim)  # 768为BERT输出维度
        self.fc = nn.Linear(config.action_emb_dim + config.desc_emb_dim + 1, config.patch_emb_dim)

    def forward(self, action_name_ids, if_anchor, desc_vecs):
        # action_name_ids: [B, L]
        # if_anchor: [B, L]
        # desc_vecs: [B, L, 768]
        a = self.action_emb(action_name_ids)  # [B, L, D]
        d = self.bert_proj(desc_vecs)         # [B, L, desc_emb_dim]
        anchor = if_anchor.unsqueeze(-1).float()  # [B, L, 1]
        x = torch.cat([a, d, anchor], dim=-1)
        return self.fc(x)

class PatchEncoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.action_emb = ActionEmbedding(config)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=config.patch_emb_dim, nhead=2),
            num_layers=1
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.patch_emb_dim))

    def forward(self, patch_seq, desc_vecs):
        # patch_seq: [B, L, 3]
        # desc_vecs: [B, L, 768]
        action_name_ids = patch_seq[:,:,0]
        if_anchor = patch_seq[:,:,1]
        x = self.action_emb(action_name_ids, if_anchor, desc_vecs)  # [B, L, D]
        cls = self.cls_token.expand(x.size(0), 1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x.transpose(0,1)  # [L+1, B, D]
        x = self.encoder(x)
        return x[0]  # [B, D]  # [CLS] token

class LiveRiskModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.patch_encoder = PatchEncoder(config)
        self.user_pos_emb = nn.Embedding(config.max_users, config.patch_emb_dim)
        self.time_pos_emb = nn.Embedding(config.max_timeslots, config.patch_emb_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.patch_emb_dim))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=config.patch_emb_dim, nhead=4),
            num_layers=2
        )
        self.head = nn.Linear(config.patch_emb_dim, config.num_classes)

    def forward(self, patch_grid, desc_texts_grid, patch_mask=None):
        # patch_grid: [B, U, T, L, 3]
        # desc_texts_grid: [B, U, T, L, 768]
        B, U, T, L, _ = patch_grid.shape
        patch_embs = []
        for b in range(B):
            user_embs = []
            for u in range(U):
                time_embs = []
                for t in range(T):
                    patch = patch_grid[b, u, t]         # [L, 3]
                    desc_vecs = desc_texts_grid[b, u, t] # [L, 768]
                    patch = patch.unsqueeze(0)          # [1, L, 3]
                    desc_vecs = desc_vecs.unsqueeze(0)  # [1, L, 768]
                    patch_emb = self.patch_encoder(patch, desc_vecs)  # [1, D]
                    time_embs.append(patch_emb.squeeze(0))
                user_embs.append(torch.stack(time_embs))
            patch_embs.append(torch.stack(user_embs))
        patch_embs = torch.stack(patch_embs)  # [B, U, T, D]

        # 位置编码
        device = patch_embs.device
        user_idx, time_idx = torch.meshgrid(
            torch.arange(U, device=device), torch.arange(T, device=device), indexing='ij'
        )
        pos = self.user_pos_emb(user_idx) + self.time_pos_emb(time_idx)  # [U, T, D]
        pos = pos.unsqueeze(0).expand(B, -1, -1, -1)
        x = patch_embs + pos  # [B, U, T, D]
        x = x.view(B, U*T, -1)  # [B, U*T, D]
        cls = self.cls_token.expand(B, 1, -1)
        x = torch.cat([cls, x], dim=1)  # [B, 1+U*T, D]
        x = x.transpose(0,1)  # [1+U*T, B, D]
        x = self.transformer(x)
        out = self.head(x[0])  # [B, num_classes]
        return out 