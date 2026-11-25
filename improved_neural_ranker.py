import json, time
from pathlib import Path
import numpy as np
import polars as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from loguru import logger
import sys

# Category dictionary size
NUM_CATEGORIES = 54

class CandidateMLP(nn.Module):
    def __init__(self, in_features, width, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, width)
        self.norm1 = nn.LayerNorm(width)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)

    def forward(self, x):
        h = self.drop(self.act(self.norm1(self.fc1(x))))
        out = self.norm2(self.fc2(h) + h)
        return out

class UserMLP(nn.Module):
    def __init__(self, in_features, width, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, width)
        self.norm = nn.LayerNorm(width)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(width, width)
        self.out_norm = nn.LayerNorm(width)

    def forward(self, x):
        h = self.drop(self.act(self.norm(self.fc1(x))))
        return self.out_norm(self.fc2(h))

class ImprovedTransformerRanker(nn.Module):
    def __init__(self, in_dense_features, width=128, layers=3, heads=8, dropout=0.1, cat_dim=16):
        super().__init__()
        self.width = width
        self.category_embed = nn.Embedding(NUM_CATEGORIES + 1, cat_dim, padding_idx=0)
        
        # Candidate MLP projector
        cand_in_dim = in_dense_features + cat_dim
        self.cand_mlp = CandidateMLP(cand_in_dim, width, dropout=dropout)
        
        # User projector
        self.user_mlp = UserMLP(128, width, dropout=dropout)
        
        # Cross-Attention: Candidates query the User representation
        self.cross_attn = nn.MultiheadAttention(embed_dim=width, num_heads=heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(width)
        
        # Self-Attention: Candidates compare against each other in the feed
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=4 * width,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        
        # Scoring head
        self.score_head = nn.Sequential(
            nn.Linear(width, width // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width // 2, 1)
        )

    def forward(self, dense_feats: torch.Tensor, cat_ids: torch.Tensor, users: torch.Tensor, padding: torch.Tensor) -> torch.Tensor:
        """
        dense_feats: [B, M, D]
        cat_ids: [B, M] (int64)
        users: [B, 128]
        padding: [B, M] (bool, True = padding)
        Returns: logits [B, M]
        """
        B, M, _ = dense_feats.shape
        cat_emb = self.category_embed(cat_ids) # [B, M, cat_dim]
        cand_input = torch.cat([dense_feats, cat_emb], dim=-1)
        cands = self.cand_mlp(cand_input) # [B, M, width]
        
        # User representation [B, 1, width]
        user_rep = self.user_mlp(users).unsqueeze(1)
        
        # Cross-Attention (Candidates query User)
        cand_user_ctx, _ = self.cross_attn(query=cands, key=user_rep, value=user_rep)
        cands = self.cross_norm(cands + cand_user_ctx)
        
        # Self-Attention across candidates with padding mask
        context = self.encoder(cands, src_key_padding_mask=padding) # [B, M, width]
        
        # Raw ranking logits [B, M]
        logits = self.score_head(context).squeeze(-1)
        return logits

def listwise_loss_fn(logits: torch.Tensor, targets: torch.Tensor, padding: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Combines ListNet Softmax Cross-Entropy over positive candidates with auxiliary BCE loss.
    """
    masked_logits = logits.masked_fill(padding, -1e9)
    log_probs = F.log_softmax(masked_logits / temperature, dim=-1)
    
    pos_targets = (targets > 0).float()
    feed_pos_counts = pos_targets.sum(dim=-1, keepdim=True)
    valid_feeds = (feed_pos_counts.squeeze(-1) > 0)
    if valid_feeds.any():
        true_dist = (pos_targets / feed_pos_counts.clamp(min=1.0))[valid_feeds]
        listwise_loss = -(true_dist * log_probs[valid_feeds]).sum(dim=-1).mean()
    else:
        listwise_loss = torch.tensor(0.0, device=logits.device)
        
    valid_cands = ~padding
    bce_loss = F.binary_cross_entropy_with_logits(logits[valid_cands], targets[valid_cands])
    return listwise_loss + 0.3 * bce_loss

def distillation_loss_fn(
    logits: torch.Tensor,
    targets: torch.Tensor,
    padding: torch.Tensor,
    teacher_logits: torch.Tensor = None,
    temperature: float = 0.10,
    alpha_teacher: float = 0.5,
) -> torch.Tensor:
    """
    Phase Slate InfoNCE Loss (tau=0.10) + Softmax Distillation from teacher CatBoost.
    """
    masked_logits = logits.masked_fill(padding, -1e9)
    log_probs = F.log_softmax(masked_logits / temperature, dim=-1)
    
    pos_targets = (targets > 0).float()
    feed_pos_counts = pos_targets.sum(dim=-1, keepdim=True)
    valid_feeds = (feed_pos_counts.squeeze(-1) > 0)
    if valid_feeds.any():
        true_dist = (pos_targets / feed_pos_counts.clamp(min=1.0))[valid_feeds]
        listwise_loss = -(true_dist * log_probs[valid_feeds]).sum(dim=-1).mean()
    else:
        listwise_loss = torch.tensor(0.0, device=logits.device)
        
    if teacher_logits is not None and alpha_teacher > 0:
        masked_teacher = teacher_logits.masked_fill(padding, -1e9)
        teacher_probs = F.softmax(masked_teacher / temperature, dim=-1)
        distill_loss = -(teacher_probs * log_probs).sum(dim=-1).mean()
        main_loss = (1.0 - alpha_teacher) * listwise_loss + alpha_teacher * distill_loss
    else:
        main_loss = listwise_loss
        
    valid_cands = ~padding
    bce_loss = F.binary_cross_entropy_with_logits(logits[valid_cands], (targets[valid_cands] > 0).float())
    return main_loss + 0.2 * bce_loss
