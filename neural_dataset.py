import json, time
from pathlib import Path
import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from feature_extractor import BASE_FLOAT_FEATURES, NEW_FLOAT_FEATURES

FLOAT_FEATURES = [*BASE_FLOAT_FEATURES, *NEW_FLOAT_FEATURES]

class FeedRankDataset(Dataset):
    def __init__(self, feeds_df: pl.DataFrame, flat_df: pl.DataFrame, full_item_embs: np.ndarray, teacher_scores: np.ndarray = None, is_train: bool = False):
        """
        feeds_df: contains feed_id and user_embedding (1 row per feed)
        flat_df: exploded table with 51 float features, category_id, target, feed_id
        full_item_embs: [N_cands, 128] float32 normalized item embeddings
        teacher_scores: [N_cands] float32 teacher logits from CatBoost (optional)
        is_train: if True, applies random candidate permutation per feed for permutation-invariance
        """
        self.is_train = is_train
        self.feed_ids = feeds_df["feed_id"].to_numpy()
        
        # User embeddings [N_feeds, 128]
        u_list = feeds_df["user_embedding"].to_list()
        self.users = torch.tensor(np.array(u_list, dtype=np.float32), dtype=torch.float32)
        # Normalize user embeddings
        self.users = torch.nn.functional.normalize(self.users, dim=-1)
        
        # Candidate float features [N_cands, 44]
        scalars = flat_df.select(FLOAT_FEATURES).to_numpy().astype(np.float32)
        # Combine item_embs (128) + scalars (44) = 172 dense features
        combined_dense = np.concatenate([full_item_embs, scalars], axis=1)
        self.dense_feats = torch.tensor(combined_dense, dtype=torch.float32)
        
        # Categories [N_cands]
        cats = flat_df["category_id"].to_numpy().astype(np.int64)
        cats = np.clip(cats, 0, 54)
        self.categories = torch.tensor(cats, dtype=torch.long)
        
        # Targets [N_cands]
        if "target" in flat_df.columns:
            self.targets = torch.tensor(flat_df["target"].to_numpy().astype(np.float32), dtype=torch.float32)
        else:
            self.targets = None
            
        # Teacher scores [N_cands]
        if teacher_scores is not None:
            self.teacher_scores = torch.tensor(teacher_scores.astype(np.float32), dtype=torch.float32)
        else:
            self.teacher_scores = None
            
        # Compute feed candidate offsets
        counts = flat_df.group_by("feed_id", maintain_order=True).len()["len"].to_numpy()
        self.offsets = np.insert(counts.cumsum(), 0, 0)
        assert len(self.offsets) == len(self.users) + 1, "Offsets and feeds mismatch"

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        start = self.offsets[idx]
        end = self.offsets[idx + 1]
        
        cands_dense = self.dense_feats[start:end]
        cands_cat = self.categories[start:end]
        user = self.users[idx]
        target = self.targets[start:end] if self.targets is not None else torch.zeros(end - start)
        teacher = self.teacher_scores[start:end] if self.teacher_scores is not None else torch.zeros(end - start)
        
        if self.is_train and (end - start) > 1:
            perm = torch.randperm(end - start)
            cands_dense = cands_dense[perm]
            cands_cat = cands_cat[perm]
            target = target[perm]
            teacher = teacher[perm]
            
        return cands_dense, cands_cat, user, target, teacher

def collate_feeds(batch):
    """
    Pads batch of feeds to maximum candidate count M_max in the batch.
    """
    cands_dense_list, cands_cat_list, users_list, targets_list, teacher_list = zip(*batch)
    counts = [len(c) for c in cands_dense_list]
    max_cands = max(counts)
    
    padded_dense = pad_sequence(cands_dense_list, batch_first=True, padding_value=0.0)
    padded_cat = pad_sequence(cands_cat_list, batch_first=True, padding_value=0)
    users = torch.stack(users_list, dim=0)
    padded_targets = pad_sequence(targets_list, batch_first=True, padding_value=0.0)
    padded_teachers = pad_sequence(teacher_list, batch_first=True, padding_value=-1e9)
    
    # Padding mask: True where candidate is padding
    padding = torch.arange(max_cands)[None, :] >= torch.tensor(counts)[:, None]
    
    return padded_dense, padded_cat, users, padding, padded_targets, padded_teachers
