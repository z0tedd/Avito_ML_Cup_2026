import json, time
from pathlib import Path
import numpy as np
import polars as pl
from catboost import Pool

CAT_FEATURES = ["category_id"]

BASE_FLOAT_FEATURES = ["f1", "f2", "f3", "f4", "f5", "embedding_dot", "embedding_cosine", "embedding_missing"]

NEW_FLOAT_FEATURES = [
    # Vector geometry
    "embedding_l2",
    "item_norm",
    # User history category interaction
    "hist_len",
    "hist_cat_count",
    "hist_cat_ratio",
    "hist_cat_is_last",
    "hist_cat_in_last5",
    # History Item Embeddings interaction (Top features)
    "hist_item_sim_max",
    "hist_item_sim_mean",
    "hist_item_sim_last",
    "hist_item_sim_diff_max",
    "hist_item_sim_zscore",
    "hist_item_sim_rank",
    # History Event Types & Intent
    "hist_high_intent_count",
    "hist_high_intent_ratio",
    "last_hist_event_type",
    # Feed context
    "feed_size",
    "feed_cat_count",
    "feed_cat_ratio",
    # Within-feed relative features for embedding_cosine
    "cosine_diff_max",
    "cosine_diff_mean",
    "cosine_zscore",
    "cosine_rank",
    # Within-feed relative features for embedding_dot
    "dot_diff_max",
    "dot_diff_mean",
    "dot_zscore",
    # Within-feed relative features for f1..f5
    "f1_diff_max", "f1_diff_mean",
    "f2_diff_max", "f2_diff_mean",
    "f3_diff_max", "f3_diff_mean",
    "f4_diff_max", "f4_diff_mean",
    "f5_diff_max", "f5_diff_mean",
]

ALL_FEATURES = [*BASE_FLOAT_FEATURES, *NEW_FLOAT_FEATURES, *CAT_FEATURES]

def extract_features_from_parquet(parquet_path: Path, embeddings: np.memmap, is_test: bool = False, include_pos: bool = False) -> pl.DataFrame:
    """Reads parquet, computes all 45 interaction, history-embedding & context features."""
    read_cols = [
        'feed_id', 'user_embedding', 'item_id', 'category_id',
        'f1', 'f2', 'f3', 'f4', 'f5', 'item_id__embedding_idx',
        'history__category_id', 'history__item_id__embedding_idx', 'history__event_type'
    ]
    if not is_test:
        read_cols.append('target')
    if include_pos:
        read_cols.append('pos')
        
    df = pl.read_parquet(parquet_path, columns=read_cols)
    
    # 1. Candidate embeddings
    c_idx_flat = df['item_id__embedding_idx'].explode().to_numpy()
    mask_c = c_idx_flat >= 0
    u_c, inv_c = np.unique(c_idx_flat[mask_c], return_inverse=True)
    c_embs_u = embeddings[u_c].astype(np.float32)
    c_embs = np.zeros((len(c_idx_flat), 128), dtype=np.float32)
    c_embs[mask_c] = c_embs_u[inv_c]
    c_norm = np.linalg.norm(c_embs, axis=1, keepdims=True)
    c_embs_normed = np.divide(c_embs, c_norm, out=np.zeros_like(c_embs), where=c_norm > 0)
    
    counts = df['item_id'].list.len().to_numpy()
    u_mat = np.repeat(np.array(df['user_embedding'].to_list(), dtype=np.float32), counts, axis=0)
    
    # User-Candidate vector interactions
    dot = np.sum(c_embs * u_mat, axis=1).astype(np.float32)
    item_norm = c_norm.squeeze(-1).astype(np.float32)
    u_norm = np.linalg.norm(u_mat, axis=1)
    denom = item_norm * u_norm
    cosine = np.divide(dot, denom, out=np.zeros_like(dot), where=denom > 0).astype(np.float32)
    missing = (~mask_c).astype(np.float32)
    l2_dist = np.linalg.norm(c_embs - u_mat, axis=1).astype(np.float32)
    
    # 2. Candidate to User History Item Embeddings interactions
    hist_items_list = df['history__item_id__embedding_idx'].to_list()
    hist_needed = []
    feed_h_slices = []
    for h_items in hist_items_list:
        v = [x for x in (h_items[:5] if h_items is not None else []) if x >= 0]
        feed_h_slices.append(v)
        hist_needed.extend(v)
        
    u_h = np.unique(hist_needed) if len(hist_needed) > 0 else np.array([])
    h_embs_dict = {}
    if len(u_h) > 0:
        h_embs_u = embeddings[u_h].astype(np.float32)
        h_norm = np.linalg.norm(h_embs_u, axis=1, keepdims=True)
        h_normed = np.divide(h_embs_u, h_norm, out=np.zeros_like(h_embs_u), where=h_norm > 0)
        for idx_val, emb_val in zip(u_h, h_normed):
            h_embs_dict[idx_val] = emb_val
            
    sim_max = np.zeros(len(c_idx_flat), dtype=np.float32)
    sim_mean = np.zeros(len(c_idx_flat), dtype=np.float32)
    sim_last = np.zeros(len(c_idx_flat), dtype=np.float32)
    
    offset = 0
    for v_idxs, cnt in zip(feed_h_slices, counts):
        if len(v_idxs) > 0:
            c_chunk = c_embs_normed[offset:offset+cnt]
            h_mat = np.stack([h_embs_dict[x] for x in v_idxs], axis=0)
            sims = np.dot(c_chunk, h_mat.T)
            sim_max[offset:offset+cnt] = sims.max(axis=1)
            sim_mean[offset:offset+cnt] = sims.mean(axis=1)
            sim_last[offset:offset+cnt] = sims[:, 0]
        offset += cnt
        
    # 3. User History category & intent features
    hist_cats_list = df['history__category_id'].to_list()
    cand_cats_list = df['category_id'].to_list()
    hist_events_list = df['history__event_type'].to_list()
    
    flat_hist_len = []
    flat_hist_count = []
    flat_hist_ratio = []
    flat_hist_is_last = []
    flat_hist_in_last5 = []
    flat_high_intent_count = []
    flat_high_intent_ratio = []
    flat_last_event_type = []
    
    for h_cats, c_cats, h_evs in zip(hist_cats_list, cand_cats_list, hist_events_list):
        n_cands = len(c_cats)
        h_len = len(h_cats) if h_cats is not None else 0
        if h_len == 0:
            flat_hist_len.append(np.zeros(n_cands, dtype=np.float32))
            flat_hist_count.append(np.zeros(n_cands, dtype=np.float32))
            flat_hist_ratio.append(np.zeros(n_cands, dtype=np.float32))
            flat_hist_is_last.append(np.zeros(n_cands, dtype=np.float32))
            flat_hist_in_last5.append(np.zeros(n_cands, dtype=np.float32))
            flat_high_intent_count.append(np.zeros(n_cands, dtype=np.float32))
            flat_high_intent_ratio.append(np.zeros(n_cands, dtype=np.float32))
            flat_last_event_type.append(np.zeros(n_cands, dtype=np.float32))
        else:
            h_arr = np.array(h_cats, dtype=np.int64)
            ev_arr = np.array(h_evs, dtype=np.int64) if h_evs is not None else np.zeros(h_len, dtype=np.int64)
            
            valid_mask = (h_arr >= 0) & (h_arr <= 53)
            valid_h = h_arr[valid_mask]
            counts_map = np.bincount(valid_h, minlength=54)
            
            # High intent events (ev_arr >= 2)
            hi_mask = valid_mask & (ev_arr >= 2)
            hi_h = h_arr[hi_mask]
            hi_map = np.bincount(hi_h, minlength=54)
            n_hi = len(hi_h)
            
            c_arr = np.array(c_cats, dtype=np.int64)
            valid_c = np.clip(c_arr, 0, 53)
            c_cnt = counts_map[valid_c].astype(np.float32)
            c_ratio = (c_cnt / float(h_len)).astype(np.float32)
            
            hi_cnt = hi_map[valid_c].astype(np.float32)
            hi_ratio = (hi_cnt / float(n_hi)).astype(np.float32) if n_hi > 0 else np.zeros(n_cands, dtype=np.float32)
            
            last_cat = h_cats[0]
            c_is_last = (c_arr == last_cat).astype(np.float32)
            set_last5 = set(h_cats[:5])
            c_in_last5 = np.array([float(x in set_last5) for x in c_cats], dtype=np.float32)
            last_ev = float(h_evs[0]) if (h_evs is not None and len(h_evs) > 0) else 0.0
            
            flat_hist_len.append(np.full(n_cands, float(h_len), dtype=np.float32))
            flat_hist_count.append(c_cnt)
            flat_hist_ratio.append(c_ratio)
            flat_hist_is_last.append(c_is_last)
            flat_hist_in_last5.append(c_in_last5)
            flat_high_intent_count.append(hi_cnt)
            flat_high_intent_ratio.append(hi_ratio)
            flat_last_event_type.append(np.full(n_cands, last_ev, dtype=np.float32))
            
    explode_cols = ['item_id', 'category_id', 'f1', 'f2', 'f3', 'f4', 'f5']
    if not is_test:
        explode_cols.append('target')
    if include_pos:
        explode_cols.append('pos')
    flat_df = df.select(['feed_id', *explode_cols]).explode(explode_cols)
    
    # 4. Add candidate features
    flat_df = flat_df.with_columns(
        pl.Series('embedding_dot', dot),
        pl.Series('embedding_cosine', cosine),
        pl.Series('embedding_missing', missing),
        pl.Series('embedding_l2', l2_dist),
        pl.Series('item_norm', item_norm),
        pl.Series('hist_len', np.concatenate(flat_hist_len)),
        pl.Series('hist_cat_count', np.concatenate(flat_hist_count)),
        pl.Series('hist_cat_ratio', np.concatenate(flat_hist_ratio)),
        pl.Series('hist_cat_is_last', np.concatenate(flat_hist_is_last)),
        pl.Series('hist_cat_in_last5', np.concatenate(flat_hist_in_last5)),
        pl.Series('hist_item_sim_max', sim_max),
        pl.Series('hist_item_sim_mean', sim_mean),
        pl.Series('hist_item_sim_last', sim_last),
        pl.Series('hist_high_intent_count', np.concatenate(flat_high_intent_count)),
        pl.Series('hist_high_intent_ratio', np.concatenate(flat_high_intent_ratio)),
        pl.Series('last_hist_event_type', np.concatenate(flat_last_event_type)),
    )
    
    # 5. Feed context & within-feed relative features (Full slate context)
    flat_df = flat_df.with_columns(
        pl.col('item_id').count().over('feed_id').cast(pl.Float32).alias('feed_size'),
        pl.col('item_id').count().over(['feed_id', 'category_id']).cast(pl.Float32).alias('feed_cat_count'),
    ).with_columns(
        (pl.col('feed_cat_count') / pl.col('feed_size')).alias('feed_cat_ratio'),
        (pl.col('embedding_cosine') - pl.col('embedding_cosine').max().over('feed_id')).alias('cosine_diff_max'),
        (pl.col('embedding_cosine') - pl.col('embedding_cosine').mean().over('feed_id')).alias('cosine_diff_mean'),
        ((pl.col('embedding_cosine') - pl.col('embedding_cosine').mean().over('feed_id')) / (pl.col('embedding_cosine').std().over('feed_id') + 1e-5)).alias('cosine_zscore'),
        (pl.col('embedding_cosine').rank().over('feed_id') / pl.col('feed_size')).alias('cosine_rank'),
        (pl.col('embedding_dot') - pl.col('embedding_dot').max().over('feed_id')).alias('dot_diff_max'),
        (pl.col('embedding_dot') - pl.col('embedding_dot').mean().over('feed_id')).alias('dot_diff_mean'),
        ((pl.col('embedding_dot') - pl.col('embedding_dot').mean().over('feed_id')) / (pl.col('embedding_dot').std().over('feed_id') + 1e-5)).alias('dot_zscore'),
        (pl.col('hist_item_sim_max') - pl.col('hist_item_sim_max').max().over('feed_id')).alias('hist_item_sim_diff_max'),
        ((pl.col('hist_item_sim_max') - pl.col('hist_item_sim_max').mean().over('feed_id')) / (pl.col('hist_item_sim_max').std().over('feed_id') + 1e-5)).alias('hist_item_sim_zscore'),
        (pl.col('hist_item_sim_max').rank().over('feed_id') / pl.col('feed_size')).alias('hist_item_sim_rank'),
        (pl.col('f1') - pl.col('f1').max().over('feed_id')).alias('f1_diff_max'),
        (pl.col('f1') - pl.col('f1').mean().over('feed_id')).alias('f1_diff_mean'),
        (pl.col('f2') - pl.col('f2').max().over('feed_id')).alias('f2_diff_max'),
        (pl.col('f2') - pl.col('f2').mean().over('feed_id')).alias('f2_diff_mean'),
        (pl.col('f3') - pl.col('f3').max().over('feed_id')).alias('f3_diff_max'),
        (pl.col('f3') - pl.col('f3').mean().over('feed_id')).alias('f3_diff_mean'),
        (pl.col('f4') - pl.col('f4').max().over('feed_id')).alias('f4_diff_max'),
        (pl.col('f4') - pl.col('f4').mean().over('feed_id')).alias('f4_diff_mean'),
        (pl.col('f5') - pl.col('f5').max().over('feed_id')).alias('f5_diff_max'),
        (pl.col('f5') - pl.col('f5').mean().over('feed_id')).alias('f5_diff_mean'),
    )
    return flat_df

def load_split_pool(split_dir: Path, num_files: int = 0, is_test: bool = False) -> tuple[Pool, pl.DataFrame]:
    """Loads a split into a CatBoost Pool with all 45 features and STRICT BINARY TARGET."""
    files = sorted((split_dir / 'impressions').glob('*.parquet'))
    if num_files > 0:
        files = files[:num_files]
        
    meta = json.loads((split_dir / 'item_embeddings' / 'data.meta').read_text())
    embeddings = np.memmap(
        split_dir / 'item_embeddings' / 'data.memmap',
        mode='r',
        dtype=meta['dtype'],
        shape=tuple(meta['shape']),
        order=meta['order'],
    )
    
    dfs = []
    for f in files:
        dfs.append(extract_features_from_parquet(f, embeddings, is_test=is_test, include_pos=False))
        
    combined = pl.concat(dfs).rechunk()
    
    if not is_test:
        pool = Pool(
            combined.select(ALL_FEATURES),
            label=combined['target'],  # STRICTLY BINARY TARGET {0, 1}
            cat_features=CAT_FEATURES,
            group_id=combined['feed_id'],
        )
    else:
        pool = Pool(
            combined.select(ALL_FEATURES),
            cat_features=CAT_FEATURES,
            group_id=combined['feed_id'],
        )
    return pool, combined
