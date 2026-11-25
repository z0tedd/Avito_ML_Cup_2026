import json, time
from pathlib import Path
import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader
from catboost import CatBoostRanker, Pool
from catboost.utils import eval_metric

from feature_extractor import ALL_FEATURES, CAT_FEATURES, extract_features_from_parquet, BASE_FLOAT_FEATURES, NEW_FLOAT_FEATURES
from neural_dataset import FeedRankDataset, collate_feeds, FLOAT_FEATURES
from improved_neural_ranker import ImprovedTransformerRanker, distillation_loss_fn

def load_feeds_and_features_with_teacher(split_dir: Path, num_files: int = 0, teacher_model: CatBoostRanker = None, is_test: bool = False):
    files = sorted((split_dir / "impressions").glob("*.parquet"))
    if num_files > 0:
        files = files[:num_files]
        
    meta = json.loads((split_dir / "item_embeddings" / "data.meta").read_text())
    embeddings = np.memmap(
        split_dir / "item_embeddings" / "data.memmap",
        mode="r",
        dtype=meta["dtype"],
        shape=tuple(meta["shape"]),
        order=meta["order"],
    )
    
    feed_dfs = []
    flat_dfs = []
    item_embs_list = []
    
    for f in files:
        f_df = pl.read_parquet(f, columns=["feed_id", "user_embedding", "item_id", "item_id__embedding_idx"])
        feed_dfs.append(f_df.select(["feed_id", "user_embedding"]))
        
        all_idx = f_df["item_id__embedding_idx"].explode().to_numpy()
        mask_valid = all_idx >= 0
        valid_indices = all_idx[mask_valid]
        unique_indices, inverse = np.unique(valid_indices, return_inverse=True)
        unique_embs = embeddings[unique_indices].astype(np.float32)
        
        full_embs = np.zeros((len(all_idx), 128), dtype=np.float32)
        full_embs[mask_valid] = unique_embs[inverse]
        denom = np.linalg.norm(full_embs, axis=1, keepdims=True)
        norm_embs = np.divide(full_embs, denom, out=np.zeros_like(full_embs), where=denom > 0)
        item_embs_list.append(norm_embs)
        
        flat = extract_features_from_parquet(f, embeddings, is_test=is_test, include_pos=False)
        flat_dfs.append(flat)
        
    combined_feeds = pl.concat(feed_dfs).rechunk()
    combined_flat = pl.concat(flat_dfs).rechunk()
    combined_item_embs = np.concatenate(item_embs_list, axis=0)
    
    teacher_scores = None
    if teacher_model is not None:
        t0 = time.time()
        cb_pool = Pool(
            combined_flat.select(ALL_FEATURES),
            cat_features=CAT_FEATURES,
            group_id=combined_flat["feed_id"],
        )
        teacher_scores = teacher_model.predict(cb_pool, thread_count=8).astype(np.float32)
        print(f"Teacher predictions generated in {time.time()-t0:.2f}s")
        
    dataset = FeedRankDataset(
        feeds_df=combined_feeds,
        flat_df=combined_flat,
        full_item_embs=combined_item_embs,
        teacher_scores=teacher_scores,
        is_train=(not is_test and teacher_scores is not None)
    )
    return dataset, combined_flat

def evaluate_model(model, loader, val_flat_df, device):
    model.eval()
    all_scores = []
    
    with torch.inference_mode():
        for batch in loader:
            dense_feats, cat_ids, users, padding, _, _ = (x.to(device) for x in batch)
            logits = model(dense_feats, cat_ids, users, padding)
            valid_scores = logits[~padding].cpu().numpy()
            all_scores.append(valid_scores)
            
    scores = np.concatenate(all_scores).astype(np.float64)
    labels = val_flat_df["target"].to_numpy().tolist()
    groups = val_flat_df["feed_id"].to_numpy().tolist()
    
    ndcg_30 = eval_metric(labels, scores.tolist(), "NDCG:top=30;type=Base", group_id=groups)[0]
    ndcg_full = eval_metric(labels, scores.tolist(), "NDCG:type=Base", group_id=groups)[0]
    ndcg_10 = eval_metric(labels, scores.tolist(), "NDCG:top=10;type=Base", group_id=groups)[0]
    
    return {
        "val_ndcg_at_30": float(ndcg_30),
        "val_ndcg": float(ndcg_full),
        "val_ndcg_at_10": float(ndcg_10),
        "scores": scores,
    }

def run_v7_transformer():
    output_dir = Path("runs/v7_fixed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cpu")
    torch.set_num_threads(8)
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("=" * 70)
    print("TRANSFORMER V7: LISTWISE CROSS-ENTROPY (4 TRAIN FILES, DISTILLATION FROM CATBOOST V7)")
    print("=" * 70)
    
    # 1. Load teacher CatBoost v7
    cb_teacher_path = output_dir / "catboost_v7.cbm"
    assert cb_teacher_path.exists(), f"Teacher model not found at {cb_teacher_path}"
    print(f"Loading CatBoost teacher model from {cb_teacher_path}...")
    teacher_model = CatBoostRanker()
    teacher_model.load_model(str(cb_teacher_path))
    
    # 2. Load Train with teacher scores (4 files)
    num_train_files = 4
    print(f"\nLoading TRAIN dataset ({num_train_files} files, ~1.9M candidates) + generating teacher scores...")
    t0 = time.time()
    train_dataset, train_flat = load_feeds_and_features_with_teacher(
        Path("competition/train"), num_files=num_train_files, teacher_model=teacher_model
    )
    print(f"Train dataset loaded: {len(train_dataset):,} feeds in {time.time()-t0:.2f}s")
    
    # 3. Load full Val
    print("\nLoading FULL VAL dataset (14,423 feeds, 1.39M candidates)...")
    t0 = time.time()
    val_dataset, val_flat = load_feeds_and_features_with_teacher(
        Path("competition/val"), num_files=0, teacher_model=None
    )
    print(f"Val dataset loaded: {len(val_dataset):,} feeds in {time.time()-t0:.2f}s")
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_feeds)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_feeds)
    
    # In dense dim: 128 (item emb) + 44 (scalars) = 172
    in_dense_dim = 128 + len(FLOAT_FEATURES)
    print(f"Transformer input dense dimensions: {in_dense_dim}")
    
    model = ImprovedTransformerRanker(
        in_dense_features=in_dense_dim,
        width=128,
        layers=2,
        heads=8,
        dropout=0.1,
        cat_dim=16
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model initialized: {num_params:,} parameters")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4, eta_min=1e-4)
    
    epochs = 4
    best_ndcg_full = 0.0
    best_metrics = {}
    best_scores = None
    
    print("\nStarting Listwise Cross-Entropy & Phase Distillation training (alpha_teacher=0.5)...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0
        t_epoch = time.time()
        
        for batch in train_loader:
            dense_feats, cat_ids, users, padding, targets, teacher_scores = (x.to(device) for x in batch)
            optimizer.zero_grad(set_to_none=True)
            
            logits = model(dense_feats, cat_ids, users, padding)
            loss = distillation_loss_fn(
                logits, targets, padding,
                teacher_logits=teacher_scores,
                temperature=0.15,
                alpha_teacher=0.5
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            n_batches += 1
            
        scheduler.step()
        train_time = time.time() - t_epoch
        train_loss /= n_batches
        
        t_val = time.time()
        val_eval = evaluate_model(model, val_loader, val_flat, device)
        val_time = time.time() - t_val
        
        print(f"Epoch {epoch:2d}/{epochs:2d} | Train Loss: {train_loss:.4f} ({train_time:.1f}s) | Val Full NDCG: {val_eval['val_ndcg']:.6f} | Val NDCG@30: {val_eval['val_ndcg_at_30']:.6f} ({val_time:.1f}s)")
        
        if val_eval["val_ndcg"] > best_ndcg_full:
            best_ndcg_full = val_eval["val_ndcg"]
            best_scores = val_eval["scores"]
            best_metrics = {
                "epoch": epoch,
                "parameters": num_params,
                "val_full_ndcg": val_eval["val_ndcg"],
                "val_ndcg_at_30": val_eval["val_ndcg_at_30"],
                "val_ndcg_at_10": val_eval["val_ndcg_at_10"],
            }
            torch.save(model.state_dict(), output_dir / "transformer_v7.pt")
            
    print("\n" + "=" * 70)
    print("BEST TRANSFORMER V7 METRICS:")
    print(json.dumps(best_metrics, indent=2))
    
    metrics_path = output_dir / "transformer_v7_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(best_metrics, f, indent=2)
        
    # Save validation predictions for ensembling
    val_preds_df = val_flat.select(['feed_id', 'item_id', 'target']).with_columns(
        pl.Series('score_tf_v7', best_scores.astype(np.float32))
    )
    val_preds_path = output_dir / "val_predictions_transformer_v7.parquet"
    val_preds_df.write_parquet(val_preds_path)
    print(f"Val predictions saved to: {val_preds_path}")
    print("=" * 70)

if __name__ == "__main__":
    run_v7_transformer()
