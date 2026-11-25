import json, time, shutil
from pathlib import Path
import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader
from catboost import CatBoostRanker, Pool

from feature_extractor import ALL_FEATURES, CAT_FEATURES
from neural_dataset import FeedRankDataset, collate_feeds, FLOAT_FEATURES
from improved_neural_ranker import ImprovedTransformerRanker
from train_v7_transformer import load_feeds_and_features_with_teacher

def compute_transformer_predictions(model, loader, device):
    model.eval()
    all_scores = []
    with torch.inference_mode():
        for batch in loader:
            dense_feats, cat_ids, users, padding = (x.to(device) for x in batch[:4])
            logits = model(dense_feats, cat_ids, users, padding)
            valid_scores = logits[~padding].cpu().numpy()
            all_scores.append(valid_scores)
    return np.concatenate(all_scores).astype(np.float64)

def run_v7_predictions():
    output_dir = Path("runs/v7_fixed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("V7 INFERENCE & SUBMISSION GENERATOR (STRICT BINARY FULL SLATES, NO MMR)")
    print("=" * 70)
    
    cb_path = output_dir / "catboost_v7.cbm"
    tf_path = output_dir / "transformer_v7.pt"
    assert cb_path.exists(), f"CatBoost model not found at {cb_path}"
    assert tf_path.exists(), f"Transformer model not found at {tf_path}"
    
    # 1. Load models
    print("\n1. Loading CatBoost v7 and Transformer v7 models...")
    cb_model = CatBoostRanker()
    cb_model.load_model(str(cb_path))
    print(f"   CatBoost model loaded from: {cb_path}")
    
    device = torch.device("cpu")
    torch.set_num_threads(8)
    in_dense_dim = 128 + len(FLOAT_FEATURES)
    tf_model = ImprovedTransformerRanker(
        in_dense_features=in_dense_dim,
        width=128,
        layers=2,
        heads=8,
        dropout=0.1,
        cat_dim=16
    ).to(device)
    tf_model.load_state_dict(torch.load(str(tf_path), map_location=device))
    print(f"   Transformer model loaded from: {tf_path}")
    
    # 2. Load test data
    print("\n2. Loading FULL TEST dataset (64 files, 2.9M candidates)...")
    t0 = time.time()
    test_dataset, test_flat = load_feeds_and_features_with_teacher(Path("competition/test"), num_files=0, is_test=True)
    load_time = time.time() - t0
    print(f"   Test data loaded: {len(test_dataset):,} feeds, {len(test_flat):,} candidates in {load_time:.2f}s")
    
    # 3. Predict CatBoost v7
    print("\n3. Predicting CatBoost v7 test scores...")
    t0 = time.time()
    test_pool = Pool(
        test_flat.select(ALL_FEATURES),
        group_id=test_flat["feed_id"],
        cat_features=CAT_FEATURES,
    )
    cb_scores = cb_model.predict(test_pool, thread_count=8).astype(np.float32)
    cb_time = time.time() - t0
    print(f"   CatBoost predictions done in {cb_time:.2f}s")
    
    # 4. Predict Transformer v7
    print("\n4. Predicting Transformer v7 test scores...")
    t0 = time.time()
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_feeds)
    tf_scores = compute_transformer_predictions(tf_model, test_loader, device).astype(np.float32)
    tf_time = time.time() - t0
    print(f"   Transformer predictions done in {tf_time:.2f}s")
    
    # 5. Base dataframe with scores
    print("\n5. Computing within-feed Z-Score normalization...")
    base_df = test_flat.select(["feed_id", "item_id"]).with_columns([
        pl.Series("score_cb", cb_scores),
        pl.Series("score_tf", tf_scores),
    ])
    
    z_df = base_df.with_columns([
        ((pl.col("score_cb") - pl.col("score_cb").mean().over("feed_id")) / (pl.col("score_cb").std().over("feed_id") + 1e-5)).alias("z_cb"),
        ((pl.col("score_tf") - pl.col("score_tf").mean().over("feed_id")) / (pl.col("score_tf").std().over("feed_id") + 1e-5)).alias("z_tf"),
    ])
    
    sample_sub = pl.read_parquet("competition/sample_submission.parquet")
    expected_rows = len(sample_sub)
    print(f"   Sample submission expected rows: {expected_rows:,}")
    
    # Helper to format and align with sample_submission
    def make_aligned_submission(score_expr):
        sub_raw = z_df.select([
            pl.col("feed_id").cast(pl.Int32),
            pl.col("item_id").cast(pl.Int32),
            score_expr.cast(pl.Float32).alias("score")
        ])
        aligned = sample_sub.select(["feed_id", "item_id"]).join(sub_raw, on=["feed_id", "item_id"], how="left")
        assert len(aligned) == expected_rows, f"Row count mismatch: {len(aligned)} vs {expected_rows}"
        assert not aligned["score"].is_null().any(), "Found null scores!"
        assert not aligned["score"].is_nan().any(), "Found NaN scores!"
        assert aligned["score"].is_finite().all(), "Found infinite scores!"
        assert aligned.schema == sample_sub.schema, f"Schema mismatch: {aligned.schema}"
        np.testing.assert_array_equal(aligned["feed_id"].to_numpy(), sample_sub["feed_id"].to_numpy())
        np.testing.assert_array_equal(aligned["item_id"].to_numpy(), sample_sub["item_id"].to_numpy())
        return aligned
        
    # Variant 1: Requested 70/30 Ensemble (0.70 CB + 0.30 TF, Val NDCG = 0.361415)
    sub_70_30 = make_aligned_submission(0.70 * pl.col("z_cb") + 0.30 * pl.col("z_tf"))
    sub_70_30_path = output_dir / "submission_v7_ensemble_70_30.parquet"
    sub_70_30.write_parquet(sub_70_30_path, compression="zstd")
    print(f"\n   [Variant 1] Saved 70/30 Ensemble to: {sub_70_30_path} ({sub_70_30_path.stat().st_size / (1024*1024):.2f} MB)")
    
    # Variant 2: Optimal Validation Ensemble (0.80 CB + 0.20 TF, Val NDCG = 0.363077)
    sub_80_20 = make_aligned_submission(0.80 * pl.col("z_cb") + 0.20 * pl.col("z_tf"))
    sub_80_20_path = output_dir / "submission_v7_ensemble_80_20.parquet"
    sub_80_20.write_parquet(sub_80_20_path, compression="zstd")
    print(f"   [Variant 2] Saved Optimal 80/20 Ensemble to: {sub_80_20_path} ({sub_80_20_path.stat().st_size / (1024*1024):.2f} MB)")
    
    # Variant 3: Solo CatBoost v7 (16 files, Val NDCG = 0.362992)
    sub_cb_solo = make_aligned_submission(pl.col("score_cb"))
    sub_cb_solo_path = output_dir / "submission_v7_catboost_solo.parquet"
    sub_cb_solo.write_parquet(sub_cb_solo_path, compression="zstd")
    print(f"   [Variant 3] Saved Solo CatBoost v7 to: {sub_cb_solo_path} ({sub_cb_solo_path.stat().st_size / (1024*1024):.2f} MB)")
    
    # Default submission.parquet (Primary submission: 70/30 ensemble as requested in plan.md)
    default_sub_path = output_dir / "submission.parquet"
    shutil.copyfile(sub_70_30_path, default_sub_path)
    print(f"   [Main Upload] Copied 70/30 ensemble to default submission file: {default_sub_path}")
    
    # 6. Final Summary Metrics
    summary = {
        "version": "v7_fixed",
        "date": "2026-11-24",
        "description": "Avito RecSys 2026 v7: Strict Binary Target, Full Slate Intact, 16 Train Files, Clean 45 Features, Pure YetiRank, No MMR",
        "validation_full_ndcg": {
            "catboost_v7_solo": 0.362992,
            "transformer_v7_solo": 0.345541,
            "ensemble_v7_70_30": 0.361415,
            "ensemble_v7_80_20": 0.363077,
            "previous_v4_best_lb": 0.353913,
            "previous_v6_drop": 0.351775,
            "net_validation_gain_over_v4": 0.363077 - 0.354301,
        },
        "submissions_generated": {
            "requested_70_30": str(sub_70_30_path),
            "optimal_80_20": str(sub_80_20_path),
            "catboost_solo": str(sub_cb_solo_path),
            "default_upload": str(default_sub_path),
        },
        "test_integrity": {
            "total_rows": expected_rows,
            "null_count": 0,
            "nan_count": 0,
            "infinite_count": 0,
            "exact_alignment_with_sample_submission": True,
        }
    }
    
    summary_path = output_dir / "metrics_v7.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n6. Summary metrics saved to: {summary_path}")
    print("\n" + "=" * 70)
    print("V7 PREDICTIONS COMPLETED SUCCESSFULLY")
    print("=" * 70)

if __name__ == "__main__":
    run_v7_predictions()
