import json, time
from pathlib import Path
import numpy as np
import polars as pl
from catboost import CatBoostRanker

from feature_extractor import ALL_FEATURES, CAT_FEATURES, load_split_pool

def run_v7_catboost():
    output_dir = Path("runs/v7_fixed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("CATBOOST V7: 16 FULL TRAIN FILES, STRICT BINARY TARGET, PURE YETIRANK")
    print("=" * 70)
    
    num_train_files = 16
    print(f"\n1. Loading {num_train_files} FULL TRAIN files (full feeds, binary target)...")
    t0 = time.time()
    train_pool, train_df = load_split_pool(Path("competition/train"), num_files=num_train_files)
    train_time = time.time() - t0
    print(f"   Train loaded: {train_pool.num_row():,} rows, {train_df['feed_id'].n_unique():,} feeds in {train_time:.2f}s")
    target_dist = train_df['target'].value_counts().to_dicts()
    print(f"   Target distribution: {target_dist}")
    
    print("\n2. Loading FULL VALIDATION set (64 files, 1.39M candidates)...")
    t0 = time.time()
    val_pool, val_df = load_split_pool(Path("competition/val"), num_files=0)
    val_time = time.time() - t0
    print(f"   Val loaded: {val_pool.num_row():,} rows, {val_df['feed_id'].n_unique():,} feeds in {val_time:.2f}s")
    
    params = {
        "loss_function": "YetiRank",
        "eval_metric": "NDCG:type=Base",
        "depth": 7,
        "learning_rate": 0.15,
        "num_trees": 1800,
        "early_stopping_rounds": 80,
        "metric_period": 25,
        "task_type": "GPU",
        "devices": "0",
        "random_state": 666,
        "l2_leaf_reg": 5,
        "nan_mode": "Min",
        "verbose": 50,
        "use_best_model": True,
    }
    
    print("\n3. Training CatBoostRanker v7 on GPU (16 full files, pure YetiRank)...")
    print(f"   Features count: {len(ALL_FEATURES)}")
    t0 = time.time()
    model = CatBoostRanker(**params)
    model.fit(train_pool, eval_set=val_pool)
    fit_time = time.time() - t0
    print(f"\n   Training completed in {fit_time:.2f}s ({fit_time/60:.2f}m)")
    
    best_iteration = int(model.get_best_iteration())
    print(f"   Best iteration: {best_iteration}")
    
    # Save model
    model_path = output_dir / "catboost_v7.cbm"
    model.save_model(str(model_path))
    print(f"\n4. Model saved to: {model_path}")
    
    # Evaluate validation metrics
    print("\n5. Evaluating Full NDCG metrics on validation set...")
    quality = model.eval_metrics(
        val_pool,
        ["NDCG:type=Base", "NDCG:top=30;type=Base", "NDCG:top=10;type=Base"],
        eval_period=model.tree_count_,
    )
    val_ndcg = float(quality["NDCG:type=Base"][-1])
    val_ndcg_30 = float(quality["NDCG:top=30;type=Base"][-1])
    val_ndcg_10 = float(quality["NDCG:top=10;type=Base"][-1])
    
    print(f"   Validation Full NDCG:     {val_ndcg:.6f}")
    print(f"   Validation NDCG@30:       {val_ndcg_30:.6f}")
    print(f"   Validation NDCG@10:       {val_ndcg_10:.6f}")
    
    # Save validation predictions for ensembling
    print("\n6. Generating validation predictions for ensembling...")
    val_scores = model.predict(val_pool)
    val_preds_df = val_df.select(['feed_id', 'item_id', 'target']).with_columns(
        pl.Series('score_cb_v7', val_scores.astype(np.float32))
    )
    val_preds_path = output_dir / "val_predictions_catboost_v7.parquet"
    val_preds_df.write_parquet(val_preds_path)
    print(f"   Val predictions saved to: {val_preds_path}")
    
    # Feature importances
    feature_importances = model.get_feature_importance(train_pool)
    top_fi = sorted(zip(ALL_FEATURES, feature_importances), key=lambda x: x[1], reverse=True)
    print("\n7. Top 15 Features by Importance:")
    for rank, (feat, imp) in enumerate(top_fi[:15], 1):
        print(f"   {rank:2d}. {feat:<25}: {imp:.3f}%")
        
    metrics = {
        "model": "CatBoost v7 (16 Full Train Files, Binary Target, Pure YetiRank)",
        "val_full_ndcg": val_ndcg,
        "val_ndcg_at_30": val_ndcg_30,
        "val_ndcg_at_10": val_ndcg_10,
        "best_iteration": best_iteration,
        "tree_count": int(model.tree_count_),
        "num_train_files": num_train_files,
        "train_rows": int(train_pool.num_row()),
        "val_rows": int(val_pool.num_row()),
        "train_time_sec": float(fit_time),
        "target_distribution": target_dist,
        "top_features": [{f: float(imp)} for f, imp in top_fi[:15]],
    }
    
    metrics_path = output_dir / "catboost_v7_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"   Metrics saved to: {metrics_path}")
    print("\n" + "=" * 70)
    print("V7 CATBOOST TRAINING COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    run_v7_catboost()
