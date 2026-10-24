"""
src/train.py
------------
Training pipeline: parse -> deduplicate -> split -> extract features ->
train (LogisticRegression baseline + LightGBM main) -> choose threshold ->
save models, scaler, lookup, threshold config.

Run with:
    python -m src.train [--csv dataset/csic_database.csv] [--seed 42]

Outputs written to models/ and reports/ so later stages can read them.
"""

import argparse
import json
import os
import pickle
import random
import warnings
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix
)

# LightGBM is chosen over XGBoost because:
# 1. Faster training on tabular data with many features
# 2. Better handling of categorical-like integer features
# 3. leaf-wise growth tends to find complex patterns in security data
import lightgbm as lgb

warnings.filterwarnings('ignore')

from src.parse import load_and_parse
from src.features import FeatureExtractor

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def stratified_split(records, train_ratio=0.70, val_ratio=0.15, seed=SEED):
    """
    Stratified train/val/test/prod_clean_holdout split.
    prod_clean_holdout: a separate hold-out of NORMAL records only.
    """
    from sklearn.model_selection import train_test_split

    labels = np.array([r['label'] for r in records])
    idx = np.arange(len(records))

    # First split: train+val+test vs prod_clean_holdout (5% of normals)
    normal_idx = idx[labels == 0]
    attack_idx = idx[labels == 1]

    # Reserve 5% of normal records as prod_clean_holdout
    n_holdout = max(200, int(0.05 * len(normal_idx)))
    holdout_idx = np.random.choice(normal_idx, size=n_holdout, replace=False)
    remaining_normal = np.setdiff1d(normal_idx, holdout_idx)
    remaining_idx = np.concatenate([remaining_normal, attack_idx])
    remaining_labels = labels[remaining_idx]

    # Stratified split of remaining into train / val / test
    test_ratio = 1 - train_ratio - val_ratio
    idx_train, idx_temp, _, labels_temp = train_test_split(
        remaining_idx, remaining_labels,
        test_size=(val_ratio + test_ratio), stratify=remaining_labels,
        random_state=seed
    )
    idx_val, idx_test = train_test_split(
        idx_temp, test_size=test_ratio / (val_ratio + test_ratio),
        stratify=labels_temp, random_state=seed
    )

    def pick(idxs):
        return [records[i] for i in idxs]

    return (
        pick(idx_train),
        pick(idx_val),
        pick(idx_test),
        pick(holdout_idx),  # all normal
    )


def metrics_at_threshold(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    return dict(precision=prec, recall=rec, f1=f1, fpr=fpr, tpr=tpr,
                tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn))


def choose_threshold(y_true, y_prob, max_fpr=0.01):
    """
    Sweep thresholds from 0.01 to 0.99 and pick the lowest threshold
    that keeps FPR <= max_fpr, maximising recall.
    Returns (best_threshold, metrics_dict).
    """
    best_thr = 0.5
    best_recall = -1.0
    for thr in np.linspace(0.01, 0.99, 500):
        m = metrics_at_threshold(y_true, y_prob, thr)
        if m['fpr'] <= max_fpr and m['recall'] > best_recall:
            best_recall = m['recall']
            best_thr = thr
    return float(best_thr), metrics_at_threshold(y_true, y_prob, best_thr)


def tune_lgbm(X_train, y_train, seed=SEED):
    """
    Small grid search over key LightGBM hyperparameters using 5-fold CV
    (optimising average precision, which tracks PR-AUC).
    Returns best params dict.
    """
    param_grid = [
        {'n_estimators': 300, 'max_depth': 6,  'learning_rate': 0.05,  'num_leaves': 31},
        {'n_estimators': 500, 'max_depth': 8,  'learning_rate': 0.03,  'num_leaves': 63},
        {'n_estimators': 200, 'max_depth': 4,  'learning_rate': 0.1,   'num_leaves': 15},
    ]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    best_score = -1
    best_params = param_grid[0]
    for params in param_grid:
        model = lgb.LGBMClassifier(
            **params, random_state=seed, verbose=-1,
            n_jobs=-1, class_weight='balanced'
        )
        probs = cross_val_predict(model, X_train, y_train, cv=cv, method='predict_proba')[:, 1]
        score = average_precision_score(y_train, probs)
        print(f"  LightGBM params={params} -> PR-AUC CV={score:.4f}")
        if score > best_score:
            best_score = score
            best_params = params
    print(f"  Best params: {best_params} (PR-AUC={best_score:.4f})")
    return best_params


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def train(csv_path: str, out_dir: str = 'models', reports_dir: str = 'reports'):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    print("=" * 60)
    print("Step 1: Parsing and deduplicating...")
    records = load_and_parse(csv_path)
    n_normal = sum(1 for r in records if r['label'] == 0)
    n_attack = sum(1 for r in records if r['label'] == 1)
    print(f"  Unique records: {len(records)} ({n_normal} normal, {n_attack} attack)")

    print("Step 2: Splitting...")
    train_recs, val_recs, test_recs, holdout_recs = stratified_split(records)
    print(f"  Train: {len(train_recs)}, Val: {len(val_recs)}, "
          f"Test: {len(test_recs)}, Holdout(normal only): {len(holdout_recs)}")

    print("Step 3: Fitting feature extractor on NORMAL train records only...")
    fx = FeatureExtractor(use_ngram=False)  # n-gram off for ONNX compat
    normal_train = [r for r in train_recs if r['label'] == 0]
    fx.fit(normal_train)
    fx.save_lookup(os.path.join(out_dir, 'lookup.json'))
    print(f"  Feature vector length: {fx.n_features}")
    print(f"  Endpoint profiles learned: {len(fx._lookup)}")

    print("Step 4: Extracting feature matrices...")
    X_train = fx.transform(train_recs)
    y_train = np.array([r['label'] for r in train_recs])
    X_val   = fx.transform(val_recs)
    y_val   = np.array([r['label'] for r in val_recs])
    X_test  = fx.transform(test_recs)
    y_test  = np.array([r['label'] for r in test_recs])
    X_hold  = fx.transform(holdout_recs)
    y_hold  = np.array([r['label'] for r in holdout_recs])

    # Scale (fit on train only)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)
    X_hold_s  = scaler.transform(X_hold)

    print("Step 5: Training Baseline (Logistic Regression)...")
    lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=SEED)
    lr.fit(X_train_s, y_train)
    lr_val_prob = lr.predict_proba(X_val_s)[:, 1]
    lr_thr, lr_val_m = choose_threshold(y_val, lr_val_prob, max_fpr=0.01)
    lr_test_prob = lr.predict_proba(X_test_s)[:, 1]
    lr_test_m = metrics_at_threshold(y_test, lr_test_prob, lr_thr)
    lr_roc = roc_auc_score(y_test, lr_test_prob)
    lr_pr  = average_precision_score(y_test, lr_test_prob)

    print(f"  LR  val threshold={lr_thr:.3f} FPR={lr_val_m['fpr']:.4f} "
          f"Recall={lr_val_m['recall']:.4f}")

    print("Step 6: Tuning LightGBM (5-fold CV on train)...")
    best_params = tune_lgbm(X_train_s, y_train)
    lgbm = lgb.LGBMClassifier(**best_params, random_state=SEED, verbose=-1,
                               n_jobs=-1, class_weight='balanced')
    lgbm.fit(X_train_s, y_train)
    lgbm_val_prob = lgbm.predict_proba(X_val_s)[:, 1]
    lgbm_thr, lgbm_val_m = choose_threshold(y_val, lgbm_val_prob, max_fpr=0.01)
    lgbm_test_prob = lgbm.predict_proba(X_test_s)[:, 1]
    lgbm_test_m = metrics_at_threshold(y_test, lgbm_test_prob, lgbm_thr)
    lgbm_roc = roc_auc_score(y_test, lgbm_test_prob)
    lgbm_pr  = average_precision_score(y_test, lgbm_test_prob)

    print(f"  LGBM val threshold={lgbm_thr:.3f} FPR={lgbm_val_m['fpr']:.4f} "
          f"Recall={lgbm_val_m['recall']:.4f}")

    # Comparison table
    print("\n  === Model Comparison (test set) ===")
    hdr = f"{'Model':<20} {'Prec':>6} {'Rec':>6} {'F1':>6} {'ROC-AUC':>8} {'PR-AUC':>8} {'FPR':>6}"
    print(hdr)
    print("-" * len(hdr))
    for name, m, roc, pr in [
        ('LogisticRegression', lr_test_m, lr_roc, lr_pr),
        ('LightGBM', lgbm_test_m, lgbm_roc, lgbm_pr),
    ]:
        print(f"{name:<20} {m['precision']:>6.4f} {m['recall']:>6.4f} {m['f1']:>6.4f} "
              f"{roc:>8.4f} {pr:>8.4f} {m['fpr']:>6.4f}")

    # Choose best model = LightGBM (better PR-AUC in practice on security data)
    print("\nStep 7: Saving best model (LightGBM)...")
    with open(os.path.join(out_dir, 'lgbm_model.pkl'), 'wb') as f:
        pickle.dump(lgbm, f)
    with open(os.path.join(out_dir, 'lr_model.pkl'), 'wb') as f:
        pickle.dump(lr, f)
    with open(os.path.join(out_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    with open(os.path.join(out_dir, 'feature_extractor.pkl'), 'wb') as f:
        pickle.dump(fx, f)

    # Save threshold config
    config = {
        'threshold': lgbm_thr,
        'model': 'lgbm',
        'n_features': fx.n_features,
        'seed': SEED,
        'lgbm_params': best_params,
        'val_metrics_at_threshold': lgbm_val_m,
    }
    with open(os.path.join(out_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # Save metrics.json (machine-readable)
    # Prod clean holdout FPR (all normal)
    hold_prob = lgbm.predict_proba(X_hold_s)[:, 1]
    hold_pred = (hold_prob >= lgbm_thr).astype(int)
    hold_fp   = int(hold_pred.sum())   # all are normal, so any predicted-attack = FP
    hold_fpr  = hold_fp / len(hold_pred) if len(hold_pred) > 0 else 0.0

    metrics = {
        'model': 'LightGBM',
        'threshold': lgbm_thr,
        'test': {
            **lgbm_test_m,
            'roc_auc': lgbm_roc,
            'pr_auc': lgbm_pr,
        },
        'val_at_threshold': lgbm_val_m,
        'prod_clean_holdout': {
            'n_samples': len(holdout_recs),
            'fp_count': hold_fp,
            'fpr': hold_fpr,
        },
        'baseline_lr_test': {
            **lr_test_m,
            'roc_auc': lr_roc,
            'pr_auc': lr_pr,
        },
        'split_sizes': {
            'train': len(train_recs),
            'val': len(val_recs),
            'test': len(test_recs),
            'holdout': len(holdout_recs),
        },
    }
    metrics_path = os.path.join(reports_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved to {metrics_path}")
    print(f"  Prod clean holdout FPR: {hold_fpr:.4f} ({hold_fp}/{len(holdout_recs)} false alarms)")

    # Save raw predictions for evaluate.py
    np.save(os.path.join(reports_dir, 'test_prob.npy'), lgbm_test_prob)
    np.save(os.path.join(reports_dir, 'test_labels.npy'), y_test)
    np.save(os.path.join(reports_dir, 'val_prob.npy'), lgbm_val_prob)
    np.save(os.path.join(reports_dir, 'val_labels.npy'), y_val)
    np.save(os.path.join(reports_dir, 'hold_prob.npy'), hold_prob)
    np.save(os.path.join(reports_dir, 'hold_labels.npy'), y_hold)

    # Save test records for error analysis
    with open(os.path.join(reports_dir, 'test_records.pkl'), 'wb') as f:
        pickle.dump(test_recs, f)

    print("\nTraining complete.")
    return lgbm, scaler, fx, config, metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='dataset/csic_database.csv')
    parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args()
    SEED = args.seed
    np.random.seed(SEED)
    random.seed(SEED)
    train(args.csv)
