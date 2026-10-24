"""
src/evaluate.py
---------------
Generate evaluation artefacts:
  - Confusion matrix, classification report
  - ROC and PR curves saved as PNG
  - Feature importance plot (LightGBM)
  - Error analysis: 10 FP + 10 FN with raw request details
  - All metrics already in reports/metrics.json (written by train.py)

Run with:
    python -m src.evaluate
"""

import json
import os
import pickle
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
)

warnings.filterwarnings('ignore')

REPORTS  = 'reports'
MODELS   = 'models'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_artifacts():
    test_prob  = np.load(os.path.join(REPORTS, 'test_prob.npy'))
    y_test     = np.load(os.path.join(REPORTS, 'test_labels.npy'))
    hold_prob  = np.load(os.path.join(REPORTS, 'hold_prob.npy'))
    y_hold     = np.load(os.path.join(REPORTS, 'hold_labels.npy'))
    with open(os.path.join(MODELS, 'config.json')) as f:
        config = json.load(f)
    with open(os.path.join(REPORTS, 'test_records.pkl'), 'rb') as f:
        test_recs = pickle.load(f)
    with open(os.path.join(MODELS, 'lgbm_model.pkl'), 'rb') as f:
        lgbm = pickle.load(f)
    with open(os.path.join(MODELS, 'feature_extractor.pkl'), 'rb') as f:
        fx = pickle.load(f)
    return test_prob, y_test, hold_prob, y_hold, config, test_recs, lgbm, fx


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_roc_pr(y_test, test_prob, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor('#1a1a2e')
    for ax in axes:
        ax.set_facecolor('#16213e')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#444')

    # ROC
    fpr, tpr, _ = roc_curve(y_test, test_prob)
    auc = roc_auc_score(y_test, test_prob)
    axes[0].plot(fpr, tpr, color='#00d4ff', lw=2, label=f'ROC-AUC={auc:.4f}')
    axes[0].plot([0, 1], [0, 1], 'w--', lw=0.8, alpha=0.5)
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curve')
    axes[0].legend(facecolor='#16213e', labelcolor='white')

    # PR
    prec, rec, _ = precision_recall_curve(y_test, test_prob)
    pr_auc = average_precision_score(y_test, test_prob)
    axes[1].plot(rec, prec, color='#ff6b6b', lw=2, label=f'PR-AUC={pr_auc:.4f}')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curve')
    axes[1].legend(facecolor='#16213e', labelcolor='white')

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'roc_pr_curves.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {out_path}")


def plot_feature_importance(lgbm, fx, out_dir, top_n=30):
    importances = lgbm.feature_importances_
    names = fx.feature_names()
    idx = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')

    colors = plt.cm.viridis(np.linspace(0.3, 1.0, top_n))
    bars = ax.barh([names[i] for i in reversed(idx)],
                   [importances[i] for i in reversed(idx)],
                   color=colors)
    ax.set_xlabel('Feature Importance (split count)', color='white')
    ax.set_title('Top Feature Importances', color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'feature_importance.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {out_path}")


def plot_confusion_matrix(y_true, y_pred, out_dir, title='Confusion Matrix'):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Normal', 'Attack'], color='white')
    ax.set_yticklabels(['Normal', 'Attack'], color='white')
    ax.set_xlabel('Predicted', color='white')
    ax.set_ylabel('True', color='white')
    ax.set_title(title, color='white')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white', fontsize=16, fontweight='bold')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    out_path = os.path.join(out_dir, 'confusion_matrix.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------

def _request_summary(rec):
    lines = [
        f"  Method : {rec.get('method','')}",
        f"  Path   : {rec.get('path','')}",
        f"  Query  : {rec.get('query_string','')[:120]}",
        f"  Body   : {rec.get('body_string','')[:120]}",
        f"  Label  : {'normal' if rec.get('label')==0 else 'attack'}",
    ]
    return '\n'.join(lines)


def _categorise_miss(rec):
    """Heuristic: group misclassified requests by attack type."""
    qs = (rec.get('query_string', '') + ' ' + rec.get('body_string', '')).lower()
    if any(k in qs for k in ['select', 'union', 'drop', 'insert', 'sleep', '--', '/*']):
        return 'SQL Injection'
    if any(k in qs for k in ['<script', 'onerror', 'javascript:', 'alert(']):
        return 'XSS'
    if any(k in qs for k in ['../', 'etc/passwd', '/proc']):
        return 'Path Traversal'
    if any(k in qs for k in [';', '&&', '`', '|']):
        return 'Command Injection'
    # Check for parameter name tampering vs value tampering
    return 'Param Tampering / Unknown'


def error_analysis(y_test, test_prob, threshold, test_recs, out_dir):
    y_pred = (test_prob >= threshold).astype(int)

    fp_idx = [i for i, (yt, yp) in enumerate(zip(y_test, y_pred)) if yt == 0 and yp == 1]
    fn_idx = [i for i, (yt, yp) in enumerate(zip(y_test, y_pred)) if yt == 1 and yp == 0]

    lines = ["=" * 70, "ERROR ANALYSIS", "=" * 70]

    lines.append(f"\n--- 10 False Positives (normal traffic flagged as attack) ---")
    fp_sample = fp_idx[:10]
    for rank, i in enumerate(fp_sample, 1):
        lines.append(f"\nFP #{rank} (prob={test_prob[i]:.4f}):")
        lines.append(_request_summary(test_recs[i]))

    lines.append(f"\n--- 10 False Negatives (attacks missed) ---")
    fn_sample = fn_idx[:10]
    for rank, i in enumerate(fn_sample, 1):
        cat = _categorise_miss(test_recs[i])
        lines.append(f"\nFN #{rank} (prob={test_prob[i]:.4f}, type={cat}):")
        lines.append(_request_summary(test_recs[i]))

    # Group FNs by category
    lines.append("\n--- FN category breakdown ---")
    cats = {}
    for i in fn_idx:
        c = _categorise_miss(test_recs[i])
        cats[c] = cats.get(c, 0) + 1
    for c, cnt in sorted(cats.items(), key=lambda x: -x[1]):
        lines.append(f"  {c}: {cnt}")

    report_text = '\n'.join(lines)
    out_path = os.path.join(out_dir, 'error_analysis.txt')
    with open(out_path, 'w') as f:
        f.write(report_text)
    print(report_text[:2000])   # show first 2000 chars
    print(f"\n  Full error analysis saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate():
    print("=" * 60)
    print("Loading artifacts...")
    test_prob, y_test, hold_prob, y_hold, config, test_recs, lgbm, fx = _load_artifacts()
    threshold = config['threshold']

    print(f"Threshold: {threshold:.4f}")
    y_pred = (test_prob >= threshold).astype(int)

    print("\n--- Classification Report (test set) ---")
    print(classification_report(y_test, y_pred, target_names=['normal', 'attack']))

    print("\nGenerating plots...")
    os.makedirs(REPORTS, exist_ok=True)
    plot_roc_pr(y_test, test_prob, REPORTS)
    plot_feature_importance(lgbm, fx, REPORTS)
    plot_confusion_matrix(y_test, y_pred, REPORTS)

    print("\nError analysis...")
    error_analysis(y_test, test_prob, threshold, test_recs, REPORTS)

    # Feature group analysis
    names = fx.feature_names()
    imps  = lgbm.feature_importances_
    groups = {
        'content_length': ['total_len', 'path_len', 'query_len', 'body_len'],
        'entropy': ['entropy'],
        'special_chars': [n for n in names if n.startswith('cnt_') or n.startswith('ratio_')],
        'encoding': ['pct_encoded_url', 'pct_encoded_body', 'double_encoding'],
        'injection_keywords': ['sql_count', 'xss_count', 'path_traversal', 'cmd_injection'],
        'structure': ['unknown_endpoint', 'unknown_param_count', 'missing_param_count',
                      'param_count_dev', 'val_len_zscore', 'unseen_chars_flag'],
    }
    print("\n--- Feature Group Importance ---")
    group_imps = {}
    for g, feat_names in groups.items():
        total = sum(imps[names.index(n)] for n in feat_names if n in names)
        group_imps[g] = total
    for g, v in sorted(group_imps.items(), key=lambda x: -x[1]):
        print(f"  {g:<25}: {v:>8.1f}")

    print("\nEvaluation complete.")


if __name__ == '__main__':
    evaluate()
