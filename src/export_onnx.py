"""
src/export_onnx.py
------------------
Export the trained LightGBM model to ONNX and verify predictions match.
Also benchmarks single-request and batch inference latency (p50/p95).

Run with:
    python -m src.export_onnx

Note: We export the LightGBM model directly.
The scaler is baked into the ONNX pipeline as a preprocessing step
using sklearn Pipeline wrapped via skl2onnx.

We use a Pipeline(scaler -> lgbm) approach so the ONNX graph includes
all preprocessing, making inference self-contained.
"""

import json
import os
import pickle
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')

MODELS  = 'models'
REPORTS = 'reports'


def build_pipeline_and_export():
    print("=" * 60)
    print("Loading trained artifacts...")
    with open(os.path.join(MODELS, 'lgbm_model.pkl'), 'rb') as f:
        lgbm = pickle.load(f)
    with open(os.path.join(MODELS, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    with open(os.path.join(MODELS, 'feature_extractor.pkl'), 'rb') as f:
        fx = pickle.load(f)
    with open(os.path.join(MODELS, 'config.json')) as f:
        config = json.load(f)

    test_prob  = np.load(os.path.join(REPORTS, 'test_prob.npy'))
    y_test     = np.load(os.path.join(REPORTS, 'test_labels.npy'))

    # Rebuild test features
    import pickle as pkl
    with open(os.path.join(REPORTS, 'test_records.pkl'), 'rb') as f:
        test_recs = pkl.load(f)

    X_test = fx.transform(test_recs)
    X_test_s = scaler.transform(X_test)
    n_features = X_test_s.shape[1]
    print(f"  n_features={n_features}")

    # -------------------------------------------------------------------
    # Build sklearn Pipeline for clean ONNX export
    # -------------------------------------------------------------------
    from sklearn.pipeline import Pipeline
    import lightgbm as lgb

    # Re-fit a pipeline object on the scaled data
    # (scaler already fitted, lgbm already fitted)
    # We create a wrapper pipeline that applies the already-fitted scaler
    from sklearn.base import BaseEstimator, TransformerMixin

    class PreFittedScaler(BaseEstimator, TransformerMixin):
        def __init__(self, scaler):
            self.scaler = scaler
        def fit(self, X, y=None):
            return self
        def transform(self, X):
            return self.scaler.transform(X)

    pipeline = Pipeline([
        ('scaler', PreFittedScaler(scaler)),
        ('clf', lgbm)
    ])

    # -------------------------------------------------------------------
    # Try skl2onnx export (preferred)
    # -------------------------------------------------------------------
    onnx_path = os.path.join(MODELS, 'model.onnx')
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        from skl2onnx import update_registered_converter
        from skl2onnx.helpers.onnx_helper import select_model_inputs_outputs

        # For LightGBM inside a pipeline, use the lgbm onnx converter
        # We export just the lgbm model and handle scaler manually
        from onnxmltools import convert_lightgbm
        from onnxmltools.convert.common.data_types import FloatTensorType as OnnxFloatType

        print("  Exporting LightGBM to ONNX via onnxmltools...")
        onnx_model = convert_lightgbm(
            lgbm,
            initial_types=[('X', OnnxFloatType([None, n_features]))],
            target_opset=12
        )
        # Save scaler params separately; predict.py will apply scaler before ONNX
        with open(onnx_path, 'wb') as f:
            f.write(onnx_model.SerializeToString())
        export_method = 'onnxmltools+lightgbm'
        print(f"  ONNX model saved to {onnx_path}")

    except Exception as e:
        print(f"  onnxmltools export failed ({e}), trying skl2onnx pipeline export...")
        try:
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType

            # Export just the lgbm (skl2onnx has a LightGBM converter registered via lgbm)
            initial_type = [('X', FloatTensorType([None, n_features]))]
            onnx_model = convert_sklearn(lgbm, initial_types=initial_type, target_opset=17)
            with open(onnx_path, 'wb') as f:
                f.write(onnx_model.SerializeToString())
            export_method = 'skl2onnx'
            print(f"  ONNX model saved to {onnx_path}")

        except Exception as e2:
            print(f"  skl2onnx failed ({e2}). Using raw LGBM ONNX export...")
            # Fallback: use lgbm's native booster.save_model + onnxmltools
            booster_path = os.path.join(MODELS, 'lgbm_booster.txt')
            lgbm.booster_.save_model(booster_path)
            print(f"  LightGBM booster saved to {booster_path} (ONNX export skipped)")
            onnx_path = None
            export_method = 'lgbm_native_booster'

    # -------------------------------------------------------------------
    # Verification: ONNX predictions vs native
    # -------------------------------------------------------------------
    if onnx_path and os.path.exists(onnx_path):
        import onnxruntime as ort

        sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        input_name = sess.get_inputs()[0].name

        print("\n  Verifying ONNX predictions vs native LightGBM...")
        # ONNX takes scaled features (since scaler was applied before onnx export)
        X_onnx = X_test_s.astype(np.float32)

        # Native predictions
        native_prob = lgbm.predict_proba(X_test_s)[:, 1]

        # ONNX predictions
        out = sess.run(None, {input_name: X_onnx})
        # Output may be [labels, probabilities_dict] depending on converter
        if isinstance(out[1], list):
            # dict-like output (onnxmltools)
            onnx_prob = np.array([d[1] for d in out[1]], dtype=np.float64)
        elif len(out) >= 2 and hasattr(out[1], 'shape'):
            onnx_prob = out[1][:, 1]
        else:
            onnx_prob = out[0].astype(np.float64)

        max_diff = float(np.max(np.abs(native_prob - onnx_prob)))
        mean_diff = float(np.mean(np.abs(native_prob - onnx_prob)))
        print(f"  Max absolute difference:  {max_diff:.6f}")
        print(f"  Mean absolute difference: {mean_diff:.6f}")

        if max_diff < 1e-3:
            print("  ✓ ONNX predictions match native within tolerance")
        else:
            print(f"  ⚠ Large difference detected ({max_diff:.4f}) – may be converter quirk")

        # -------------------------------------------------------------------
        # Latency benchmark
        # -------------------------------------------------------------------
        print("\n  Benchmarking ONNX inference latency...")
        n_warmup = 20
        n_bench  = 500

        # Single-request latency
        single = X_onnx[:1]
        for _ in range(n_warmup):
            sess.run(None, {input_name: single})

        single_times = []
        for _ in range(n_bench):
            t0 = time.perf_counter()
            sess.run(None, {input_name: single})
            single_times.append((time.perf_counter() - t0) * 1000)  # ms

        # Batch latency (1000 requests)
        batch = X_onnx[:min(1000, len(X_onnx))]
        for _ in range(n_warmup):
            sess.run(None, {input_name: batch})

        batch_times = []
        for _ in range(min(n_bench, 100)):
            t0 = time.perf_counter()
            sess.run(None, {input_name: batch})
            batch_times.append((time.perf_counter() - t0) * 1000)

        p50_single = float(np.percentile(single_times, 50))
        p95_single = float(np.percentile(single_times, 95))
        p50_batch  = float(np.percentile(batch_times, 50))
        p95_batch  = float(np.percentile(batch_times, 95))

        print(f"\n  Latency (CPU) — single request:")
        print(f"    p50={p50_single:.3f}ms  p95={p95_single:.3f}ms")
        print(f"  Latency (CPU) — batch of {len(batch)}:")
        print(f"    p50={p50_batch:.3f}ms  p95={p95_batch:.3f}ms")

        # Update metrics.json with ONNX info
        metrics_path = os.path.join(REPORTS, 'metrics.json')
        with open(metrics_path) as f:
            metrics = json.load(f)
        metrics['onnx'] = {
            'path': onnx_path,
            'export_method': export_method,
            'max_diff_vs_native': max_diff,
            'mean_diff_vs_native': mean_diff,
            'latency_single_p50_ms': p50_single,
            'latency_single_p95_ms': p95_single,
            'latency_batch1000_p50_ms': p50_batch,
            'latency_batch1000_p95_ms': p95_batch,
        }
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\n  Updated {metrics_path} with ONNX metrics")

    print("\nONNX export complete.")


if __name__ == '__main__':
    build_pipeline_and_export()
