"""
src/predict.py
--------------
Production inference: given a raw request dict (same schema as parse.py output),
return a prediction dict with label and probability.

Can also run from the command line:
    python -m src.predict --method GET --url "http://localhost:8080/tienda1/index.jsp"
"""

import argparse
import json
import os
import pickle
import numpy as np

MODELS = 'models'


def load_predictor():
    """Load all artifacts needed for inference. Returns a Predictor instance."""
    return Predictor()


class Predictor:
    def __init__(self, models_dir: str = MODELS):
        with open(os.path.join(models_dir, 'lgbm_model.pkl'), 'rb') as f:
            self.model = pickle.load(f)
        with open(os.path.join(models_dir, 'scaler.pkl'), 'rb') as f:
            self.scaler = pickle.load(f)
        with open(os.path.join(models_dir, 'feature_extractor.pkl'), 'rb') as f:
            self.fx = pickle.load(f)
        with open(os.path.join(models_dir, 'config.json')) as f:
            config = json.load(f)
        self.threshold = config['threshold']

    def predict_one(self, record: dict) -> dict:
        """
        Predict a single request record.
        record must have at least: method, path, query_string, query_params,
                                   body_string, body_params, headers
        Returns: {label: 0|1, probability: float, is_attack: bool}
        """
        vec = self.fx.transform_one(record).reshape(1, -1)
        vec_s = self.scaler.transform(vec)
        prob = float(self.model.predict_proba(vec_s)[0, 1])
        label = int(prob >= self.threshold)
        return {'label': label, 'probability': prob, 'is_attack': bool(label)}

    def predict_batch(self, records: list[dict]) -> list[dict]:
        X = self.fx.transform(records)
        X_s = self.scaler.transform(X)
        probs = self.model.predict_proba(X_s)[:, 1]
        results = []
        for prob in probs:
            label = int(prob >= self.threshold)
            results.append({'label': label, 'probability': float(prob),
                            'is_attack': bool(label)})
        return results

    def predict_onnx(self, record: dict) -> dict:
        """Use ONNX runtime for inference (requires models/model.onnx)."""
        import onnxruntime as ort
        onnx_path = os.path.join(MODELS, 'model.onnx')
        if not hasattr(self, '_ort_sess'):
            self._ort_sess = ort.InferenceSession(
                onnx_path, providers=['CPUExecutionProvider']
            )
        sess = self._ort_sess
        input_name = sess.get_inputs()[0].name
        vec = self.fx.transform_one(record).reshape(1, -1)
        vec_s = self.scaler.transform(vec).astype(np.float32)
        out = sess.run(None, {input_name: vec_s})
        if isinstance(out[1], list):
            prob = float(out[1][0][1])
        elif len(out) >= 2 and hasattr(out[1], 'shape'):
            prob = float(out[1][0, 1])
        else:
            prob = float(out[0][0])
        label = int(prob >= self.threshold)
        return {'label': label, 'probability': prob, 'is_attack': bool(label),
                'backend': 'onnx'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict HTTP request label')
    parser.add_argument('--method', default='GET')
    parser.add_argument('--url', default='http://localhost:8080/tienda1/index.jsp')
    parser.add_argument('--body', default='')
    parser.add_argument('--onnx', action='store_true', help='Use ONNX backend')
    args = parser.parse_args()

    from src.parse import _parse_row
    import pandas as pd

    row = pd.Series({
        'Method': args.method,
        'URL': args.url + ' HTTP/1.1',
        'content': args.body if args.body else None,
        'classification': 0,
        'User-Agent': '', 'Pragma': '', 'Cache-Control': '', 'Accept': '',
        'Accept-encoding': '', 'Accept-charset': '', 'language': '',
        'host': 'localhost:8080', 'cookie': '', 'content-type': '',
        'connection': '', 'lenght': '',
    })
    record = _parse_row(row)

    predictor = Predictor()
    if args.onnx:
        result = predictor.predict_onnx(record)
    else:
        result = predictor.predict_one(record)

    print(json.dumps(result, indent=2))
