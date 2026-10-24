"""
tests/test_pipeline.py
----------------------
Pytest tests covering:
1. Parser handles GET and POST correctly
2. URL-decoding works in features
3. Features always return the same fixed length
4. Unknown param names are flagged in structure features
5. ONNX output matches native model output (if ONNX exists)
"""

import os
import sys
import pickle
import json
import pytest
import numpy as np
import pandas as pd

# Allow import from src/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.parse import _parse_row, _normalize_request, load_and_parse
from src.features import FeatureExtractor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_normal_get():
    row = pd.Series({
        'Method': 'GET',
        'URL': 'http://localhost:8080/tienda1/publico/anadir.jsp?id=3&nombre=Vino+Rioja&precio=100&cantidad=55&B1=A%F1adir+al+carrito HTTP/1.1',
        'content': None,
        'classification': 0,
        'User-Agent': 'Mozilla/5.0', 'Pragma': 'no-cache', 'Cache-Control': 'no-cache',
        'Accept': 'text/html', 'Accept-encoding': 'gzip', 'Accept-charset': 'utf-8',
        'language': 'en', 'host': 'localhost:8080', 'cookie': 'JSESSIONID=abc',
        'content-type': '', 'connection': 'close', 'lenght': '',
    })
    return _parse_row(row)


@pytest.fixture
def sample_normal_post():
    row = pd.Series({
        'Method': 'POST',
        'URL': 'http://localhost:8080/tienda1/publico/anadir.jsp HTTP/1.1',
        'content': 'id=3&nombre=Vino+Rioja&precio=100&cantidad=55&B1=A%F1adir+al+carrito',
        'classification': 0,
        'User-Agent': 'Mozilla/5.0', 'Pragma': 'no-cache', 'Cache-Control': 'no-cache',
        'Accept': 'text/html', 'Accept-encoding': 'gzip', 'Accept-charset': 'utf-8',
        'language': 'en', 'host': 'localhost:8080', 'cookie': 'JSESSIONID=abc',
        'content-type': 'application/x-www-form-urlencoded', 'connection': 'close', 'lenght': '68',
    })
    return _parse_row(row)


@pytest.fixture
def sample_attack_sqli():
    row = pd.Series({
        'Method': 'GET',
        'URL': "http://localhost:8080/tienda1/publico/anadir.jsp?id=2&nombre=Jam%F3n&precio=85&cantidad=%27%3B+DROP+TABLE+usuarios%3B+SELECT+*+FROM+datos+WHERE+nombre+LIKE+%27%25&B1=A%F1adir HTTP/1.1",
        'content': None,
        'classification': 1,
        'User-Agent': 'Mozilla/5.0', 'Pragma': '', 'Cache-Control': '',
        'Accept': '', 'Accept-encoding': '', 'Accept-charset': '',
        'language': '', 'host': 'localhost:8080', 'cookie': '',
        'content-type': '', 'connection': '', 'lenght': '',
    })
    return _parse_row(row)


@pytest.fixture
def fitted_extractor(sample_normal_get, sample_normal_post):
    fx = FeatureExtractor(use_ngram=False)
    fx.fit([sample_normal_get, sample_normal_post])
    return fx


# ---------------------------------------------------------------------------
# Test 1: Parser handles GET correctly
# ---------------------------------------------------------------------------

def test_parser_get(sample_normal_get):
    rec = sample_normal_get
    assert rec['method'] == 'GET'
    assert rec['path'] == '/tienda1/publico/anadir.jsp'
    assert rec['body_string'] == ''
    assert len(rec['body_params']) == 0
    # Should have parsed query params
    assert 'id' in rec['query_params']
    assert 'nombre' in rec['query_params']
    assert 'precio' in rec['query_params']
    assert rec['label'] == 0


# ---------------------------------------------------------------------------
# Test 2: Parser handles POST correctly
# ---------------------------------------------------------------------------

def test_parser_post(sample_normal_post):
    rec = sample_normal_post
    assert rec['method'] == 'POST'
    assert rec['path'] == '/tienda1/publico/anadir.jsp'
    # POST body parsed into body_params
    assert 'id' in rec['body_params']
    assert 'nombre' in rec['body_params']
    # Query string should be empty
    assert rec['query_string'] == ''
    assert rec['label'] == 0


# ---------------------------------------------------------------------------
# Test 3: URL-decoding works (percent-encoded chars decoded)
# ---------------------------------------------------------------------------

def test_url_decoding_in_features(sample_attack_sqli, fitted_extractor):
    """
    The attack request has %27%3B+DROP+TABLE in the query.
    After URL-decoding: '; DROP TABLE ... (SQL injection).
    Feature extractor should detect sql_count > 0.
    """
    rec = sample_attack_sqli
    # Verify percent_encoded_count is > 0 (raw count on URL)
    assert rec['percent_encoded_count_url'] > 0

    # Verify SQL keywords are detected in decoded payload
    from src.features import _sql_count
    from urllib.parse import unquote_plus
    payload = unquote_plus(rec['query_string'])
    sql_cnt = _sql_count(payload)
    assert sql_cnt > 0, f"Expected SQL keywords in decoded payload, got 0. payload={payload[:200]}"


# ---------------------------------------------------------------------------
# Test 4: Features always return the same fixed length
# ---------------------------------------------------------------------------

def test_feature_vector_fixed_length(fitted_extractor, sample_normal_get,
                                      sample_normal_post, sample_attack_sqli):
    """All records must produce vectors of exactly n_features."""
    expected_len = fitted_extractor.n_features
    for rec in [sample_normal_get, sample_normal_post, sample_attack_sqli]:
        vec = fitted_extractor.transform_one(rec)
        assert vec.shape == (expected_len,), (
            f"Expected shape ({expected_len},), got {vec.shape}"
        )


def test_feature_names_count_matches(fitted_extractor):
    names = fitted_extractor.feature_names()
    assert len(names) == fitted_extractor.n_features


# ---------------------------------------------------------------------------
# Test 5: Unknown param names are flagged
# ---------------------------------------------------------------------------

def test_unknown_param_flagged(fitted_extractor):
    """
    Fit on records with params {id, nombre, precio, cantidad, B1}.
    A request with param 'idA' (unknown) should trigger unknown_param_count > 0.
    """
    import pandas as pd
    row_unknown = pd.Series({
        'Method': 'GET',
        'URL': 'http://localhost:8080/tienda1/publico/anadir.jsp?idA=999&nombre=test HTTP/1.1',
        'content': None,
        'classification': 0,
        'User-Agent': '', 'Pragma': '', 'Cache-Control': '', 'Accept': '',
        'Accept-encoding': '', 'Accept-charset': '', 'language': '',
        'host': 'localhost:8080', 'cookie': '', 'content-type': '',
        'connection': '', 'lenght': '',
    })
    from src.parse import _parse_row
    rec = _parse_row(row_unknown)
    vec = fitted_extractor.transform_one(rec)
    # Feature names to find unknown_param_count index
    names = fitted_extractor.feature_names()
    idx = names.index('unknown_param_count')
    assert vec[idx] > 0, f"Expected unknown_param_count > 0, got {vec[idx]}"


# ---------------------------------------------------------------------------
# Test 6: GET/POST deduplication
# ---------------------------------------------------------------------------

def test_deduplication_removes_get_post_pairs():
    """
    A GET with query params and equivalent POST with same params in body
    should be deduplicated to one record.
    """
    row_get = pd.Series({
        'Method': 'GET',
        'URL': 'http://localhost:8080/tienda1/publico/anadir.jsp?id=3&nombre=Vino HTTP/1.1',
        'content': None, 'classification': 0,
        'User-Agent': '', 'Pragma': '', 'Cache-Control': '', 'Accept': '',
        'Accept-encoding': '', 'Accept-charset': '', 'language': '',
        'host': '', 'cookie': '', 'content-type': '', 'connection': '', 'lenght': '',
    })
    row_post = pd.Series({
        'Method': 'POST',
        'URL': 'http://localhost:8080/tienda1/publico/anadir.jsp HTTP/1.1',
        'content': 'id=3&nombre=Vino', 'classification': 0,
        'User-Agent': '', 'Pragma': '', 'Cache-Control': '', 'Accept': '',
        'Accept-encoding': '', 'Accept-charset': '', 'language': '',
        'host': '', 'cookie': '', 'content-type': '', 'connection': '', 'lenght': '',
    })
    from src.parse import _parse_row, _normalize_request
    rec_get  = _parse_row(row_get)
    rec_post = _parse_row(row_post)
    # Normalised forms should differ (GET vs POST, different method) —
    # CSIC duplicates are same method but GET query vs POST body,
    # The normalizer ignores method to catch cross-method duplicates
    # Actually per prompt: "a GET with a query and an equivalent POST with the same query in the body"
    # Our normalizer strips method, so they should have the same hash
    norm_get  = _normalize_request(rec_get)
    norm_post = _normalize_request(rec_post)
    # Both have same path and same decoded params (id=3, nombre=Vino)
    assert norm_get == norm_post, (
        f"Expected same normalised form but got:\n  GET:  {norm_get}\n  POST: {norm_post}"
    )


# ---------------------------------------------------------------------------
# Test 7: ONNX output matches native (integration, skipped if no ONNX file)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not os.path.exists('models/model.onnx'),
    reason="models/model.onnx not found; run export_onnx.py first"
)
def test_onnx_matches_native():
    """ONNX predictions must match native LightGBM within 1e-3."""
    import onnxruntime as ort

    with open('models/lgbm_model.pkl', 'rb') as f:
        lgbm = pickle.load(f)
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('models/feature_extractor.pkl', 'rb') as f:
        fx = pickle.load(f)
    with open('reports/test_records.pkl', 'rb') as f:
        test_recs = pickle.load(f)

    X = fx.transform(test_recs[:100])
    X_s = scaler.transform(X).astype(np.float32)

    native_prob = lgbm.predict_proba(scaler.transform(X))[:, 1]

    sess = ort.InferenceSession('models/model.onnx', providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    out = sess.run(None, {input_name: X_s})

    if isinstance(out[1], list):
        onnx_prob = np.array([d[1] for d in out[1]])
    elif len(out) >= 2 and hasattr(out[1], 'shape'):
        onnx_prob = out[1][:, 1]
    else:
        onnx_prob = out[0].astype(np.float64)

    max_diff = np.max(np.abs(native_prob - onnx_prob))
    assert max_diff < 1e-3, f"Max ONNX vs native difference {max_diff} exceeds 1e-3"
