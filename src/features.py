"""
src/features.py
---------------
Pure function: request record -> fixed-length numeric feature vector.

The same module is used during training and inference (no hidden state).
Learned artifacts (endpoint lookup, scaler params) are loaded from JSON files
saved during training.

Feature groups
--------------
1. Content features   : lengths, entropy, special-char counts/ratios, pct-encoding,
                        keyword injection flags (SQL, XSS, path-traversal, cmd-injection)
2. Structure-aware    : endpoint validity, unknown param names, param count deviation,
                        value-length z-score, unseen chars
3. (Optional) n-gram  : char 3-gram hashed features (switchable via USE_NGRAM env var
                        or constructor arg). Disabled by default for ONNX export compat.

Usage
-----
    from src.features import FeatureExtractor
    fx = FeatureExtractor()
    fx.fit(normal_train_records)           # learns endpoint/param lookup
    fx.save_lookup('models/lookup.json')
    vec = fx.transform_one(record)         # -> np.ndarray shape (N_FEATURES,)
    X   = fx.transform(records)            # -> np.ndarray shape (n, N_FEATURES)
"""

import re
import math
import json
import os
from collections import defaultdict
from urllib.parse import unquote_plus

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SQL_KEYWORDS = re.compile(
    r'\b(select|union|or|and|drop|insert|delete|update|sleep|'
    r'exec|cast|convert|declare|char|benchmark|load_file|outfile)\b|--|\\/\\*',
    re.IGNORECASE
)
XSS_PATTERNS = re.compile(
    r'<script|onerror|onload|javascript:|alert\s*\(|document\.cookie|eval\s*\(',
    re.IGNORECASE
)
PATH_TRAVERSAL = re.compile(r'\.\./|\.\.\\|etc/passwd|/etc/shadow|/proc/', re.IGNORECASE)
CMD_INJECTION = re.compile(r'[;|`]|&&|\$\(', re.IGNORECASE)
SPECIAL_CHARS = list("'\";<>-%()=/\\#")

USE_NGRAM_ENV = os.environ.get('USE_NGRAM', '0') == '1'
NGRAM_N_FEATURES = 512   # hashed feature space size


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _shannon_entropy(s: str) -> float:
    """Shannon entropy of a string (bits per character)."""
    if not s:
        return 0.0
    counts = defaultdict(int)
    for c in s:
        counts[c] += 1
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _special_char_counts(s: str) -> list[int]:
    return [s.count(c) for c in SPECIAL_CHARS]


def _ratio(count: int, total: int) -> float:
    return count / total if total > 0 else 0.0


def _sql_count(s: str) -> int:
    return len(SQL_KEYWORDS.findall(s))


def _xss_count(s: str) -> int:
    return len(XSS_PATTERNS.findall(s))


def _path_traversal_count(s: str) -> int:
    return len(PATH_TRAVERSAL.findall(s))


def _cmd_injection_count(s: str) -> int:
    return len(CMD_INJECTION.findall(s))


# ---------------------------------------------------------------------------
# FeatureExtractor
# ---------------------------------------------------------------------------

class FeatureExtractor:
    """
    Stateful only in its learned endpoint/param lookup and the optional n-gram
    vectoriser. The transform functions are otherwise pure.
    """

    def __init__(self, use_ngram: bool = USE_NGRAM_ENV):
        self.use_ngram = use_ngram
        # lookup: {endpoint: {'params': set, 'param_counts': [int,...],
        #                      'param_stats': {name: {'lengths': [], 'chars': set}}}}
        self._lookup: dict = {}
        self._fitted = False

        if self.use_ngram:
            self._hv = HashingVectorizer(
                analyzer='char', ngram_range=(3, 3),
                n_features=NGRAM_N_FEATURES, norm='l2', alternate_sign=False
            )

    # ------------------------------------------------------------------
    # Fit (on normal training records only)
    # ------------------------------------------------------------------

    def fit(self, normal_records: list[dict]) -> 'FeatureExtractor':
        """Learn endpoint/parameter statistics from normal traffic only."""
        raw: dict[str, dict] = {}
        for rec in normal_records:
            ep = rec['path']
            if ep not in raw:
                raw[ep] = {
                    'param_counts': [],
                    'param_stats': defaultdict(lambda: {'lengths': [], 'chars': set()}),
                }
            all_params = {}
            all_params.update(rec.get('query_params', {}))
            all_params.update(rec.get('body_params', {}))
            raw[ep]['param_counts'].append(len(all_params))
            for name, val in all_params.items():
                val_str = val if isinstance(val, str) else str(val)
                decoded = unquote_plus(val_str)
                raw[ep]['param_stats'][name]['lengths'].append(len(decoded))
                raw[ep]['param_stats'][name]['chars'].update(decoded)

        # Compact representation
        for ep, data in raw.items():
            counts = data['param_counts']
            mean_c = float(np.mean(counts))
            std_c = float(np.std(counts)) if len(counts) > 1 else 0.0

            param_stats = {}
            for name, stats in data['param_stats'].items():
                lengths = stats['lengths']
                param_stats[name] = {
                    'mean_len': float(np.mean(lengths)),
                    'std_len': float(np.std(lengths)) if len(lengths) > 1 else 0.0,
                    'chars': list(stats['chars']),
                }

            self._lookup[ep] = {
                'valid_params': list(param_stats.keys()),
                'mean_param_count': mean_c,
                'std_param_count': std_c,
                'param_stats': param_stats,
            }

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_lookup(self, path: str) -> None:
        with open(path, 'w') as f:
            json.dump(self._lookup, f)

    def load_lookup(self, path: str) -> 'FeatureExtractor':
        with open(path) as f:
            self._lookup = json.load(f)
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Feature extraction (core)
    # ------------------------------------------------------------------

    def _content_features(self, rec: dict) -> list[float]:
        """Group 1: content / payload features."""
        method = rec.get('method', '')
        path = rec.get('path', '')
        qs = rec.get('query_string', '')
        body = rec.get('body_string', '')
        decoded_payload = unquote_plus(qs + ' ' + body)

        total_len = len(path) + len(qs) + len(body)
        path_len = len(path)
        query_len = len(qs)
        body_len = len(body)

        entropy = _shannon_entropy(decoded_payload)

        spc = _special_char_counts(decoded_payload)
        total_chars = max(len(decoded_payload), 1)
        spc_ratio = [c / total_chars for c in spc]
        digit_ratio = _ratio(sum(c.isdigit() for c in decoded_payload), total_chars)
        upper_ratio = _ratio(sum(c.isupper() for c in decoded_payload), total_chars)
        nonascii_ratio = _ratio(sum(ord(c) > 127 for c in decoded_payload), total_chars)

        pct_url = int(rec.get('percent_encoded_count_url', 0))
        pct_body = int(rec.get('percent_encoded_count_body', 0))
        double_enc = float(rec.get('has_double_encoding', False))

        sql_cnt = _sql_count(decoded_payload)
        xss_cnt = _xss_count(decoded_payload)
        path_trav = _path_traversal_count(decoded_payload)
        cmd_inj = _cmd_injection_count(decoded_payload)

        # method one-hot (GET=1,0 POST=0,1 other=0,0)
        is_get = float(method == 'GET')
        is_post = float(method == 'POST')

        feats = (
            [total_len, path_len, query_len, body_len, entropy]
            + spc                  # 15 special char counts
            + spc_ratio            # 15 special char ratios
            + [digit_ratio, upper_ratio, nonascii_ratio]
            + [pct_url, pct_body, double_enc]
            + [sql_cnt, xss_cnt, path_trav, cmd_inj]
            + [is_get, is_post]
        )
        return feats  # length = 5+15+15+3+3+4+2 = 47

    def _structure_features(self, rec: dict) -> list[float]:
        """Group 2: structure-aware features (requires fitted lookup)."""
        ep = rec.get('path', '')
        all_params: dict = {}
        all_params.update(rec.get('query_params', {}))
        all_params.update(rec.get('body_params', {}))

        if not self._fitted or ep not in self._lookup:
            # unknown endpoint
            unknown_endpoint = 1.0
            unknown_param_count = float(len(all_params))
            missing_param_count = 0.0
            param_count_dev = 0.0
            val_len_zscore = 0.0
            unseen_chars_flag = 0.0
        else:
            info = self._lookup[ep]
            valid_params = set(info['valid_params'])
            unknown_endpoint = 0.0

            unknown_param_count = float(
                sum(1 for p in all_params if p not in valid_params)
            )
            missing_param_count = float(
                sum(1 for p in valid_params if p not in all_params)
            )

            mean_c = info['mean_param_count']
            std_c = info['std_param_count']
            actual_c = len(all_params)
            param_count_dev = abs(actual_c - mean_c) / (std_c + 1e-6) if std_c > 0 else 0.0

            # Average value-length z-score across known params
            zscores = []
            unseen_any = 0
            for name, val in all_params.items():
                if name in info.get('param_stats', {}):
                    pstat = info['param_stats'][name]
                    decoded = unquote_plus(val if isinstance(val, str) else str(val))
                    vlen = len(decoded)
                    zscore = abs(vlen - pstat['mean_len']) / (pstat['std_len'] + 1e-6)
                    zscores.append(min(zscore, 100.0))   # cap outliers
                    seen_chars = set(pstat['chars'])
                    if any(c not in seen_chars for c in decoded):
                        unseen_any = 1
            val_len_zscore = float(np.mean(zscores)) if zscores else 0.0
            unseen_chars_flag = float(unseen_any)

        return [
            unknown_endpoint,
            unknown_param_count,
            missing_param_count,
            param_count_dev,
            val_len_zscore,
            unseen_chars_flag,
        ]  # length = 6

    def _ngram_features(self, rec: dict) -> np.ndarray:
        """Group 3: hashed char 3-gram features."""
        payload = rec.get('query_string', '') + ' ' + rec.get('body_string', '')
        mat = self._hv.transform([payload])
        return mat.toarray().flatten()   # length = NGRAM_N_FEATURES

    # ------------------------------------------------------------------
    # Public transform API
    # ------------------------------------------------------------------

    def transform_one(self, rec: dict) -> np.ndarray:
        """Transform a single record to a feature vector."""
        feats = self._content_features(rec) + self._structure_features(rec)
        vec = np.array(feats, dtype=np.float32)
        if self.use_ngram:
            ng = self._ngram_features(rec).astype(np.float32)
            vec = np.concatenate([vec, ng])
        return vec

    def transform(self, records: list[dict]) -> np.ndarray:
        """Transform a list of records into a 2D feature matrix."""
        return np.stack([self.transform_one(r) for r in records])

    @property
    def n_features(self) -> int:
        # Compute dynamically to avoid hardcoded constant mismatch
        n_spc = len(SPECIAL_CHARS)
        base = 5 + n_spc + n_spc + 3 + 3 + 4 + 2 + 6  # content + structure
        return base + (NGRAM_N_FEATURES if self.use_ngram else 0)

    def feature_names(self) -> list[str]:
        """Return human-readable feature names (useful for importance plots)."""
        spc_names = [f"cnt_spc{i}" for i in range(len(SPECIAL_CHARS))]
        spc_ratio_names = [f"ratio_spc{i}" for i in range(len(SPECIAL_CHARS))]
        names = (
            ['total_len', 'path_len', 'query_len', 'body_len', 'entropy']
            + spc_names
            + spc_ratio_names
            + ['digit_ratio', 'upper_ratio', 'nonascii_ratio']
            + ['pct_encoded_url', 'pct_encoded_body', 'double_encoding']
            + ['sql_count', 'xss_count', 'path_traversal', 'cmd_injection']
            + ['is_get', 'is_post']
            + ['unknown_endpoint', 'unknown_param_count',
               'missing_param_count', 'param_count_dev',
               'val_len_zscore', 'unseen_chars_flag']
        )
        if self.use_ngram:
            names += [f'ngram_{i}' for i in range(NGRAM_N_FEATURES)]
        assert len(names) == self.n_features, (
            f"feature_names length {len(names)} != n_features {self.n_features}"
        )
        return names
