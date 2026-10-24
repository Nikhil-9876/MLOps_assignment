"""
src/parse.py
------------
Parse the CSIC 2010 CSV dataset into structured request records.

Dataset format (discovered during inspection):
  - Single CSV file: dataset/csic_database.csv
  - Columns: Unnamed:0 (index/class label string), Method, User-Agent, Pragma,
    Cache-Control, Accept, Accept-encoding, Accept-charset, language, host,
    cookie, content-type, connection, lenght (typo), content, classification, URL
  - classification: 0 = normal, 1 = attack
  - URL column contains the full URL + " HTTP/1.1" suffix
  - content column holds URL-encoded body for POST requests (NaN for GET)

Each request is parsed into a dict with keys:
  method, path, query_string, query_params, body_string, body_params,
  headers, label, raw_url, percent_encoded_count_url, percent_encoded_count_body
"""

import re
import hashlib
from urllib.parse import urlparse, parse_qs, unquote_plus
import pandas as pd

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_PCT_RE = re.compile(r'%[0-9A-Fa-f]{2}')
_DOUBLE_PCT_RE = re.compile(r'%25[0-9A-Fa-f]{2}', re.IGNORECASE)  # %25XX == double-encoded


def _count_pct_encoded(s: str) -> int:
    """Count the number of percent-encoded sequences in a raw string."""
    if not s:
        return 0
    return len(_PCT_RE.findall(s))


def _has_double_encoding(s: str) -> bool:
    """Return True if the string contains double-percent-encoded sequences."""
    if not s:
        return False
    return bool(_DOUBLE_PCT_RE.search(s))


def _parse_query(query_string: str) -> dict:
    """Parse a query string into {param: [values, ...]} using stdlib."""
    if not query_string:
        return {}
    parsed = parse_qs(query_string, keep_blank_values=True)
    # Flatten single-value lists for convenience; keep multi-value as list
    return {k: v[0] if len(v) == 1 else v for k, v in parsed.items()}


def _extract_path_and_query(url_field: str):
    """
    Extract (path, query_string) from the URL column value.
    The URL column contains entries like:
        http://localhost:8080/tienda1/index.jsp HTTP/1.1
    or  http://localhost:8080/tienda1/publico/anadir.jsp?id=3&... HTTP/1.1
    """
    # Strip HTTP version suffix
    raw_url = re.sub(r'\s+HTTP/\d+\.\d+\s*$', '', url_field.strip())
    try:
        parsed = urlparse(raw_url)
        path = parsed.path or '/'
        query_string = parsed.query or ''
    except Exception:
        path = '/'
        query_string = ''
    return raw_url, path, query_string


def _parse_row(row: pd.Series) -> dict:
    """Convert a single DataFrame row into a structured request record."""
    method = str(row.get('Method', '')).strip().upper()

    # URL / path / query
    url_field = str(row.get('URL', ''))
    raw_url, path, query_string = _extract_path_and_query(url_field)
    query_params = _parse_query(query_string)

    # Body
    body_raw = row.get('content', None)
    body_string = '' if pd.isna(body_raw) else str(body_raw).strip()
    body_params = _parse_query(body_string) if body_string else {}

    # Headers dict (everything that isn't method/url/content/classification)
    header_cols = [
        'User-Agent', 'Pragma', 'Cache-Control', 'Accept',
        'Accept-encoding', 'Accept-charset', 'language', 'host',
        'cookie', 'content-type', 'connection', 'lenght'
    ]
    headers = {}
    for col in header_cols:
        val = row.get(col, None)
        if val is not None and not (isinstance(val, float) and pd.isna(val)):
            headers[col] = str(val).strip()

    # Percent-encoding counts (on raw strings, before URL-decoding)
    pct_url = _count_pct_encoded(query_string)
    pct_body = _count_pct_encoded(body_string)
    double_enc = _has_double_encoding(query_string) or _has_double_encoding(body_string)

    label = int(row['classification'])  # 0=normal, 1=attack

    return {
        'method': method,
        'raw_url': raw_url,
        'path': path,
        'query_string': query_string,
        'query_params': query_params,
        'body_string': body_string,
        'body_params': body_params,
        'headers': headers,
        'label': label,
        'percent_encoded_count_url': pct_url,
        'percent_encoded_count_body': pct_body,
        'has_double_encoding': double_enc,
    }


def _normalize_request(record: dict) -> str:
    """
    Return a canonical string for deduplication.

    Per prompt §2: "CSIC has duplicate pairs (a GET with a query and an
    equivalent POST with the same query in the body). Deduplicate on a
    normalised form of the request before splitting."

    We intentionally DROP the method from the normalised form so that a GET
    with query params and a POST with the same params in the body produce the
    same hash and are treated as duplicates.
    """
    path = unquote_plus(record['path'])

    # Merge query and body params (decoded) and sort
    all_params = {}
    all_params.update(record['query_params'])
    all_params.update(record['body_params'])
    param_str = '&'.join(
        f"{k}={unquote_plus(v) if isinstance(v, str) else v}"
        for k, v in sorted(all_params.items())
    )
    return f"{path}|{param_str}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_and_parse(csv_path: str) -> list[dict]:
    """
    Load the CSIC CSV, parse every row into a structured record,
    deduplicate on normalised form, and return a list of records.

    Deduplication strategy (per prompt §2):
      CSIC has GET/POST duplicate pairs — same params in query vs body.
      We normalise each request (method-agnostic path+sorted-params) and
      keep only the first occurrence, so the same logical request cannot
      appear in both train and test after splitting.

    Assumption: 'classification' column: 0=normal, 1=attack.
    """
    df = pd.read_csv(csv_path, low_memory=False)

    records = []
    seen_hashes: set[str] = set()

    for _, row in df.iterrows():
        rec = _parse_row(row)
        norm = _normalize_request(rec)
        h = hashlib.md5(norm.encode()).hexdigest()
        if h not in seen_hashes:
            seen_hashes.add(h)
            records.append(rec)

    return records


if __name__ == '__main__':
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else 'dataset/csic_database.csv'
    records = load_and_parse(csv_path)
    n = sum(1 for r in records if r['label'] == 0)
    a = sum(1 for r in records if r['label'] == 1)
    print(f"Parsed {len(records)} unique records: {n} normal, {a} attack")
    print("\nSample normal:")
    for r in records[:2]:
        print(f"  {r['method']} {r['path'][:60]} params={list(r['query_params'].keys())[:5]}")
    print("\nSample attack:")
    attacks = [r for r in records if r['label'] == 1]
    for r in attacks[:2]:
        print(f"  {r['method']} {r['path'][:60]} params={list(r['query_params'].keys())[:5]}")
        qs = r['query_string'][:80]
        print(f"    query_string={qs}")
