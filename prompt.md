You are helping me build the ML core of an MLOps assignment: a classifier that labels individual HTTP requests as "normal" or "attack". The dataset is CSIC 2010 and is already in this repo. Work step by step, and do not skip the inspection step.

## 0. Inspect first
- Explore the repo and find the dataset files. List them and show me a few raw examples of each class before writing any code.
- Tell me the file format (raw HTTP request text vs CSV), how requests are delimited, and how normal vs anomalous files are separated.
- Do NOT assume column names or file names. Adapt to what is actually there.

## 1. Goal
Build a reproducible pipeline: parse raw requests -> extract features -> train -> evaluate -> export to ONNX.
Primary metric is NOT accuracy. Report precision, recall, F1, ROC-AUC, PR-AUC, and especially the FALSE POSITIVE RATE on normal traffic, because a later CI quality gate will fail the build if FPR > 1.5%.

## 2. Parsing
- Write a parser that turns each raw request into a structured record: method, path/endpoint, query parameters (name -> value), body parameters (name -> value), headers, and the label.
- URL-decode values before feature extraction, but also keep a count of percent-encoded characters in the raw string.
- CSIC has duplicate pairs (a GET with a query and an equivalent POST with the same query in the body). Deduplicate on a normalized form of the request before splitting, so the same request cannot appear in both train and test.

## 3. Features (in a single reusable module, src/features.py)
Implement features as a pure function: request record -> fixed-length numeric vector, so the exact same code runs in training and later in the production worker.

Content features:
- total length, path length, query length, body length
- Shannon entropy of the decoded payload
- counts and ratios of special characters (' " ; < > - % ( ) = / \ # --), digit ratio, uppercase ratio, non-ASCII ratio
- count of percent-encoded sequences, and whether double-encoding appears
- keyword flags/counts: SQL (select, union, or, and, drop, insert, sleep, --, /*), XSS (<script, onerror, javascript:, alert), path traversal (../, etc/passwd), command injection (; | && `)
- optionally a char n-gram TF-IDF / hashed n-gram block (make this switchable, since it affects ONNX export)

Structure-aware features (important, because many CSIC anomalies are parameter tampering with no attack payload, e.g. a parameter named "idA" instead of "id"):
- Learn from the NORMAL TRAINING DATA ONLY a lookup of endpoint -> set of valid parameter names, endpoint -> typical number of parameters, and optionally parameter -> typical value length and character set.
- Features: unknown_endpoint, unknown_param_name_count, missing_expected_param_count, param_count_deviation, value_length_zscore_vs_normal_for_that_param, value_has_chars_never_seen_for_that_param.
- Save this learned lookup as an artifact (JSON) so inference can reuse it. Fit it on the training split only to avoid leakage.

## 4. Splits and leakage
- Do a stratified train/validation/test split after deduplication, with a fixed random seed.
- Keep a separate hold-out set of NORMAL requests that is used only for the FPR check (call it "prod_clean_holdout").
- Fit anything learned from data (lookups, scalers, TF-IDF) on train only.

## 5. Models
- Baseline: logistic regression on the features.
- Main model: XGBoost or LightGBM (pick one, note why). Tune a small set of hyperparameters with cross-validation on the training set.
- Compare both in a table. Keep it simple; no deep learning unless trivial to add.
- Choose the decision threshold on the validation set so that FPR on normal traffic is <= 1.0%, and report the recall you get at that threshold. Save the chosen threshold in a config file.

## 6. Evaluation
- Confusion matrix, classification report, ROC and PR curves (save as PNG), and FPR at the chosen threshold on the test set and on prod_clean_holdout.
- Feature importance plot, and a short analysis of which feature groups matter (content vs structure-aware).
- Error analysis: print 10 false positives and 10 false negatives with the raw request, and group the misses by type if possible (e.g. parameter-name tampering vs value tampering vs injection payloads).
- Save all metrics to metrics.json (machine-readable) so CI can read it later.

## 7. Export
- Export the final model to ONNX (onnxmltools / skl2onnx as appropriate) and verify that ONNX predictions match the native model's predictions on the test set within a small tolerance. Print the max difference.
- Benchmark single-request and batch inference latency with onnxruntime on CPU and report p50/p95.

## 8. Repo structure and quality
Use this layout (adapt if the repo already has one):
- data/ (raw data stays untouched), src/parse.py, src/features.py, src/train.py, src/evaluate.py, src/export_onnx.py, src/predict.py
- models/ (model.onnx, threshold + feature-lookup artifacts), reports/ (metrics.json, plots)
- tests/ with pytest: parser handles GET and POST, URL-decoding works, features always return the same fixed length, unknown param names are flagged, and ONNX output matches native model output.
- requirements.txt with pinned versions, a README section explaining how to run each step, and a Makefile or simple CLI (python -m src.train).
- Set all random seeds. Make the pipeline deterministic.
- Structure the stages so they can later be wrapped in a DVC pipeline (each stage reads inputs from files and writes outputs to files; no hidden global state).

## 9. Output expectations
- Do the work in small steps and run the code at each step to prove it works. Show me the metrics at the end in a compact table.
- Do not use accuracy as the headline metric. If results look suspiciously perfect (e.g. >99.9%), investigate leakage or generator artifacts and tell me what you find.
- If you must make an assumption, state it explicitly. Ask me before making any choice that significantly changes scope.