# Methodology Specification: JCCD-X-V6

## 1. System Overview

JCCD-X-V6 is a **hybrid code clone detection pipeline** that combines rule-based detection for Type-1/2 clones with machine learning classification for Type-3 clones. The system processes Java source code pairs and outputs clone classifications with confidence scores.

### Hybrid Design Rationale

- **Algorithmic hybridity**: Deterministic hashing for exact matches (Type-1/2) + statistical ML for structural similarity (Type-3)
- **Implementation hybridity**: Zig for performance-critical preprocessing/features, Python for data orchestration/ML

### Pipeline Flow

```
Raw Java Files → Normalization → Tokenization → Feature Engineering → Shingling/MinHash → LSH Filtering → Rule-Based Detection (Type-1/2) → ML Classification (Type-3) → Evaluation
```

---

## 2. Detailed Pipeline Specification

### 2.1 Data Preparation

| Aspect | Details |
|--------|---------|
| **Purpose** | Load TOMA dataset, validate source files, split into training/test sets |
| **Input** | Raw CSV files (type-1.csv through nonclone.csv) + Java source files |
| **Output** | training_dataset.csv (binary balanced) + testing_dataset.csv (multi-class) |
| **Algorithm** | Stratified 70/30 split on Type-3/4 and Type-5/Non-clone pairs; Type-1/2 pairs added 100% to test set |
| **Parameters** | test_ratio=0.3, seed=42, balanced training via undersampling |
| **Implementation** | Validation removes pairs with missing source files; binary training labels (0/1) vs multi-class test labels (0/3 for non-clone/Type-3) |

### 2.2 Source Code Normalization

| Aspect | Details |
|--------|---------|
| **Purpose** | Remove comments and normalize whitespace for Type-1 detection |
| **Input** | Raw Java source files |
| **Output** | Normalized source files (comments removed, whitespace collapsed) |
| **Algorithm** | Two-pass normalization implemented in Zig finite-state machine |
| **Pass 1** | Comment removal using states for normal/string/char/line-comment/block-comment |
| **Pass 2** | Whitespace normalization collapsing spaces/tabs, removing blank lines |
| **Key Features** | Preserves string/char literals, handles escape sequences, single-pass optimization for small files (<64KB) |

### 2.3 Tokenization & CST Generation

| Aspect | Details |
|--------|---------|
| **Purpose** | Parse normalized code into CST and extract abstract token sequences |
| **Input** | Normalized Java source code |
| **Output** | Token sequences (abstract categories) + AST statistics |
| **Algorithm** | tree-sitter parser with Java grammar, depth-first traversal with node classification |
| **Categories** | 12 abstract categories (KEYWORD=0, IDENTIFIER=1, OPERATOR=2, LITERAL=3, MODIFIER=4, TYPE=5, SEPARATOR=6, DECLARATION=7, EXPRESSION=8, STATEMENT=9, ANNOTATION=10, OTHER=255) |
| **Notes** | Only leaf tokens added to token sequences; all nodes contribute to AST statistics |

### 2.4 Feature Engineering

| Aspect | Details |
|--------|---------|
| **Purpose** | Compute 26 pairwise similarity features per candidate pair |
| **Input** | Token sequences + AST statistics for file pairs |
| **Output** | Feature matrix (26 dimensions) |
| **Algorithm** | Hybrid Zig/Python computation (Zig for 23 features, Python for 3 features) |
| **Parallelization** | Parallel thread pool in Zig, sequential Python features |
| **Breakdown** | 5 lexical (Zig) + 1 lexical (Python) + 7 structural (Zig) + 12 quantitative (Zig+Python) |

### 2.5 Shingling & MinHash

| Aspect | Details |
|--------|---------|
| **Purpose** | Convert token sequences to fixed-size signatures for LSH |
| **Input** | Token sequences per file |
| **Output** | MinHash signatures (128 dimensions default) |
| **Algorithm** | k-gram shingling (k=3) with FNV-1a hashing, universal hashing for MinHash |
| **Parameters** | k=3 (shingle size), num_hashes=36 (default), seed=42 |
| **FNV-1a Constants** | offset_basis=14695981039346656037, prime=1099511628211 |
| **MinHash Prime** | p=0xFFFFFFFFFFFFFFC5 (2^64 - 59) |

### 2.6 LSH Candidate Filtering

| Aspect | Details |
|--------|---------|
| **Purpose** | Reduce quadratic pairwise comparisons to linear candidate set |
| **Input** | MinHash signatures |
| **Output** | Candidate pairs (reduced by ~50%) |
| **Algorithm** | Banding technique with FNV-1a bucket hashing |
| **Parameters** | bands=12, rows_per_band=num_hashes//bands, max_bucket_size=5000 |
| **Candidate Selection** | Two files are candidates if they share any bucket ID |

### 2.7 Rule-Based Detection

| Aspect | Details |
|--------|---------|
| **Purpose** | Identify Type-1/2 clones via exact matching |
| **Input** | Normalized source files + token sequences |
| **Output** | Type-1/2 classified pairs |
| **Type-1 Detection** | MD5 hash matching on normalized source (byte-identical) |
| **Type-2 Detection** | Token sequence equality (identical abstract tokens, different source) |

### 2.8 ML Classification

| Aspect | Details |
|--------|---------|
| **Purpose** | Classify remaining pairs as Type-3 or non-clone |
| **Input** | 26-feature matrix (reduced to 3 via RFE) |
| **Output** | Binary classification (0=non-clone, 1=Type-3) with probabilities |
| **Algorithm** | XGBoost gradient boosting classifier |
| **Hyperparameters** | n_estimators=200, max_depth=6, learning_rate=0.1, eval_metric='logloss', random_state=42 |
| **RFE Features** | lex_dice_tokens, struct_node_count2, quant_cc2 |
| **Classification Threshold** | 0.30 |

---

## 3. Feature Engineering Specification

### 3.1 Lexical Features (7)

| # | Feature Name | Description | Implementation |
|---|--------------|-------------|----------------|
| 1 | lex_levenshtein_ratio | Normalized Levenshtein similarity | Python (rapidfuzz) |
| 2 | lex_token_set_ratio | Token-set ratio | Zig (Jaccard-like) |
| 3 | lex_sequence_match_ratio | LCS ratio approximation | Zig (same as lex_lcs_ratio) |
| 4 | lex_jaccard_tokens | Jaccard similarity of token sets | Zig (8192-bit bitmap) |
| 5 | lex_dice_tokens | Dice coefficient of token sets | Zig (8192-bit bitmap) |
| 6 | lex_lcs_ratio | Longest common subsequence ratio | Zig (max 512 tokens) |
| 7 | lex_char_jaccard | Character-level Jaccard similarity | Python |

### 3.2 Structural Features (8)

| # | Feature Name | Description | Implementation |
|---|--------------|-------------|----------------|
| 8 | struct_ast_hist_cosine | Cosine similarity of AST node-type histograms | Zig |
| 9 | struct_bigram_cosine | Jaccard similarity of parent-child bigram sets | Zig (O(n²) comparison) |
| 10 | struct_depth_diff | Absolute difference of max AST depths | Zig |
| 11 | struct_depth_ratio | Ratio of min/max depth | Zig |
| 12 | struct_node_count1 | Node count of first snippet | Zig |
| 13 | struct_node_count2 | Node count of second snippet | Zig |
| 14 | struct_node_ratio | Ratio of min/max node counts | Zig |
| 15 | struct_node_diff | Absolute difference of node counts | Zig |

### 3.3 Quantitative Features (13)

| # | Feature Name | Description | Implementation |
|---|--------------|-------------|----------------|
| 16 | quant_token_ratio | Ratio of token counts | Zig |
| 17 | quant_line_delta | Normalized line-count difference | Python |
| 18 | quant_identifier_ratio | Dice coefficient on IDENTIFIER tokens (type=1) | Zig (approximation) |
| 19 | quant_cc1 | Cyclomatic complexity of first snippet | Zig |
| 20 | quant_cc2 | Cyclomatic complexity of second snippet | Zig |
| 21 | quant_cc_ratio | Ratio of min/max CC | Zig |
| 22 | quant_cc_diff | Absolute CC difference | Zig |
| 23 | quant_lines1 | Line count of first snippet | Python |
| 24 | quant_lines2 | Line count of second snippet | Python |
| 25 | quant_chars1 | Character count of first snippet | Python |
| 26 | quant_chars2 | Character count of second snippet | Python |
| 27 | quant_char_ratio | Ratio of min/max char counts | Python |

---

## 4. Model Training Specification

### 4.1 Model Details

| Aspect | Details |
|--------|---------|
| **Selected Model** | XGBoost (version 2.0) |
| **Hyperparameters** | n_estimators=200, max_depth=6, learning_rate=0.1, n_jobs=-1, random_state=42 |
| **Training Data** | Balanced binary dataset (~150k pairs, 75k each class) |
| **Feature Selection** | RFE with 5-fold CV reduced to 3 features |

### 4.2 Training Process

1. **Cross-validation**: 5-fold stratified CV on full training set
2. **Model Comparison**: XGBoost, Random Forest, Logistic Regression, Linear SVM, KNN
3. **Model Selection**: Best model by F1-score selected
4. **Retraining**: Selected model retrained on full training set
5. **RFE Analysis**: Feature importance analysis to identify optimal subset
6. **Final Model**: Retrained on RFE-selected features and saved

### 4.3 Validation Strategy

| Aspect | Details |
|--------|---------|
| **Cross-validation** | Stratified 5-fold CV with random_state=42 |
| **Primary Metric** | F1-score (harmonic mean of precision/recall) |
| **Evaluation Set** | Binary subset of test set (Type-3 vs Non-clone only) |
| **Metrics Computed** | Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC |

---

## 5. LSH & Similarity Approximation

### 5.1 Technical Description

| Aspect | Details |
|--------|---------|
| **Hash Functions** | Universal hashing family: h_i(x) = (a_i * x + b_i) mod p |
| **Prime Modulus** | p = 0xFFFFFFFFFFFFFFC5 (2^64 - 59) |
| **Coefficient Generation** | hash_mix(seed, counter) using bit mixing operations |
| **Signature Construction** | Minimum hash value across all shingles per hash function |

### 5.2 Banding Logic

| Aspect | Details |
|--------|---------|
| **Signature Division** | Signatures divided into bands of rows_per_band elements |
| **Bucket Hashing** | FNV-1a hash of each band's rows to produce bucket ID |
| **Candidate Selection** | Two files are candidates if they share any bucket ID |
| **Max Bucket Size** | 5000 (prevents memory explosion for highly similar groups) |

### 5.3 Parameterization

| Configuration | Hashes | Bands | Rows/Band | Use Case |
|---------------|--------|-------|-----------|----------|
| **Default** | 36 | 12 | 3 | Balanced |
| **Balanced** | 48 | 16 | 3 | Standard |
| **High Recall** | 72 | 24 | 3 | Maximum recall |
| **Aggressive** | 40 | 8 | 5 | Maximum speed |

---

## 6. Data Processing & Splitting

### 6.1 Exact Methodology

1. Load all TOMA CSV files (type-1 through nonclone)
2. Validate source files exist (remove pairs with missing files)
3. Assign multi-class labels for testing (0=non-clone, 1=Type-1, 2=Type-2, 3=Type-3)
4. Split Type-3/4 and Type-5/Non-clone pairs 70/30 (stratified by binary label)
5. Balance training set via undersampling majority class
6. Add 100% of Type-1/2 pairs to test set

### 6.2 Leakage Prevention

| Mechanism | Details |
|-----------|---------|
| **Early Splitting** | Split performed before any feature computation or normalization |
| **File-Level Processing** | Normalization/tokenization performed on unique files, not pairs |
| **Independent Signatures** | LSH signatures computed independently per file |
| **Strict Separation** | Train/test separation maintained throughout pipeline |

### 6.3 Data Statistics

| Dataset | Size | Description |
|---------|------|-------------|
| **Raw Dataset** | 73,319 files, ~530k pairs | Full TOMA dataset |
| **Training Set** | ~150k pairs | Balanced binary (Type-3 vs Non-clone) |
| **Test Set** | ~201k pairs | Multi-class (Type-1/2/3 + Non-clone) |

---

## 7. Implementation Details

### 7.1 Languages and Tools

| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| **Orchestration** | Python | 3.10-3.14 | Data handling, ML, evaluation |
| **Performance** | Zig | 0.11+ | Preprocessing, features, LSH |
| **Parsing** | tree-sitter | 0.23 | Java CST generation |

### 7.2 Performance Optimizations

| Optimization | Details |
|--------------|---------|
| **Parallelization** | Zig thread pool for feature computation (auto-detects CPU cores) |
| **Batch Processing** | All Zig components process data in batches |
| **Caching** | LSH signatures and shingles cached to disk (SHA-256 fingerprint) |
| **Memory Efficiency** | Flat arrays (numpy) for feature matrices |

### 7.3 Runtime Characteristics

| Component | Runtime | Technology |
|-----------|---------|------------|
| Normalization | ~2s | Zig |
| Tokenization | ~6s | Zig + tree-sitter |
| Feature Engineering | ~15s | Zig (parallel) |
| LSH Filtering | ~5s | Zig |
| ML Training | ~6s | Python (XGBoost) |
| **Total Pipeline** | **~41s** | **Hybrid** |

---

## 8. Missing / Assumed Details

### 8.1 Inferred Assumptions

| Issue | Details |
|-------|---------|
| **Feature Count Discrepancy** | Methodology states 26 features, but Zig implementation shows FEATURE_COUNT=25. One feature (lex_levenshtein_ratio) is computed in Python as placeholder 0.0 in Zig. |
| **XGBoost Hyperparameters** | Methodology mentions "default" but actual implementation uses n_estimators=200 (not default 100), learning_rate=0.1 (not default 0.3). |
| **Identifier Ratio** | Implemented as approximation (min(count_a, count_b)/max(count_a, count_b)) rather than true Dice coefficient. |
| **Cyclomatic Complexity** | Counts all keywords and operators, not just control flow keywords. |
| **Bigram Similarity** | Uses O(n²) comparison rather than set-based Jaccard. |

### 8.2 Missing Information

| Gap | Description |
|-----|-------------|
| **Node Type Mapping** | Only partial mapping shown; some tree-sitter node types may map to CATEGORY_OTHER. |
| **Thread Pool Config** | Auto-detects CPU cores but no explicit configuration. |
| **Cache Invalidation** | Based on SHA-256 fingerprint of token data. |
| **Error Handling** | Limited details on parsing errors or missing files. |

---

## 9. Issues & Risks

### 9.1 Potential Weaknesses

| Issue | Impact | Mitigation |
|-------|--------|------------|
| **Test Set Imbalance** | 100% Type-1/2 in test vs 30% Type-3/Non-clone may bias evaluation | Report per-class metrics |
| **Feature Inaccuracies** | Some features may not match mathematical definitions exactly | Document actual implementation |
| **Bigram Complexity** | O(n²) comparison could be slow for large ASTs | Limit AST size in practice |
| **Bitmap Collisions** | 8192-entry bitmap may cause hash collisions | Acceptable for similarity estimation |
| **Fixed LCS Length** | LCS limited to 512 tokens | Balances accuracy and performance |
| **Simplified CC** | Counts all keywords/operators | May overestimate complexity |

### 9.2 Methodological Concerns

| Concern | Details |
|---------|---------|
| **Leakage Risk** | Same file may appear in both train/test contexts via normalization |
| **Evaluation Bias** | Binary evaluation on subset may not reflect full pipeline performance |
| **Feature Selection** | RFE on training data may not generalize to test distribution |
| **LSH Recall** | Filtering may miss true clones, affecting recall |

---

## Appendix A: Source Code References

| Component | File Path |
|-----------|-----------|
| Normalization | src/zig/preprocessing/normalization.zig |
| Tokenization | src/python/preprocessing/tokenizer.py |
| Feature Engineering | src/python/feature_engineering/features.py, src/zig/features/features.zig |
| Shingling | src/zig/shingling/shingling.zig |
| MinHash/LSH | src/zig/lsh/minhash.zig |
| ML Training | src/python/model/train.py |
| Feature Selection | src/python/model/feature_selection.py |
| Pipeline Orchestration | src/python/pipeline/full_pipeline.py |
| Data Preparation | src/python/preprocessing/data_prep.py |

---

## Appendix B: Key Implementation Constants

### FNV-1a Hash Constants

| Constant | Value | Description |
|----------|-------|-------------|
| offset_basis | 14695981039346656037 | Initial hash value |
| prime | 1099511628211 | FNV prime for multiplication |

### MinHash Constants

| Constant | Value | Description |
|----------|-------|-------------|
| prime_modulus | 0xFFFFFFFFFFFFFFC5 | Large prime for modular arithmetic |
| max_signature | 0xFFFFFFFFFFFFFFFF | Initial signature value (max u64) |

### Feature Constants

| Constant | Value | Description |
|----------|-------|-------------|
| TOKEN_BITMAP_SIZE | 8192 | Size of token set bitmap |
| MAX_LCS_LENGTH | 512 | Maximum tokens for LCS computation |
| FEATURE_COUNT | 25 | Zig feature count (Python adds 1 more) |

---

*This methodology specification is based on direct analysis of the JCCD-X-V6 implementation (commit hash not available). All parameters and algorithms are extracted from source code unless explicitly marked as inferred.*
