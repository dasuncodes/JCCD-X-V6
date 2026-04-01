# III. METHODOLOGY

## A. System Architecture Overview

JCCD-X-V6 employs a hybrid multi-language pipeline that integrates high-performance systems programming with machine learning orchestration to detect Type 1--3 syntactic code clones in Java source code. The system is organized into six distinct layers: (1) Data Layer, (2) Preprocessing Layer, (3) Feature Engineering Layer, (4) Locality-Sensitive Hashing (LSH) Filtering Layer, (5) Classification Layer, and (6) Evaluation Layer. Performance-critical operations---source code normalization, tokenization, feature computation, shingling, and MinHash-LSH---are implemented in Zig and compiled to shared libraries (`.so`), while data orchestration, machine learning, and evaluation are implemented in Python. The two language runtimes are bridged via Python's `ctypes` foreign function interface (FFI), enabling zero-copy transfer of contiguous NumPy arrays to Zig without serialization overhead.

Fig.~1 illustrates the end-to-end architecture. The pipeline proceeds sequentially through the following stages: (i)~data preparation and stratified splitting, (ii)~source code normalization (comment and whitespace removal), (iii)~abstract tokenization via tree-sitter concrete syntax tree (CST) parsing, (iv)~pairwise feature extraction (26 features), (v)~LSH-based candidate filtering, (vi)~rule-based Type-1/Type-2 detection, (vii)~machine learning classification for Type-3 clones, and (viii)~comprehensive evaluation including ablation studies, sensitivity analysis, and stability assessment.

> **Fig. 1.** *System architecture of JCCD-X-V6. The pipeline consists of six layers: Data, Preprocessing (Zig), Feature Engineering (Zig + Python), LSH Filtering (Zig), Classification (Python), and Evaluation (Python). Inter-language communication occurs via ctypes FFI with contiguous NumPy array buffers.*

## B. Dataset and Preprocessing

### B.1. Dataset Description

The system is evaluated on the TOMA (Table of Multiple Applications) dataset~\cite{toma}, a widely used benchmark for Java code clone detection. The dataset comprises 73,319 Java source files organized into six pair sets: Type-1 (48,116 pairs of semantically identical clones), Type-2 (4,234 pairs with renamed identifiers), Type-3 (21,395 pairs with modified statements), Type-4 (86,341 pairs with syntactically different but functionally similar code, treated as Type-3), Type-5 (109,914 pairs of non-clones), and a supplementary non-clone set (279,032 pairs). Each pair is represented by two function identifiers (`id1`, `id2`) mapping to Java source files in the `id2sourcecode` directory.

### B.2. Source File Validation

Prior to processing, all pairs are validated to ensure both constituent source files exist on disk. Pairs referencing missing files are removed. This validation step eliminates data integrity issues that could propagate through subsequent pipeline stages.

### B.3. Dataset Construction and Splitting

The training and testing sets are constructed with distinct label schemas to support both binary classification and multi-class evaluation:

**Training set (binary).** Type-3 and Type-4 pairs are assigned label~1 (clone), while Type-5 and non-clone pairs are assigned label~0 (non-clone). Type-1 and Type-2 pairs are excluded from the training set because they are detectable via deterministic rule-based methods and would introduce trivially separable classes. The combined set is split using stratified random sampling with a 70:30 ratio (random seed 42), where stratification is performed on the binary label to preserve class proportions. The training subset is subsequently balanced via undersampling of the majority class to achieve equal representation of positive and negative samples.

**Testing set (multi-class).** The 30\% holdout portion of Type-3/Type-4/Type-5/non-clone pairs is augmented with 100\% of Type-1 and Type-2 pairs, yielding a multi-class test set with four label categories: 0 (non-clone), 1 (Type-1), 2 (Type-2), and 3 (Type-3). This design enables end-to-end evaluation of the full hybrid pipeline, including rule-based detection stages.

> **Table I** summarizes the dataset composition after validation and the resulting train/test split sizes.

## C. Source Code Normalization

Source code normalization is the first preprocessing step, implemented in Zig for performance and exposed to Python via FFI. The normalization pipeline consists of two sequential passes:

**Comment removal.** The first pass strips single-line comments (`//`) and multi-line/block comments (`/* ... */`) from Java source. The algorithm operates as a single-pass state machine that tracks four mutually exclusive states: *in-line-comment*, *in-block-comment*, *in-string-literal*, and *in-char-literal*. String literals (`"..."`) and character literals (`'...'`) are preserved, with escape sequences (`\\`, `\"`, `\'`) correctly handled. Newline characters within block comments are retained to maintain line-number correspondence.

**Whitespace normalization.** The second pass collapses runs of spaces and tabs into a single space character, removes carriage returns, and eliminates blank lines (lines containing no non-whitespace content). A trailing newline is removed to ensure deterministic output. For files exceeding 65,536 bytes, both passes operate in-place on a shared output buffer to avoid heap allocation; smaller files use a stack-allocated temporary buffer.

The normalization output is written to `data/processed/normalized/` as individual `.java` files, one per function identifier. On the TOMA dataset, normalization removes an average of approximately 20\% of tokens and 30\% of lines per file, primarily attributable to comment removal and blank-line elimination.

## D. Tokenization and Abstract Representation

### D.1. Concrete Syntax Tree Parsing

Tokenization uses the tree-sitter parsing library~\cite{tree-sitter} with the `tree-sitter-java` grammar to generate concrete syntax trees (CSTs) for each normalized Java source file. Tree-sitter provides incremental, error-tolerant parsing, ensuring robust token extraction even for syntactically incomplete or malformed code fragments.

### D.2. Token Abstraction

Raw tree-sitter node types are mapped to a reduced set of 12 abstract token categories via a lookup table. This abstraction eliminates identifier-specific and literal-specific variation, enabling structural comparison across renamed code fragments. The categories are:

| Category ID | Label | Examples |
|:-----------:|:------|:---------|
| 0 | KEYWORD | `if`, `for`, `while`, `return`, `new`, `class` |
| 1 | IDENTIFIER | variable names, type names |
| 2 | OPERATOR | `+`, `==`, `&&`, `=`, `++` |
| 3 | LITERAL | integer, float, string, char, boolean, null literals |
| 4 | MODIFIER | `public`, `static`, `final`, `abstract` |
| 5 | TYPE | `void`, `int`, `boolean`, generic/array types |
| 6 | SEPARATOR | `(`, `)`, `{`, `}`, `[`, `]`, `;`, `,` |
| 7 | DECLARATION | method, class, field, variable declarations |
| 8 | EXPRESSION | binary, unary, method invocation, cast expressions |
| 9 | STATEMENT | `if`, `for`, `while`, `try`, `return` statements |
| 10 | ANNOTATION | `@Override`, `@Test`, marker annotations |
| 255 | OTHER | unmatched node types |

Only leaf nodes (nodes with zero children) are appended to the token sequence used for similarity computation. All nodes (internal and leaf) contribute to AST statistics.

### D.3. AST Statistics

For each parsed file, the following structural statistics are computed in Zig:

- **Node-type histogram.** A 12-bin histogram counting occurrences of each abstract token category across all AST nodes.
- **Parent-child bigrams.** For each non-root node, the pair *(parent-type, child-type)* is hashed using FNV-1a to produce a u64 identifier. The set of bigram hashes captures structural composition patterns.
- **Maximum tree depth.** The maximum depth of any node in the AST.
- **Node count.** The total number of nodes in the AST.
- **Cyclomatic complexity.** Computed as $1 + C$, where $C$ is the count of control-flow decision points (keywords and operators in the abstract representation).

## E. Feature Engineering

### E.1. Feature Set Definition

For each candidate pair $(f_i, f_j)$, 26 pairwise similarity features are computed. These features are organized into three groups:

**Lexical features (7).**
1. `lex_levenshtein_ratio`: Normalized Levenshtein similarity between token ID sequences, computed using the rapidfuzz library's `Levenshtein.normalized_similarity` function.
2. `lex_token_set_ratio`: Ratio of the intersection to the union of token sets (Jaccard similarity on token types).
3. `lex_sequence_match_ratio`: Ratio of the longest common subsequence (LCS) length to the mean sequence length, computed via two-row dynamic programming with a 512-token truncation limit.
4. `lex_jaccard_tokens`: Set-based Jaccard similarity on token type identifiers, using an 8,192-entry bitfield for efficient set operations.
5. `lex_dice_tokens`: Dice coefficient on token sets, $2|A \cap B| / (|A| + |B|)$.
6. `lex_lcs_ratio`: LCS-based similarity, $2 \cdot |LCS| / (|A| + |B|)$.
7. `lex_char_jaccard`: Character-level Jaccard similarity on the string representations of token sequences.

**Structural features (8).**
8. `struct_ast_hist_cosine`: Cosine similarity between 12-dimensional AST node-type histograms.
9. `struct_bigram_cosine`: Jaccard similarity on parent-child bigram hash sets.
10. `struct_depth_diff`: Absolute difference in maximum AST depth, $|d_i - d_j|$.
11. `struct_depth_ratio`: $\min(d_i, d_j) / \max(d_i, d_j)$.
12. `struct_node_count1`, `struct_node_count2`: Raw node counts for each file (non-similarity features providing absolute scale).
13. `struct_node_ratio`: $\min(n_i, n_j) / \max(n_i, n_j)$.
14. `struct_node_diff`: $|n_i - n_j|$.

**Quantitative features (11).**
15. `quant_token_ratio`: $\min(|t_i|, |t_j|) / \max(|t_i|, |t_j|)$, where $|t|$ is the token count.
16. `quant_line_delta`: Normalized absolute difference in line counts.
17. `quant_identifier_ratio`: Dice coefficient on identifier tokens (category 1).
18. `quant_cc1`, `quant_cc2`: Cyclomatic complexity values for each file.
19. `quant_cc_ratio`: $\min(cc_i, cc_j) / \max(cc_i, cc_j)$.
20. `quant_cc_diff`: $|cc_i - cc_j|$.
21. `quant_lines1`, `quant_lines2`: Line counts for each file.
22. `quant_chars1`, `quant_chars2`: Character counts for each file.
23. `quant_char_ratio`: $\min(|c_i|, |c_j|) / \max(|c_i|, |c_j|)$, where $|c|$ is the character count.

> **Table II** lists all 26 features with their group, computation method, and data type.

### E.2. Batch Computation Architecture

Feature computation is split between Zig and Python to balance performance and library availability:

**Zig computation (25 features).** The `compute_features_batch` function accepts flat arrays of concatenated token sequences, AST histograms, depths, node counts, cyclomatic complexities, and bigram hashes, along with index arrays specifying per-file offsets and counts. For each pair, it dispatches to `compute_pair_features`, which computes all Zig-side features. The batch function parallelizes across available CPU cores using a thread-per-chunk strategy: the pair list is divided into $\min(n_{\text{threads}}, 256)$ chunks, each processed by an OS thread. For pair counts below 1,000, parallelization is bypassed to avoid thread overhead.

**Python computation (1 feature).** The Levenshtein ratio (`lex_levenshtein_ratio`) is computed in Python using the rapidfuzz library, which provides a C-accelerated implementation of normalized Levenshtein similarity. Additionally, source-code-level features (line counts, character counts, line delta, character ratio) are computed by reading raw source files with an in-memory cache to avoid redundant I/O.

The FFI bridge (`src/bindings/features.py`) marshals NumPy arrays to contiguous C-typed pointers via `ctypes`, and reshapes the flat output array into a $(n_{\text{pairs}} \times 25)$ matrix.

## F. Locality-Sensitive Hashing (LSH) Candidate Filtering

To reduce the computational burden of exhaustive pairwise comparison, an LSH-based candidate filtering stage is employed prior to feature extraction and ML classification.

### F.1. K-Gram Shingling

Each file's abstract token sequence is converted into a set of $k$-grams (shingles) with $k=3$. Each shingle is hashed to a 64-bit unsigned integer using the FNV-1a hash function~\cite{fnv}:

$$h_{\text{FNV}}(s) = \bigoplus_{i=0}^{k-1} \bigl((h \oplus b_i) \times 1099511628211\bigr) \mod 2^{64}$$

where $h$ is initialized to the FNV offset basis (14,695,981,039,346,656,037) and $b_i$ are the bytes of each token identifier. Shingling is implemented in Zig with batch support for multiple files, producing a flat array of concatenated shingle hashes with per-file offset and count arrays.

### F.2. MinHash Signatures

For each file, a MinHash signature of length $N_h$ (default 128) is computed using universal hashing. Each hash function $h_i$ is defined as:

$$h_i(x) = (a_i \cdot x + b_i) \mod p$$

where $p = 2^{64} - 59$ (a large prime), and $a_i$ and $b_i$ are pseudo-random coefficients derived from a seed via a mixing function. The MinHash signature for a set $S$ is the vector of minimum hash values:

$$\text{sig}_i = \min_{x \in S} h_i(x)$$

The estimated Jaccard similarity between two signatures is the fraction of matching positions: $\hat{J} = |\{i : \text{sig}_A[i] = \text{sig}_B[i]\}| / N_h$.

### F.3. Band-Based LSH Bucketing

The signature is divided into $b$ bands of $r$ rows each, where $N_h = b \times r$ (default: $b = 16$, $r = 8$). Each band is hashed to a bucket identifier using FNV-1a. Two files are declared a candidate pair if they share at least one bucket in any band. This scheme provides a tunable approximation of the Jaccard similarity threshold $t \approx (1/b)^{1/r}$~\cite{leskovec2020mining}.

With default parameters ($b=16$, $r=8$), the theoretical threshold is $t \approx 0.668$, meaning pairs with Jaccard similarity below approximately 0.67 are unlikely to be candidates. The LSH stage achieves approximately 50\% pair reduction on the TOMA dataset with less than 2\% degradation in F1 score.

### F.4. Caching

LSH computation results are cached to disk using a SHA-256 fingerprint derived from the token data content and LSH parameters. Cache files are stored in `artifacts/cache/lsh/` and invalidated automatically when input data or parameters change.

## G. Hybrid Detection Pipeline

The full detection pipeline operates in four sequential stages:

### G.1. Stage 1: Type-1 Detection (Rule-Based)

Type-1 clones are detected by comparing MD5 hash values of normalized source files. Two functions $f_i$ and $f_j$ are classified as Type-1 clones if and only if:

$$\text{MD5}(\text{normalize}(f_i)) = \text{MD5}(\text{normalize}(f_j))$$

This stage achieves 99.98\% recall on the TOMA dataset's Type-1 subset, with the small number of missed detections attributable to files that failed normalization.

### G.2. Stage 2: Type-2 Detection (Rule-Based)

Pairs not classified as Type-1 are tested for Type-2 equivalence by comparing their abstract token sequences element-wise. Two functions are classified as Type-2 clones if their token sequences (produced by the abstraction process in Section~D.2) are identical:

$$\text{tokens}(f_i) = \text{tokens}(f_j)$$

This detects clones that differ only in identifier names, literal values, or whitespace---structural twins after abstraction. The stage achieves 99.65\% recall on the Type-2 subset.

### G.3. Stage 3: LSH Candidate Filtering

Pairs not resolved by Stages~1 or~2 are subjected to LSH candidate filtering (Section~F). Only candidate pairs identified by LSH proceed to feature extraction and ML classification. This reduces the number of pairs requiring expensive feature computation by approximately 50\%.

### G.4. Stage 4: ML Classification

Candidate pairs are classified using a trained XGBoost binary classifier (Section~H) with the 26-dimensional feature vector. The classifier outputs a probability $\hat{p} \in [0,1]$ of being a Type-3 clone; pairs with $\hat{p} \geq 0.5$ are classified as clones.

The pipeline's final output merges all stage predictions: Type-1 pairs from Stage~1, Type-2 pairs from Stage~2, and Type-3/non-clone labels from Stage~4. Pairs filtered out by LSH are implicitly classified as non-clones.

## H. Model Training and Selection

### H.1. Candidate Models

Five candidate classifiers are evaluated, each configured with the following hyperparameters:

| Model | Key Hyperparameters |
|:------|:-------------------|
| XGBoost~\cite{chen2016xgboost} | `n_estimators=200`, `max_depth=6`, `learning_rate=0.1`, `eval_metric=logloss`, `n_jobs=-1` |
| Random Forest~\cite{breiman2001random} | `n_estimators=200`, `max_depth=10`, `n_jobs=-1` |
| Logistic Regression | `solver=lbfgs`, `C=1.0`, `max_iter=5000` |
| Linear SVM | `C=1.0`, `max_iter=5000` |
| K-Nearest Neighbors | `n_neighbors=7`, `n_jobs=-1` |

> **Table III** lists the complete hyperparameter configurations for all five models.

### H.2. Cross-Validation Protocol

Each model is evaluated using 5-fold stratified cross-validation on the balanced training set, with random seed 42 ensuring reproducible fold assignments via `sklearn.model_selection.StratifiedKFold`. For each fold, the model is trained on 80\% of the data and evaluated on the remaining 20\%, with class proportions preserved across folds. Per-fold metrics include accuracy, precision, recall, F1-score, and area under the ROC curve (AUC-ROC). The best model is selected by the highest mean F1-score across folds.

### H.3. Final Model Training

The best-performing model (XGBoost in our experiments, with CV F1 = $0.9727 \pm 0.0006$) is retrained on the full balanced training set and serialized via `joblib` for deployment.

## I. Feature Selection via Recursive Feature Elimination

To identify the minimal effective feature subset, Recursive Feature Elimination (RFE)~\cite{guyon2002gene} is applied using the XGBoost classifier as the base estimator. RFE iteratively removes the least important feature (determined by `feature_importances_`) and re-evaluates classification performance via 5-fold cross-validation.

The feature count is swept over the set $\{3, 5, 8, 10, 13, 15, 18, 20\}$. The optimal number of features is determined by an elbow-point heuristic: the first point where adding more features yields less than 1\% F1 improvement. In our experiments, the optimal subset contains 3 features:

1. `lex_dice_tokens` (Dice coefficient on token sets)
2. `struct_node_count2` (node count of the second file)
3. `quant_cc2` (cyclomatic complexity of the second file)

A reduced XGBoost model trained on these 3 features achieves comparable performance to the full 26-feature model, demonstrating substantial feature redundancy in the original set.

## J. Probability Calibration

Post-hoc probability calibration is applied to improve the reliability of the classifier's probability outputs. Two calibration methods are compared:

- **Platt scaling (sigmoid).** Fits a logistic regression to the classifier's uncalibrated outputs, mapping them through a sigmoid function $\sigma(Az + B)$ where $z$ is the raw score and $A, B$ are fitted parameters~\cite{platt1999probabilistic}.
- **Isotonic regression.** Fits a non-parametric, monotonically increasing function to the classifier outputs, providing a more flexible calibration at the cost of potential overfitting on small datasets~\cite{zadrozny2002transforming}.

Calibration quality is assessed via the Brier score ($\text{BS} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{p}_i)^2$) and log loss. The method yielding the lowest Brier score on the training set (via internal cross-validation within `CalibratedClassifierCV`) is selected.

Additionally, optimal probability threshold analysis is performed by sweeping thresholds from 0.10 to 0.90 in increments of 0.01 and selecting the threshold maximizing F1-score on the test set.

## K. Evaluation Metrics

The following metrics are computed for all evaluations:

- **Accuracy**: $(TP + TN) / (TP + TN + FP + FN)$
- **Precision**: $TP / (TP + FP)$
- **Recall (Sensitivity)**: $TP / (TP + FN)$
- **F1-Score**: $2 \cdot \text{Precision} \cdot \text{Recall} / (\text{Precision} + \text{Recall})$
- **AUC-ROC**: Area under the Receiver Operating Characteristic curve
- **AUC-PR**: Area under the Precision-Recall curve
- **False Positive Rate (FPR)**: $FP / (FP + TN)$
- **False Negative Rate (FNR)**: $FN / (FN + TP)$
- **Confusion Matrix**: Full $2 \times 2$ matrix for binary evaluation; per-class matrices for multi-class evaluation

For the multi-class pipeline evaluation, weighted precision, recall, and F1 are computed by weighting each class by its support in the test set. Per-clone-type recall is reported separately for Type-1, Type-2, and Type-3 to characterize detection performance across clone categories.

## L. Ablation Studies

Three categories of ablation studies are conducted to quantify the contribution of each system component:

### L.1. Pipeline Component Ablation

Four configurations are compared: (i) the full pipeline, (ii) without normalization (raw source input), (iii) without CST-based token abstraction (whitespace-delimited tokens only), and (iv) without ML classification (rule-based detection only). Each configuration is evaluated on the full test set with identical metrics.

### L.2. Feature Group Ablation

Three experiments remove one feature group at a time: (i) without lexical features (7 removed), (ii) without structural features (8 removed), and (iii) without quantitative features (11 removed). Each configuration is evaluated using 5-fold cross-validation on the training set.

### L.3. Runtime Ablation

The impact of LSH filtering on end-to-end runtime is quantified by comparing pipeline execution time and pair throughput with and without LSH, measuring the reduction ratio (fraction of pairs filtered) and speedup factor.

## M. Sensitivity and Stability Analysis

### M.1. Sensitivity Analysis

Model robustness to input perturbation is assessed via three perturbation strategies applied to the test feature matrix:

- **Gaussian noise injection**: Additive noise $\epsilon \sim \mathcal{N}(0, \sigma)$ with $\sigma \in \{0.01, 0.05, 0.1, 0.2\}$.
- **Feature masking**: Random zeroing of a fraction of features, with mask rates $\in \{0.1, 0.2, 0.3, 0.5\}$.
- **Feature shifting**: Additive constant shift $\delta \in \{0.1, 0.5, 1.0, 2.0\}$ applied uniformly.

For each perturbation type and intensity, the model's F1-score is recorded and compared to the unperturbed baseline.

### M.2. Cross-Validation Stability

The stability of cross-validation results is quantified using the coefficient of variation (CV) of per-fold metrics:

$$\text{CV} = \frac{\sigma_{\text{fold}}}{\mu_{\text{fold}}}$$

Additionally, fold-to-fold prediction correlation is assessed via Pearson correlation of predicted probabilities between fold pairs, and feature importance stability is measured by computing the CV of importance ranks across folds.

## N. Implementation Details

### N.1. Languages and Frameworks

- **Python** ($\geq$3.10, $<$3.15): Pipeline orchestration, machine learning, data handling, evaluation.
- **Zig** (0.11+, compiled with `-Doptimize=ReleaseFast`): Performance-critical normalization, tokenization, feature computation, shingling, and MinHash-LSH.
- **Bash**: Pipeline automation scripts.

### N.2. Key Libraries

| Library | Version | Purpose |
|:--------|:--------|:--------|
| scikit-learn | $\geq$1.3 | Model selection, RFE, calibration, metrics |
| XGBoost | $\geq$2.0 | Gradient boosting classifier |
| pandas | $\geq$2.0 | Data manipulation and CSV I/O |
| NumPy | $\geq$1.24 | Numerical array operations |
| tree-sitter | $\geq$0.23 | Java CST parsing |
| tree-sitter-java | $\geq$0.23 | Java grammar for tree-sitter |
| rapidfuzz | $\geq$3.0 | C-accelerated Levenshtein similarity |
| joblib | $\geq$1.3 | Model serialization |
| matplotlib | $\geq$3.7 | Visualization |
| Poetry | -- | Dependency management and virtual environment |

### N.3. Foreign Function Interface (FFI)

Six Zig shared libraries are compiled to `zig-out/lib/`:

| Library | Source | Functions |
|:--------|:-------|:----------|
| `libnormalization.so` | `normalization.zig` | `remove_comments`, `normalize_whitespace`, `normalize_source` |
| `libtokenizer.so` | `tokenizer.zig` | `abstract_tokens`, `encode_tokens`, `count_token_types` |
| `libast_stats.so` | `ast_stats.zig` | `compute_ast_stats`, `compute_bigrams`, `compute_cyclomatic_complexity` |
| `libfeatures.so` | `features.zig` | `compute_pair_features`, `compute_features_batch` |
| `libshingling.so` | `shingling.zig` | `compute_shingles`, `compute_shingles_batch` |
| `libminhash.so` | `minhash.zig` | `minhash_signature`, `minhash_similarity`, `lsh_buckets`, batch variants |

Python loads each library via `ctypes.CDLL` and invokes exported functions with NumPy array pointers obtained via `ndarray.ctypes.data_as(ctypes.POINTER(...))`. All arrays are ensured to be contiguous (`np.ascontiguousarray`) prior to FFI calls to guarantee correct memory layout.

### N.4. Parallelization

The Zig `compute_features_batch` function implements thread-level parallelism by dividing the pair list into chunks and spawning one OS thread per chunk (up to 256 threads). The number of threads defaults to the CPU core count (obtained via `std.Thread.getCpuCount()`). Thread creation failures fall back to sequential computation. Python-side parallelization uses `n_jobs=-1` in scikit-learn and XGBoost for multi-core tree construction and prediction.

### N.5. Caching

LSH intermediate results (shingle hashes, MinHash signatures, bucket assignments) are cached to `artifacts/cache/lsh/` using a content-addressed key derived from a SHA-256 hash of the token data and LSH parameters. Cache invalidation occurs automatically when any input changes.

### N.6. Build System

The project uses a dual build system: (i) `build.zig` compiles all six Zig shared libraries and their unit tests, and (ii) a `Makefile` with 17 targets orchestrates the full pipeline, individual stages, analysis modules, testing, and cleanup. The `make setup` target performs end-to-end environment initialization.

## O. Reproducibility

Reproducibility is ensured through the following mechanisms:

1. **Fixed random seeds.** All stochastic operations (train/test splitting, cross-validation fold assignment, model initialization) use a fixed seed of 42.
2. **Dependency pinning.** The `poetry.lock` file pins exact versions of all Python dependencies.
3. **Automated pipeline.** Shell scripts in `scripts/` and Makefile targets provide one-command execution of every pipeline stage.
4. **Deterministic normalization.** The Zig normalization functions produce byte-identical output for identical input, with no locale-dependent behavior.
5. **Artifact serialization.** Trained models, evaluation metrics, and intermediate results are serialized to `artifacts/` for direct inspection and reuse.
6. **Unit tests.** Python tests (`tests/`) and Zig tests (invoked via `zig build test`) verify correctness of individual components.
