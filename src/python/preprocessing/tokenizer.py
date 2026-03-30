"""Tokenization and CST generation using tree-sitter + Zig abstraction."""

import argparse
import logging
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tree_sitter_java as tsj
from tree_sitter import Language, Node, Parser

from src.bindings.ast_stats import AstStatsLib
from src.python.utils.io import load_source_code, save_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tree-sitter setup
# ---------------------------------------------------------------------------

JAVA_LANGUAGE = Language(tsj.language())
PARSER = Parser(JAVA_LANGUAGE)

# Abstract token categories
CATEGORY_KEYWORD = 0
CATEGORY_IDENTIFIER = 1
CATEGORY_OPERATOR = 2
CATEGORY_LITERAL = 3
CATEGORY_MODIFIER = 4
CATEGORY_TYPE = 5
CATEGORY_SEPARATOR = 6
CATEGORY_DECLARATION = 7
CATEGORY_EXPRESSION = 8
CATEGORY_STATEMENT = 9
CATEGORY_ANNOTATION = 10
CATEGORY_OTHER = 255

# Tree-sitter node type → abstract category mapping
NODE_TYPE_MAP: dict[str, int] = {}

# Keywords
for kw in [
    "if", "else", "for", "while", "do", "switch", "case", "default",
    "break", "continue", "return", "throw", "throws", "try", "catch",
    "finally", "new", "this", "super", "instanceof", "class", "interface",
    "enum", "extends", "implements", "package", "import", "assert",
    "yield", "record", "sealed", "permits", "var", "true", "false", "null",
]:
    NODE_TYPE_MAP[kw] = CATEGORY_KEYWORD

# Modifiers
for mod in [
    "public", "private", "protected", "static", "final", "abstract",
    "synchronized", "volatile", "transient", "native", "strictfp",
    "default", "open", "module",
]:
    NODE_TYPE_MAP[mod] = CATEGORY_MODIFIER

# Operators
for op in [
    "+", "-", "*", "/", "%", "=", "==", "!=", "<", ">", "<=", ">=",
    "&&", "||", "!", "&", "|", "^", "~", "<<", ">>", ">>>",
    "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>=", ">>>=",
    "++", "--", "?", "->", "::", ".", "..",
]:
    NODE_TYPE_MAP[op] = CATEGORY_OPERATOR

# Separators
for sep in ["(", ")", "{", "}", "[", "]", ";", ",", "@"]:
    NODE_TYPE_MAP[sep] = CATEGORY_SEPARATOR

# Tree-sitter structural node types
for node_type in [
    "identifier", "type_identifier",
]:
    NODE_TYPE_MAP[node_type] = CATEGORY_IDENTIFIER

for node_type in [
    "decimal_integer_literal", "hex_integer_literal", "octal_integer_literal",
    "binary_integer_literal", "decimal_floating_point_literal",
    "hex_floating_point_literal", "character_literal", "string_literal",
    "null_literal", "true", "false",
]:
    NODE_TYPE_MAP[node_type] = CATEGORY_LITERAL

for node_type in [
    "void_type", "integral_type", "floating_point_type", "boolean_type",
    "type_identifier", "generic_type", "array_type", "scoped_type_identifier",
]:
    NODE_TYPE_MAP[node_type] = CATEGORY_TYPE

for node_type in [
    "method_declaration", "constructor_declaration", "field_declaration",
    "variable_declarator", "formal_parameter", "local_variable_declaration",
    "enum_declaration", "class_declaration", "interface_declaration",
    "annotation_type_declaration", "record_declaration",
]:
    NODE_TYPE_MAP[node_type] = CATEGORY_DECLARATION

for node_type in [
    "if_statement", "for_statement", "while_statement", "do_statement",
    "switch_statement", "try_statement", "try_with_resources_statement",
    "return_statement", "throw_statement", "break_statement",
    "continue_statement", "assert_statement", "yield_statement",
    "expression_statement", "block", "synchronized_statement",
    "enhanced_for_statement", "labeled_statement",
]:
    NODE_TYPE_MAP[node_type] = CATEGORY_STATEMENT

for node_type in [
    "binary_expression", "unary_expression", "ternary_expression",
    "assignment_expression", "method_invocation", "array_access",
    "field_access", "object_creation_expression", "array_creation_expression",
    "cast_expression", "instanceof_expression", "lambda_expression",
    "method_reference", "parenthesized_expression",
    "update_expression", "switch_expression",
]:
    NODE_TYPE_MAP[node_type] = CATEGORY_EXPRESSION

for node_type in [
    "marker_annotation", "annotation", "type_annotation",
]:
    NODE_TYPE_MAP[node_type] = CATEGORY_ANNOTATION


def classify_node(node: Node) -> int:
    """Classify a tree-sitter node into an abstract token category."""
    ntype = node.type

    # Direct mapping
    if ntype in NODE_TYPE_MAP:
        return NODE_TYPE_MAP[ntype]

    # Leaf tokens (named children == 0)
    if node.child_count == 0:
        if ntype in ("(", ")", "{", "}", "[", "]", ";", ",", "@"):
            return CATEGORY_SEPARATOR
        # Check if it's an operator-like token
        if len(ntype) <= 4 and not ntype[0].isalpha():
            return CATEGORY_OPERATOR
        return CATEGORY_OTHER

    return CATEGORY_OTHER


# ---------------------------------------------------------------------------
# Token extraction
# ---------------------------------------------------------------------------

def extract_tokens(source: str) -> tuple[list[int], list[dict]]:
    """Extract abstract token types and AST stats from Java source.

    Returns:
        token_ids: list of abstract token type integers (for shingling)
        ast_info: list of dicts with node type, depth, parent index
    """
    tree = PARSER.parse(source.encode("utf-8"))
    root = tree.root_node

    token_ids = []
    ast_nodes = []
    ast_depths = []
    ast_parents = []

    def traverse(node: Node, depth: int, parent_idx: int):
        current_idx = len(ast_nodes)
        cat = classify_node(node)
        ast_nodes.append(cat)
        ast_depths.append(depth)
        ast_parents.append(parent_idx if parent_idx >= 0 else 2**64 - 1)

        # Only add leaf tokens to the token sequence
        if node.child_count == 0:
            token_ids.append(cat)

        for child in node.children:
            traverse(child, depth + 1, current_idx)

    traverse(root, 0, -1)
    return token_ids, {
        "node_types": ast_nodes,
        "depths": ast_depths,
        "parent_indices": ast_parents,
        "node_count": len(ast_nodes),
    }


def compute_ast_stats_zig(ast_info: dict, ast_lib: AstStatsLib) -> dict:
    """Compute AST statistics using Zig library."""
    node_types = np.array(ast_info["node_types"], dtype=np.uint8)
    depths = np.array(ast_info["depths"], dtype=np.uint16)
    parents = np.array(ast_info["parent_indices"], dtype=np.uintp)

    # Histogram
    histogram = np.zeros(12, dtype=np.uint32)
    max_depth = np.zeros(1, dtype=np.uint16)
    ast_lib.compute_ast_stats(node_types, depths, max_depth, histogram)

    # Bigrams
    bigrams = np.zeros(len(node_types), dtype=np.uint64)
    bigram_count = ast_lib.compute_bigrams(node_types, parents, bigrams)

    # Cyclomatic complexity
    cc = ast_lib.compute_cyclomatic_complexity(node_types)

    return {
        "max_depth": int(max_depth[0]),
        "node_count": int(ast_info["node_count"]),
        "histogram": histogram.tolist(),
        "bigrams": bigrams[:bigram_count].tolist(),
        "cyclomatic_complexity": int(cc),
    }


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def process_all(
    df: pd.DataFrame,
    source_dir: Path,
    normalized_dir: Path,
    ast_lib: AstStatsLib,
    output_dir: Path,
) -> dict:
    """Process all unique files: tokenize, abstract, compute AST stats."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ids = pd.unique(df[["id1", "id2"]].values.ravel())
    logger.info("Tokenizing %d unique source files...", len(ids))

    results = {}
    processed = 0
    errors = 0

    for fid in ids:
        # Try normalized first, fall back to raw
        src = None
        norm_path = normalized_dir / f"{fid}.java"
        if norm_path.exists():
            src = norm_path.read_text(encoding="utf-8")
        else:
            src = load_source_code(fid, source_dir)

        if src is None or not src.strip():
            errors += 1
            continue

        try:
            token_ids, ast_info = extract_tokens(src)
            ast_stats = compute_ast_stats_zig(ast_info, ast_lib)

            results[fid] = {
                "token_ids": token_ids,
                "token_count": len(token_ids),
                "ast_stats": ast_stats,
            }
        except Exception as e:
            logger.error("Failed to tokenize %s: %s", fid, e)
            errors += 1

        processed += 1
        if processed % 5000 == 0:
            logger.info("  Processed %d / %d", processed, len(ids))

    logger.info("Tokenization complete: %d processed, %d errors", processed, errors)

    # Save results as pickle (token IDs are variable-length lists)
    out_path = output_dir / "token_data.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    logger.info("Saved token data to %s", out_path)

    return results


# ---------------------------------------------------------------------------
# Metrics and plots
# ---------------------------------------------------------------------------

def compute_tokenization_metrics(results: dict) -> dict:
    if not results:
        return {}

    token_counts = [v["token_count"] for v in results.values()]
    depths = [v["ast_stats"]["max_depth"] for v in results.values()]
    node_counts = [v["ast_stats"]["node_count"] for v in results.values()]
    cc_values = [v["ast_stats"]["cyclomatic_complexity"] for v in results.values()]

    # Aggregate token type distribution
    type_totals = np.zeros(12, dtype=np.int64)
    for v in results.values():
        for tid in v["token_ids"]:
            if tid < 12:
                type_totals[tid] += 1

    return {
        "total_files": len(results),
        "avg_tokens": float(np.mean(token_counts)),
        "median_tokens": float(np.median(token_counts)),
        "max_tokens": int(np.max(token_counts)),
        "avg_ast_depth": float(np.mean(depths)),
        "median_ast_depth": float(np.median(depths)),
        "max_ast_depth": int(np.max(depths)),
        "avg_node_count": float(np.mean(node_counts)),
        "median_node_count": float(np.median(node_counts)),
        "avg_cyclomatic_complexity": float(np.mean(cc_values)),
        "token_type_distribution": {
            "KEYWORD": int(type_totals[0]),
            "IDENTIFIER": int(type_totals[1]),
            "OPERATOR": int(type_totals[2]),
            "LITERAL": int(type_totals[3]),
            "MODIFIER": int(type_totals[4]),
            "TYPE": int(type_totals[5]),
            "SEPARATOR": int(type_totals[6]),
            "DECLARATION": int(type_totals[7]),
            "EXPRESSION": int(type_totals[8]),
            "STATEMENT": int(type_totals[9]),
            "ANNOTATION": int(type_totals[10]),
            "OTHER": int(type_totals[11]),
        },
    }


def _safe_plot_hist(ax, data, **kwargs):
    """Plot histogram safely."""
    try:
        data_f = np.array(data, dtype=np.float64)
        data_f = data_f[np.isfinite(data_f)]
        if len(data_f) == 0 or data_f.max() == data_f.min():
            return
        ax.hist(data_f, **kwargs)
    except (ValueError, TypeError):
        pass


def generate_tokenization_plots(results: dict, plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)

    depths = [v["ast_stats"]["max_depth"] for v in results.values()]
    node_counts = [v["ast_stats"]["node_count"] for v in results.values()]

    # AST depth histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    _safe_plot_hist(ax, depths, bins=50, edgecolor="black", alpha=0.7)
    ax.set_xlabel("AST Depth")
    ax.set_ylabel("Frequency")
    ax.set_title("AST Depth Distribution")
    plt.tight_layout()
    fig.savefig(plots_dir / "ast_depth_histogram.png", dpi=150)
    plt.close(fig)

    # Node count histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    _safe_plot_hist(ax, node_counts, bins=50, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Node Count")
    ax.set_ylabel("Frequency")
    ax.set_title("AST Node Count Distribution")
    plt.tight_layout()
    fig.savefig(plots_dir / "node_count_histogram.png", dpi=150)
    plt.close(fig)

    # Token type distribution pie chart
    type_totals = np.zeros(12, dtype=np.int64)
    for v in results.values():
        for tid in v["token_ids"]:
            if tid < 12:
                type_totals[tid] += 1

    labels = [
        "KEYWORD", "IDENTIFIER", "OPERATOR", "LITERAL", "MODIFIER",
        "TYPE", "SEPARATOR", "DECLARATION", "EXPRESSION", "STATEMENT",
        "ANNOTATION", "OTHER",
    ]
    non_zero = [(lbl, int(t)) for lbl, t in zip(labels, type_totals) if t > 0]
    if non_zero:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie(
            [v for _, v in non_zero],
            labels=[lbl for lbl, _ in non_zero],
            autopct="%1.1f%%",
            startangle=140,
        )
        ax.set_title("Token Type Distribution After Abstraction")
        plt.tight_layout()
        fig.savefig(plots_dir / "token_type_distribution.png", dpi=150)
        plt.close(fig)

    # AST depth vs node count scatter
    fig, ax = plt.subplots(figsize=(8, 6))
    sample_size = min(len(depths), 5000)
    idx = np.random.RandomState(42).choice(len(depths), sample_size, replace=False)
    ax.scatter(
        [depths[i] for i in idx],
        [node_counts[i] for i in idx],
        alpha=0.3, s=5,
    )
    ax.set_xlabel("AST Depth")
    ax.set_ylabel("Node Count")
    ax.set_title("AST Depth vs Node Count")
    plt.tight_layout()
    fig.savefig(plots_dir / "depth_vs_nodes.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Tokenization & CST generation")
    parser.add_argument("--source-dir", type=Path, default=Path("data/raw/toma/id2sourcecode"))
    parser.add_argument("--normalized-dir", type=Path, default=Path("data/processed/normalized"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/intermediate"))
    parser.add_argument("--eval-dir", type=Path, default=Path("artifacts/evaluation"))
    args = parser.parse_args()

    ast_lib = AstStatsLib()

    # Load datasets
    train_path = args.data_dir / "training_dataset.csv"
    test_path = args.data_dir / "testing_dataset.csv"
    if not train_path.exists() or not test_path.exists():
        logger.error("Run data preparation first")
        sys.exit(1)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    combined = pd.concat([train_df, test_df], ignore_index=True)

    # Process
    results = process_all(
        combined, args.source_dir, args.normalized_dir, ast_lib, args.output_dir
    )

    # Metrics
    metrics = compute_tokenization_metrics(results)
    args.eval_dir.mkdir(parents=True, exist_ok=True)
    save_json(metrics, args.eval_dir / "tokenization_metrics.json")
    logger.info("Saved tokenization_metrics.json")

    # Plots
    plots_dir = args.eval_dir / "plots" / "tokenization"
    generate_tokenization_plots(results, plots_dir)
    logger.info("Plots saved to %s", plots_dir)

    print("\n=== Tokenization Summary ===")
    print(f"Files processed: {metrics.get('total_files', '?')}")
    print(f"Avg tokens per file: {metrics.get('avg_tokens', 0):.1f}")
    print(f"Avg AST depth: {metrics.get('avg_ast_depth', 0):.1f}")
    print(f"Avg node count: {metrics.get('avg_node_count', 0):.1f}")
    print(f"Avg cyclomatic complexity: {metrics.get('avg_cyclomatic_complexity', 0):.1f}")
    print("Done.")


if __name__ == "__main__":
    main()
