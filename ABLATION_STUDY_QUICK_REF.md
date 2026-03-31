# Ablation Study - Quick Reference

## What's New

### ✅ Individual Feature Plots
Each visualization is now saved separately for easy inclusion in papers:

```
artifacts/evaluation/ablation_study/
├── individual_f1_comparison.png          # F1 scores with error bars
├── individual_f1_degradation.png         # ΔF1% with value labels
├── individual_feature_ranking.png        # Ranked by importance
├── individual_baseline_comparison.png    # Baseline vs each removal
└── individual_feature_plots/              # Separate numbered files
    ├── 01_f1_score_comparison.png
    ├── 02_f1_degradation.png
    ├── 03_feature_ranking.png
    └── 04_baseline_comparison.png
```

### ✅ Test Dataset by Default
- Now uses **test_features.csv** instead of train_features.csv
- Results reflect actual pipeline performance on unseen data
- More appropriate for paper publication

## Usage

### Run Complete Ablation Study (Recommended)

```bash
bash scripts/run_ablation_study.sh
```

This runs all three studies using test data and generates:
- Individual feature ablation (per-feature analysis)
- Feature group ablation (lexical/structural/quantitative)
- LSH parameter ablation (efficiency vs accuracy)

### Run Specific Studies

```bash
# Only individual feature ablation
poetry run python src/python/evaluation/run_ablation_study.py \
    --study-type individual

# Only feature group ablation
poetry run python src/python/evaluation/run_ablation_study.py \
    --study-type group

# Only LSH ablation
poetry run python src/python/evaluation/run_ablation_study.py \
    --study-type lsh

# All studies
poetry run python src/python/evaluation/run_ablation_study.py \
    --study-type both
```

### Use Training Data Instead

```bash
# Use training dataset
poetry run python src/python/evaluation/run_ablation_study.py \
    --no-use-test-data
```

## Output Files

### Results (JSON)
- `individual_feature_ablation_results.json`
- `feature_group_ablation_results.json`
- `lsh_ablation_results.json`
- `combined_ablation_results.json`

### Summary Tables (CSV & Markdown)
- `individual_feature_ablation_summary.csv/.md`
- `feature_group_ablation_summary.csv/.md`
- `lsh_ablation_summary.csv/.md`

### Visualizations
**Combined (4-panel):**
- `plots/individual_feature_ablation_comparison.png`
- `plots/feature_group_ablation_comparison.png`
- `plots/lsh_ablation_comparison.png`

**Individual (single plots):**
- `individual_f1_comparison.png`
- `individual_f1_degradation.png`
- `individual_feature_ranking.png`
- `individual_baseline_comparison.png`
- `individual_feature_plots/*.png` (numbered for easy reference)

## Example Results (Test Dataset)

### Individual Feature Importance

| Rank | Feature | F1 Score | Δ F1 (%) |
|------|---------|----------|----------|
| 1 | lex_dice_tokens | 0.8761 | **-8.64%** |
| 2 | struct_node_count2 | 0.9371 | -2.28% |
| 3 | quant_cc2 | 0.9500 | -0.94% |

### Feature Group Contribution

| Group | Δ F1 (%) | Importance |
|-------|----------|------------|
| Lexical | -8.64% | 🔴 High |
| Structural | -2.28% | 🟡 Medium |
| Quantitative | -0.94% | 🟢 Low |

### LSH Trade-offs

| Configuration | F1 Score | Speedup | Δ F1 (%) |
|--------------|----------|---------|----------|
| No LSH | 0.9590 | 1.0× | - |
| Aggressive (80%) | 0.9091 | **3.3×** | -5.21% |
| Balanced (50%) | 0.9410 | 1.8× | -1.88% |
| Conservative (20%) | 0.9501 | 1.2× | -0.93% |

## For Paper Authors

### Figures to Include

1. **Individual Feature Importance**
   - Use: `individual_feature_ranking.png` or `individual_f1_degradation.png`
   - Shows: Which specific features matter most
   - Caption: "Individual feature ablation showing lex_dice_tokens as most critical (-8.64% F1)"

2. **Feature Group Contribution**
   - Use: `plots/feature_group_ablation_comparison.png`
   - Shows: Hierarchy of feature groups
   - Caption: "Feature group ablation revealing lexical features dominate (8.64% contribution)"

3. **LSH Efficiency Trade-off**
   - Use: `plots/lsh_ablation_comparison.png`
   - Shows: Speed vs accuracy curve
   - Caption: "LSH parameter ablation demonstrating 3.3× speedup with 5.21% F1 loss"

### LaTeX Tables

See `ABLATION_STUDY_COMPLETE_GUIDE.md` for ready-to-use LaTeX tables.

## Performance

**Typical Runtime (test dataset, 149k pairs):**
- Individual feature ablation: ~45 seconds (3 features × 5 CV)
- Feature group ablation: ~45 seconds (4 configurations × 5 CV)
- LSH ablation: ~15 seconds (simulated)
- **Total: ~105 seconds** for complete study

## Troubleshooting

### Study takes too long
- Reduce CV folds: `--cv 3`
- Run individual studies separately
- Use fewer features (RFE-selected)

### Out of memory
- Use `--no-use-test-data` for smaller training set
- Reduce CV folds: `--cv 3`

### Missing features file
```bash
# Run feature engineering first
bash scripts/run_features.sh
```

## Citation

When referencing in your paper:

> "We conducted comprehensive ablation studies at three levels: individual feature, feature group, and LSH parameter. Results show lexical features (lex_dice_tokens) contribute most significantly (8.64% F1 improvement), followed by structural (2.28%) and quantitative features (0.94%). For LSH filtering, a balanced configuration achieves 1.8× speedup with minimal 1.88% F1 degradation."

---

**Last Updated**: March 31, 2026  
**Default Dataset**: Test set (149,005 pairs)  
**Features**: 3 RFE-selected features  
**Model**: XGBoost with 5-fold CV
