# Phase 0: 3-Class Silhouette Analysis

## Objective
Identify discriminative (d, tau, signal) combinations for 3-class pain classification (baseline, low pain, high pain). Use silhouette coefficient to quantify cluster separation and guide feature selection for subsequent ML experiments.

---

## Input Data

### Raw Signals (Reference Only)
- **`data/train/`**, **`data/validation/`**, **`data/test/`**: Raw physiological signals
- Organized by signal type: `Bvp/`, `Eda/`, `Resp/`, `SpO2/`

### Extracted Features (Primary Source)
- **Location:** `data/features/results_{split}_{signal}.csv`
- **Signals:** EDA, BVP, RESP, SpO2 (4 signals)
- **Features per signal** (8 measures):
  - `pe` - Permutation Entropy (H)
  - `comp` - Statistical Complexity (C)
  - `fisher_shannon` - Fisher-Shannon Information
  - `fisher_info` - Fisher Information
  - `renyipe` - Renyi Permutation Entropy
  - `renyicomp` - Renyi Complexity
  - `tsallispe` - Tsallis Permutation Entropy
  - `tsalliscomp` - Tsallis Complexity
- **Labels** (embedded in feature files):
  - `state` - 3-class label (baseline=0, low_pain=1, high_pain=2)
  - `binaryclass` - Binary label (0 = no pain, 1 = pain)
- **Metadata:** `dimension`, `tau` for each feature row

---

## Experiment Configuration

### Parameter Space
- **Signals:** EDA, BVP, RESP, SpO2 (4 signals)
- **Embedding dimensions (d):** 3, 4, 5, 6, 7 (5 values)
- **Time delays (tau):** 1, 2, 3 (3 values)
- **Total combinations:** 60 (15 d/tau pairs × 4 signals)

### Silhouette Analysis
For each (d, tau, signal) combination:
1. Extract all 8 entropy/complexity measures for that combination
2. Create feature matrix (n_samples × 8 features)
3. Compute silhouette coefficient for 3-class labels
4. Use Euclidean distance metric
5. Store: signal, dimension, tau, silhouette_score

---

## Execution Steps

### Step 1: Data Loading
Load combined train, validation, and test sets for all signals. Extract 3-class labels (baseline, low pain, high pain).

### Step 2: Silhouette Computation
For each of 60 (d, tau, signal) combinations:
- Extract all 8 measures: PE, C, Fisher-Shannon, Fisher Info, Renyi PE, Renyi C, Tsallis PE, Tsallis C
- Create feature matrix of shape (n_samples, 8)
- Standardize features using StandardScaler (zero mean, unit variance)
- Compute silhouette score: `silhouette_score(features, labels, metric='euclidean')`
- Store: signal, dimension, tau, silhouette_score

### Step 3: Ranking and Analysis
- Sort all 60 combinations by silhouette score (descending)
- Identify top 20 combinations
- Compute distribution statistics (mean, std, quartiles)
- Compute mean silhouette score per signal
- Compute mean silhouette score per dimension
- Compute mean silhouette score per tau

### Step 4: Feature Selection Threshold Analysis
Generate threshold analysis:
- Top 10% (6 combos): silhouette range and feature count
- Top 20% (12 combos): silhouette range and feature count
- Top 30% (18 combos): silhouette range and feature count
- Top 40% (24 combos): silhouette range and feature count
- Top 50% (30 combos): silhouette range and feature count

For each threshold, calculate:
- Number of features: N_combos × 8 measures
- Signal distribution (how many EDA, BVP, RESP, SpO2 combos)
- Dimension distribution
- Tau distribution

### Step 5: Visualization
Generate visualizations:
- Histogram of all 60 silhouette scores
- Boxplot: silhouette scores by signal
- Boxplot: silhouette scores by dimension
- Boxplot: silhouette scores by tau
- Scatter plot: dimension vs. silhouette (colored by signal)
- Heatmap: (dimension, tau) grid showing mean silhouette across signals

### Step 6: SpO2 Exclusion Analysis
Compare SpO2 performance to other signals:
- Mean silhouette: SpO2 vs. EDA/BVP/RESP
- Number of SpO2 combos in top 50%
- Statistical test (t-test): SpO2 vs. other signals
- Generate exclusion justification statement

### Step 7: Report Generation
Create `3class_silhouette_report.md` containing:
- Executive summary with key findings
- Complete ranked table (all 60 combinations)
- Top 20 combinations highlighted
- Threshold analysis table (10%, 20%, 30%, 40%, 50%)
- Signal-wise performance comparison
- Dimension and tau analysis
- SpO2 exclusion rationale with statistical support
- Recommended feature selection threshold
- Visualization references

---

## Output Requirements

### Files to Generate
1. **`results/phase0_silhouette/3class_silhouette_scores.csv`**
   - Columns: rank, signal, dimension, tau, silhouette_score
   - 60 rows (one per combination)
   - Sorted by silhouette_score descending

2. **`results/phase0_silhouette/threshold_analysis.csv`**
   - Columns: threshold_percent, n_combos, n_features, silhouette_min, silhouette_max, silhouette_mean
   - Rows for 10%, 20%, 30%, 40%, 50% thresholds

3. **`results/phase0_silhouette/signal_statistics.csv`**
   - Columns: signal, mean_silhouette, std_silhouette, n_in_top50_percent
   - 4 rows (one per signal)

4. **`results/phase0_silhouette/3class_silhouette_report.md`**
   - Executive summary
   - Complete ranked table
   - Threshold analysis with recommendations
   - Signal/dimension/tau performance analysis
   - SpO2 exclusion justification
   - Feature selection recommendation for Phase 1
   - Methodology description
   - Statistical test results
   - Visualization references

5. **`results/phase0_silhouette/plots/`**
   - `silhouette_histogram.png`
   - `silhouette_by_signal_boxplot.png`
   - `silhouette_by_dimension_boxplot.png`
   - `silhouette_by_tau_boxplot.png`
   - `dimension_vs_silhouette_scatter.png`
   - `dimension_tau_heatmap.png`

---

## Checkpoint Instructions

After Phase 0 completes, this is a CRITICAL CHECKPOINT. Human review required to:
1. Review threshold analysis table
2. Decide on feature selection threshold (e.g., "use top 40%")
3. Confirm SpO2 exclusion based on statistical evidence
4. Approve recommended feature subset for Phase 1

**Human must update STATUS.md with:**
- Selected threshold percentage
- Number of features to use in Phase 1
- SpO2 inclusion decision (exclude recommended)
- Phase 0 status changed to "APPROVED"

---

## Completion Criteria

- [ ] All 60 silhouette scores computed
- [ ] All CSV files generated (scores, thresholds, statistics)
- [ ] Report markdown file generated with recommendations
- [ ] All 6 plots created
- [ ] Statistical analysis of SpO2 vs. other signals complete
- [ ] STATUS.md updated to "PHASE0_COMPLETE, AWAITING_APPROVAL"
- [ ] PLAN.md checklist updated (Phase 0 marked complete)

---

## Expected Runtime
Approximately 10-15 minutes on M2 Pro MacBook.

---

## Success Indicators
- Silhouette scores should be lower than binary (3-class is harder)
- Expected range: 0.15 - 0.50 for top combinations
- SpO2 mean silhouette should be significantly lower (p less than 0.05)
- Clear threshold recommendation emerges from analysis
- Top combinations should show preference for specific (d, tau) ranges