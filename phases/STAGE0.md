# Stage 0: Binary Classification Silhouette Analysis

## Objective
Replicate Paper 1's C-H plane separation for binary classification (pain vs. no-pain), quantified via silhouette coefficient. This provides a quick win for the PI and establishes baseline cluster separation metrics.

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
  - `binaryclass` - Binary label (0 = no pain, 1 = pain)
  - `state` - 3-class label (baseline, low_pain, high_pain)
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
1. Extract Permutation Entropy (H) and Statistical Complexity (C) values
2. Plot points on 2D C-H plane (Complexity on y-axis, Entropy on x-axis)
3. Color-code by binary class (pain vs. no-pain)
4. Compute silhouette coefficient using `sklearn.metrics.silhouette_score`
5. Use Euclidean distance metric

---

## Execution Steps

### Step 1: Data Loading
Load combined train, validation, and test sets for all signals. Extract binary classification labels (pain = 1, no-pain = 0).

### Step 2: Silhouette Computation
For each of 60 (d, tau, signal) combinations:
- Extract H and C values for that specific combination
- Create 2D array of shape (n_samples, 2) with [H, C] coordinates
- Compute silhouette score: `silhouette_score(coordinates, labels, metric='euclidean')`
- Store: signal, dimension, tau, silhouette_score

### Step 3: Ranking and Analysis
- Sort all 60 combinations by silhouette score (descending)
- Identify top 10 combinations
- Identify bottom 10 combinations
- Compute mean silhouette score per signal (to assess SpO2 vs. others)

### Step 4: Visualization
Generate C-H plane plots for:
- Top 5 combinations (highest silhouette scores)
- One plot per signal showing best d/tau combo for that signal
- Use different colors/markers for pain vs. no-pain classes
- Include silhouette score in plot title

### Step 5: Report Generation
Create `binary_silhouette_report.md` containing:
- Summary table of all 60 combinations ranked by silhouette score
- Statistical summary (mean, std, min, max silhouette by signal)
- Top 10 combinations with interpretation
- SpO2 performance analysis
- Visualization references

---

## Output Requirements

### Files to Generate
1. **`results/stage0_binary/binary_silhouette_scores.csv`**
   - Columns: rank, signal, dimension, tau, silhouette_score
   - 60 rows (one per combination)
   - Sorted by silhouette_score descending

2. **`results/stage0_binary/binary_silhouette_report.md`**
   - Executive summary
   - Top 10 combinations table
   - Signal-wise performance comparison
   - SpO2 analysis and exclusion justification
   - Methodology description
   - References to plots

3. **`results/stage0_binary/plots/`**
   - `top_combo_ch_plane.png` (best overall combination)
   - `eda_best_ch_plane.png` (best EDA combination)
   - `bvp_best_ch_plane.png` (best BVP combination)
   - `resp_best_ch_plane.png` (best RESP combination)
   - `spo2_best_ch_plane.png` (best SpO2 combination)

---

## Completion Criteria

- [ ] All 60 silhouette scores computed
- [ ] CSV file generated with ranked results
- [ ] Report markdown file generated
- [ ] Minimum 5 C-H plane plots created
- [ ] STATUS.md updated to "STAGE0_COMPLETE, AWAITING_APPROVAL"
- [ ] PLAN.md checklist updated (Stage 0 marked complete)

---

## Expected Runtime
Approximately 5-10 minutes on M2 Pro MacBook.

---

## Success Indicators
- Silhouette scores should be positive (greater than 0) for top combinations
- EDA, BVP, RESP should show similar silhouette ranges (0.3-0.7 typical)
- SpO2 should show consistently lower scores (less than 0.3) justifying exclusion
- Visual inspection of C-H planes should show distinct clusters for pain vs. no-pain