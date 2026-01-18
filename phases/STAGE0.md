# Stage 0: Binary Classification (Baseline vs Pain)

**Status:** PENDING
**Methodology:** Baseline-only (rest segments excluded)

---

## Objective

Binary classification using entropy-complexity plane features to distinguish baseline from pain states.

---

## Classification Task

| Class | Definition | Composition |
|-------|------------|-------------|
| 0 | Baseline | Pre-stimulus physiological baseline ONLY |
| 1 | Pain | Low pain + High pain combined |

**Note:** Rest segments are EXCLUDED from the dataset entirely.

---

## Data Configuration

```python
# Label mapping
BINARY_CLASS_MAPPING = {
    'baseline': 0,  # Baseline only
    'low': 1,       # Pain
    'high': 1       # Pain
    # 'rest': EXCLUDED
}

# Filter data
df = df[df['state'] != 'rest'].copy()
df['binary_label'] = df['state'].map(lambda x: 0 if x == 'baseline' else 1)
```

---

## Features

24 entropy-complexity features (8 per signal):
- EDA: pe, comp, fisher_shannon, fisher_info, renyipe, renyicomp, tsallispe, tsalliscomp
- BVP: pe, comp, fisher_shannon, fisher_info, renyipe, renyicomp, tsallispe, tsalliscomp
- RESP: pe, comp, fisher_shannon, fisher_info, renyipe, renyicomp, tsallispe, tsalliscomp

---

## Normalization

Global z-score (StandardScaler) - proven to work in LOSO validation.

**Do NOT use per-subject normalization** - causes data leakage in LOSO.

---

## Validation Methods

1. **80/20 Split** - Quick baseline (may have subject leakage)
2. **LOSO** - Gold standard subject-independent validation

---

## Expected Outcome

Binary classification (baseline vs pain) should achieve higher accuracy than 3-class classification since:
1. Only 2 classes instead of 3
2. Pain detection shown to be reliable (90%+ in Phase 5 hierarchical Stage 1)

---

## Output Files

- `results/stage0_binary/binary_results.csv`
- `results/stage0_binary/confusion_matrix.png`
- `results/stage0_binary/stage0_report.md`

---

## Execution

```bash
python src/stage0_binary.py
```

---

*Stage 0 to be implemented. Binary classification baseline vs pain.*
