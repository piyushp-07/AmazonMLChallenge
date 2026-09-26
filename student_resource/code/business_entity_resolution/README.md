# AMON Business Entity Resolution

## Overview

This project implements a two-stage business entity resolution system for the AMON / Amazon ML Challenge.

The system matches noisy business records from Source 2 and Source 3 to the deduplicated reference entities in Source 1.

The pipeline consists of:

1. Data loading and normalization
2. Candidate generation / blocking
3. Feature engineering
4. Supervised pair matching
5. Threshold selection
6. Final prediction generation
7. Submission validation

---

## Project Structure

```text
business_entity_resolution/
│
├── generate_candidates.py
├── README.md
├── requirements.txt
├── run_demo.py
├── test_dry_run.py
│
└── src/
    ├── config.py
    ├── data_loader.py
    ├── feature_engineering.py
    ├── inference.py
    ├── normalization.py
    ├── pipeline.py
    ├── similarity.py
    ├── threshold_tuning.py
    ├── train.py
    ├── training.py
    ├── validation.py
    │
    ├── candidate_filter/
    └── evaluation/
```

---

## Dataset Layout

The expected dataset layout is:

```text
student_resource/
│
├── dataset/
│   ├── train/
│   │   ├── train_source1.tsv
│   │   ├── train_source2.tsv
│   │   ├── train_source3.tsv
│   │   └── train_ground_truth.tsv
│   │
│   └── test/
│       ├── test_source1.tsv
│       ├── test_source2.tsv
│       └── test_source3.tsv
│
├── models/
│   ├── matcher.pkl
│   └── config.json
│
└── output/
    ├── candidate_pairs.tsv
    └── matching_results.tsv
```

The dataset itself should not be committed to Git.

---

## Data Location

The pipeline supports the `AMON_DATA_ROOT` environment variable.

On Windows PowerShell:

```powershell
$env:AMON_DATA_ROOT = "C:\Users\shiva\Downloads\6ab10eb3b23ba_student_resource\student_resource"
```

Verify:

```powershell
Test-Path "$env:AMON_DATA_ROOT\dataset\train\train_source1.tsv"
```

The command should return:

```
True
```

---

## Candidate Generation

Candidate generation is handled by the Member 1 blocking module.

The blocker combines multiple inexpensive blocking signals, including:

- exact normalized business name
- name prefixes
- name tokens
- address tokens
- address prefixes
- country-aware matching

The purpose of blocking is to reduce the number of possible S1-to-S2/S3 comparisons before applying the more expensive matcher.

Run:

```powershell
python generate_candidates.py
```

This produces:

```
student_resource/output/candidate_pairs.tsv
```

The candidate file contains:

| Column | Description |
|---|---|
| `source1_entity_id` | S1 entity identifier |
| `candidate_entity_ids` | Comma-separated candidate IDs from S2/S3 |

Every Source 1 entity is represented, including entities for which no candidate was generated.

---

## Matcher Training

The trained matcher uses the generated candidate pairs together with ground-truth links.

The current training configuration uses:

- controlled positive retrieval from the full training ground truth
- sampled negative candidates
- entity-level train/validation split
- engineered string/address features
- `HistGradientBoostingClassifier`
- validation-based threshold tuning
- beta = 0.5 evaluation

The trained model is saved as:

```
student_resource/models/matcher.pkl
```

The selected threshold and training metadata are saved as:

```
student_resource/models/config.json
```

---

## Inference

After candidate generation and model training, inference can be run with:

```powershell
python -m src.inference
```

Inference:

1. Loads the test Source 1, Source 2 and Source 3 data.
2. Loads `candidate_pairs.tsv`.
3. Generates features for candidate pairs.
4. Loads the trained matcher.
5. Loads the tuned threshold.
6. Scores candidate pairs.
7. Keeps pairs whose probability is at or above the threshold.
8. Produces `matching_results.tsv`.

Output:

```
student_resource/output/matching_results.tsv
```

The output format is:

| Column | Description |
|---|---|
| `source1_entity_id` | S1 entity identifier |
| `matched_entity_ids` | Comma-separated matched IDs from S2/S3 |

---

## Full Pipeline

The repository also contains the pipeline module:

```powershell
python -m src.pipeline
```

Before running the full pipeline on the complete dataset, ensure that the dataset paths, model configuration, and available system memory are appropriate.

For the final challenge run, candidate generation and inference should be performed only after the code has passed the smaller validation and dry-run checks.

---

## Submission Validation

The challenge provides a submission validator.

Run:

```powershell
python utils\validate_submission.py --matching output\matching_results.tsv --candidate output\candidate_pairs.tsv --test-dir dataset\test
```

The validation should be performed after both:

- `candidate_pairs.tsv`
- `matching_results.tsv`

have been generated.

---

## Important Constraints

The implementation follows the challenge constraints:

- No external business-data lookup
- No geocoding API
- No external entity database
- No internet-based business enrichment
- Candidate pairs are generated locally
- Final matched entities must come from the candidate set
- The model is limited to the project implementation and permitted dependencies

---

## Development Workflow

Recommended order:

```
1. Validate code
       ↓
2. Run dry-run / small tests
       ↓
3. Verify candidate generation
       ↓
4. Verify matcher
       ↓
5. Generate final candidate_pairs.tsv
       ↓
6. Run final inference
       ↓
7. Validate submission
       ↓
8. Package final submission
```

The expensive full-data end-to-end test should be run only after the project code is finalized.

---

## Current Model

The trained matcher is:

```
HistGradientBoostingClassifier
```

The model uses engineered similarity features between Source 1 and candidate Source 2/Source 3 records.

The decision threshold is stored in:

```
student_resource/models/config.json
```

and is loaded automatically during inference.

---

## Output Files

### `candidate_pairs.tsv`

Contains the candidate entities that are passed to the matcher.

```
source1_entity_id    candidate_entity_ids
```

### `matching_results.tsv`

Contains the final predicted matches.

```
source1_entity_id    matched_entity_ids
```

For an entity with no predicted match:

```
S1-12345
```

For an entity with multiple matches:

```
S1-12345    S2-100,S3-200,S3-300
```
