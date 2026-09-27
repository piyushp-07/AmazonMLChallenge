# Reproducing the business entity resolution submission

This directory contains the inference code, the supplied trained matcher, its configuration, and a local copy of the blocking and normalization source. It does not contain challenge data. Obtain the original challenge dataset separately and do not put it in this package.

## Environment

Python 3.12 is recommended. Install the pinned dependencies:

```powershell
python -m pip install -r requirements.txt
```

Set `AMON_DATA_ROOT` to the directory whose `dataset` child contains `train/` and `test/`. For example:

```powershell
$env:AMON_DATA_ROOT = 'C:\path\to\student_resource'
```

Required data files are `dataset/train/train_source1.tsv`, `train_source2.tsv`, `train_source3.tsv`, `train_ground_truth.tsv`, and `dataset/test/test_source1.tsv`, `test_source2.tsv`, `test_source3.tsv`. The pipeline uses the training set only for reproducibility context; inference uses the three test TSVs and the supplied model.

## Generate the full test candidate set and predictions

Run these commands from this directory (`final_submission/code/business_entity_resolution`):

```powershell
python src\generate_candidates.py --mode test --output ..\..\output\candidate_pairs.tsv
python -m src.inference --candidate ..\..\output\candidate_pairs.tsv --test-dir "$env:AMON_DATA_ROOT\dataset\test" --model models\matcher.pkl --config models\config.json --output ..\..\output\matching_results.tsv
```

Candidate generation applies the repository's country-aware exact normalized name, name prefix, useful name token, address token, and address prefix blocks separately to Source 2 and Source 3, then combines and deduplicates the resulting IDs. To keep the full multi-million-row test corpus tractable, exact normalized-name blocks are retained at any size and other posting lists are retained only when no more than 20 records share that country-aware key. The threshold was selected on the available training sample: the 20-key cap retained all 37 true links present among the sampled 5,000 Source 1 and 10,000-per-source comparison records, at 255,366 candidates. This small-sample recall is not a guarantee of full-set recall.

Inference computes the same 21 features described in `src/feature_engineering.py`, scores candidates in chunks with the supplied `HistGradientBoostingClassifier`, and applies the threshold in `models/config.json` (0.87 in the supplied configuration). Repeated deterministic string-pair calculations are cached. RapidFuzz supplies the exact Levenshtein distance used by the original Python implementation, with the same normalization formula. No training or external business lookup is performed.

The candidate TSV is the exact set scored by inference. The test source TSVs are read directly from `AMON_DATA_ROOT`; they are intentionally excluded from this package.

On the supplied test split, the bounded blocker generated 38,527,443 unique candidate IDs across 1,732,544 Source 1 rows (18,813,239 Source 2 candidates and 19,714,204 Source 3 candidates).

## Validate

From the repository root, run:

```powershell
python student_resource\utils\validate_submission.py --matching final_submission\output\matching_results.tsv --candidate final_submission\output\candidate_pairs.tsv --test-dir "$env:AMON_DATA_ROOT\dataset\test"
```

The final archive should include the two generated TSVs, this code directory, and the filled `Documentation_template.md`, with no dataset files.
