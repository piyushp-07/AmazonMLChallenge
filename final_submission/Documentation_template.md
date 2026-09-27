# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Not provided  
**Team Members:** Not provided  
**Submission Date:** 2026-09-27

---

## 1. Executive Summary

The solution uses the repository's country-aware name and address blocker followed by the supplied `HistGradientBoostingClassifier`. To control candidate explosion on the multi-million-row test split, the packaged blocker retains exact normalized-name postings at any frequency and caps every other country-aware posting at 20 records; the supplied 0.87 threshold is used for model predictions.

---

## 2. Methodology

### 2.1 Problem Analysis

The challenge instructions describe noisy business names and addresses, including abbreviation, punctuation, legal suffix, typo, transliteration, ordering, and missing-component variation. Country labels are treated as open-set strings so test-only countries are not filtered. The supplied train and test TSVs were extracted and used for the run. A limited blocker recall check was measured on the first 5,000 Source 1 and 10,000-per-source comparison records; no additional full-corpus EDA or error analysis was performed.

### 2.2 Solution Strategy

**Approach Type:** Blocking + supervised pair classifier  
**Core Innovation:** The repository combines multiple inexpensive, country-aware name and address blocking signals, followed by a compact engineered-feature classifier.

## 3. Candidate Generation (Blocking)

- **Blocking keys used:** Exact normalized name, four- and six-character normalized-name prefixes, useful name tokens, address tokens, and four- and six-character address prefixes, each combined with normalized country. Exact-name postings are unrestricted; all other postings with more than 20 records are skipped.
- **Candidate pairs generated:** 38,527,443 total (18,813,239 from Source 2 and 19,714,204 from Source 3), across 1,732,544 Source 1 test entities.
- **Training sample check:** On the first 5,000 Source 1 records and first 10,000 Source 2/Source 3 records, the 20-record posting cap produced 255,366 candidates and recovered all 37 ground-truth links represented in those comparison-source rows (100% on this limited sample).
- **How true matches are protected:** The blocker unions all retained rules and deduplicates IDs. The limited training sample supports the chosen cap, but does not establish full-set recall.

## 4. Matching Model

**Features used:** Name and address exact equality, Jaro-Winkler, normalized Levenshtein, token Jaccard, token overlap, TF-IDF cosine, length difference and ratio; country equality and missingness; plus name/address interaction features.

**Model type:** `HistGradientBoostingClassifier`, loaded from the repository's supplied `matcher.pkl` artifact.  
**Threshold selection method:** The supplied `config.json` sets threshold 0.87 and records validation macro precision 0.9955333, macro recall 0.9895917, and macro F0.5 0.9929686. These are pre-existing metadata values; they were not independently reproduced during this run.

## 5. Results & Error Analysis

- **F0.5 Score (macro):** The supplied configuration records validation F0.5 = 0.9929686; not independently reproduced. No test-set score is available because test ground truth is not provided.
- **Common false positives (wrong merges):** Not measured.
- **Common false negatives (missed matches):** Not measured.
- **Full test candidate count:** 38,527,443 candidate IDs across 1,732,544 Source 1 rows. The candidate TSV has 80,346 empty candidate lists.
- **Predictions:** 1,586,500 Source 1 entities have at least one prediction; 146,044 have an empty match list. The matcher predicted 10,299,303 IDs total. The supplied validator passed and confirmed every prediction was included in its candidate list. Its optional ID-existence check was left off as instructed; IDs were generated from the provided Source 2/Source 3 files.

## 6. Conclusion

The package preserves the repository's feature engineering, supplied trained matcher, and threshold configuration. The blocker uses the original keys with a frequency cap to bound work on the full corpus; inference is streamed in chunks to keep memory bounded.

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` contains the matcher source, local blocker and normalization source, supplied model and configuration, pinned Python dependencies, and exact reproduction commands in its README. Set `AMON_DATA_ROOT` to a local copy of the challenge `student_resource` directory before running the candidate generator and inference module.

### B. Additional Results

The extracted challenge dataset was used locally and is excluded from the submission package. The completed inference scored 38,527,443 candidate pairs using the supplied matcher and 0.87 threshold.



