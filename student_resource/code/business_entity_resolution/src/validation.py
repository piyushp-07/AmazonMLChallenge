import numpy as np
import pandas as pd
from typing import Dict, Tuple


def compute_f05_per_entity(true_set: set, pred_set: set) -> Tuple[float, float, float]:
    """Compute precision, recall, F0.5 for one S1 entity."""
    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    
    # Precision and recall
    precision = 1.0 if (tp + fp == 0 and len(pred_set) == 0) else (tp / (tp + fp) if (tp + fp) > 0 else 0.0)
    recall = 1.0 if (tp + fn == 0 and len(true_set) == 0) else (tp / (tp + fn) if (tp + fn) > 0 else 0.0)
    
    # F0.5 score
    beta = 0.5
    beta_sq = beta * beta
    if precision == 0 and recall == 0:
        f05 = 0.0
    else:
        denom = beta_sq * precision + recall
        f05 = (1 + beta_sq) * precision * recall / denom if denom > 0 else 0.0
    
    return precision, recall, f05


def validate_predictions(
    true_mapping: Dict[str, set],
    pred_mapping: Dict[str, set]
) -> Dict:
    """
    Validate predictions against ground truth using macro F0.5.
    
    Args:
        true_mapping: dict of s1_id -> set(matched_ids) from ground truth
        pred_mapping: dict of s1_id -> set(matched_ids) from predictions
    
    Returns:
        dict with precision, recall, f05, tp, fp, fn, and per-entity metrics
    """
    results = {
        "per_entity": [],
        "tp_total": 0,
        "fp_total": 0,
        "fn_total": 0,
        "num_entities": 0,
        "num_singletons_true": 0,
        "num_singletons_pred": 0,
    }
    
    for s1_id in true_mapping:
        true_set = true_mapping[s1_id]
        pred_set = pred_mapping.get(s1_id, set())
        
        precision, recall, f05 = compute_f05_per_entity(true_set, pred_set)
        
        results["per_entity"].append({
            "s1_id": s1_id,
            "precision": precision,
            "recall": recall,
            "f05": f05,
            "tp": len(true_set & pred_set),
            "fp": len(pred_set - true_set),
            "fn": len(true_set - pred_set),
        })
        
        results["tp_total"] += len(true_set & pred_set)
        results["fp_total"] += len(pred_set - true_set)
        results["fn_total"] += len(true_set - pred_set)
        results["num_entities"] += 1
        
        if len(true_set) == 0:
            results["num_singletons_true"] += 1
        if len(pred_set) == 0:
            results["num_singletons_pred"] += 1
    
    per_entity = np.array([[e["precision"], e["recall"], e["f05"]] for e in results["per_entity"]])
    results["macro_precision"] = per_entity[:, 0].mean()
    results["macro_recall"] = per_entity[:, 1].mean()
    results["macro_f05"] = per_entity[:, 2].mean()
    
    return results


def print_validation_report(val_results: Dict):
    """Pretty-print validation results."""
    print("\n" + "="*70)
    print("VALIDATION REPORT")
    print("="*70)
    print(f"Entities evaluated: {val_results['num_entities']}")
    print(f"True singletons: {val_results['num_singletons_true']}")
    print(f"Predicted singletons: {val_results['num_singletons_pred']}")
    print(f"\nMacro Metrics (averaged per entity):")
    print(f"  Precision: {val_results['macro_precision']:.4f}")
    print(f"  Recall:    {val_results['macro_recall']:.4f}")
    print(f"  F0.5:      {val_results['macro_f05']:.4f}")
    print(f"\nAggregate Counts:")
    print(f"  TP: {val_results['tp_total']}")
    print(f"  FP: {val_results['fp_total']}")
    print(f"  FN: {val_results['fn_total']}")
    print("="*70 + "\n")
