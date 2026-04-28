from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, precision_recall_fscore_support
from tqdm import tqdm

from .dataset_adapter import _ec_sort_key, ec_prefixes, parse_label_set


def sigmoid(logits):
    logits = np.asarray(logits, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-logits))


def choose_f1_thresholds(probs, targets, masks):
    n_classes = targets.shape[1]
    thresholds = np.full(n_classes, 0.5, dtype=np.float32)
    stats = {"classes": int(n_classes), "no_observed": 0, "no_positive": 0}

    for col in tqdm(range(n_classes), desc="validation thresholds", unit="class"):
        observed = masks[:, col] > 0
        if not np.any(observed):
            stats["no_observed"] += 1
            thresholds[col] = 1.0
            continue
        y_true = targets[observed, col].astype(np.int32)
        y_score = probs[observed, col]
        if y_true.sum() == 0:
            stats["no_positive"] += 1
            thresholds[col] = 1.0
            continue
        precision, recall, candidate_thresholds = precision_recall_curve(y_true, y_score)
        if len(candidate_thresholds) == 0:
            thresholds[col] = 0.5
            continue
        f1 = (2.0 * precision[:-1] * recall[:-1]) / np.maximum(
            precision[:-1] + recall[:-1],
            1e-12,
        )
        thresholds[col] = float(candidate_thresholds[int(np.argmax(f1))])
    return thresholds, stats


def _binary_metrics(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def _macro_metrics(targets, predictions, masks, start, end):
    per_class = []
    for col in range(start, end):
        observed = masks[:, col] > 0
        if not np.any(observed):
            continue
        per_class.append(
            _binary_metrics(targets[observed, col].astype(int), predictions[observed, col].astype(int))
        )
    if not per_class:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "observed_classes": 0}
    return {
        "precision": float(np.mean([item["precision"] for item in per_class])),
        "recall": float(np.mean([item["recall"] for item in per_class])),
        "f1": float(np.mean([item["f1"] for item in per_class])),
        "observed_classes": int(len(per_class)),
    }


def compute_hitec_metrics(logits, targets, masks, thresholds, vocab):
    probs = sigmoid(logits)
    predictions = (probs >= thresholds.reshape(1, -1)).astype(np.int32)
    target_int = targets.astype(np.int32)
    mask_bool = masks > 0

    observed_target = target_int[mask_bool]
    observed_pred = predictions[mask_bool]
    micro = _binary_metrics(observed_target, observed_pred)

    exact_rows = []
    for row_idx in range(target_int.shape[0]):
        row_mask = mask_bool[row_idx]
        if not np.any(row_mask):
            continue
        exact_rows.append(bool(np.all(target_int[row_idx, row_mask] == predictions[row_idx, row_mask])))

    per_level = {}
    level_starts = vocab["level_starts"]
    output_dims = vocab["output_dims"]
    for level_idx, start in enumerate(level_starts):
        end = start + output_dims[level_idx]
        level_mask = mask_bool[:, start:end]
        if np.any(level_mask):
            level_micro = _binary_metrics(
                target_int[:, start:end][level_mask],
                predictions[:, start:end][level_mask],
            )
        else:
            level_micro = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        per_level["level_{}".format(level_idx + 1)] = {
            "micro": level_micro,
            "macro": _macro_metrics(target_int, predictions, masks, start, end),
        }

    return {
        "micro": micro,
        "macro": _macro_metrics(target_int, predictions, masks, 0, target_int.shape[1]),
        "exact_match_accuracy": float(np.mean(exact_rows)) if exact_rows else 0.0,
        "rows": int(target_int.shape[0]),
        "observed_targets": int(mask_bool.sum()),
        "per_level": per_level,
    }


def rank_full_ecs(probs, vocab, rank_limit=50):
    l4_start = vocab["level_starts"][3]
    l4_labels = list(vocab["levels"][3])
    l4_probs = probs[:, l4_start : l4_start + len(l4_labels)]
    limit = len(l4_labels) if rank_limit is None or rank_limit <= 0 else min(rank_limit, len(l4_labels))
    ranked_rows = []
    for row in tqdm(l4_probs, desc="CARE ranks", unit="row"):
        top = np.argsort(-row)[:limit]
        ranked_rows.append([l4_labels[index] for index in top])
    return ranked_rows


def write_care_ranked_csv(manifest, probs, vocab, output_csv, rank_limit=50):
    metadata_cols = [
        col
        for col in [
            "Entry",
            "Original Entry",
            "EC number",
            "Sequence",
            "Original Sequence Length",
            "Sequence Length",
            "uniprot_date",
            "pdbs",
            "pdb_source",
            "pdb_type",
            "pdb_count",
        ]
        if col in manifest.columns
    ]
    ranked_rows = rank_full_ecs(probs, vocab, rank_limit=rank_limit)
    rank_df = pd.DataFrame(
        ranked_rows,
        columns=[str(index) for index in range(len(ranked_rows[0]) if ranked_rows else 0)],
    )
    care_df = pd.concat([manifest.loc[:, metadata_cols].reset_index(drop=True), rank_df], axis=1)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    care_df.to_csv(output_csv, index=False)
    return care_df


def _rank_columns(df):
    columns = []
    for column in df.columns:
        if str(column).isdigit():
            columns.append(column)
    return sorted(columns, key=lambda value: int(str(value)))


def _true_prefixes(ec_value):
    parsed = parse_label_set(ec_value)
    return [item["prefixes"] for item in parsed if item["prefixes"]]


def compute_care_metrics(care_df, k_values=(1, 20)):
    rank_cols = _rank_columns(care_df)
    metrics = {}
    for k in k_values:
        top_cols = rank_cols[:k]
        level_scores = {1: [], 2: [], 3: [], 4: []}
        for _, row in care_df.iterrows():
            predictions = [str(row[col]) for col in top_cols if pd.notna(row[col])]
            true_prefix_lists = _true_prefixes(row["EC number"])
            for level in (1, 2, 3, 4):
                label_hits = []
                for prefixes in true_prefix_lists:
                    if len(prefixes) < level:
                        continue
                    true_prefix = prefixes[level - 1]
                    pred_hit = any(
                        len(ec_prefixes(pred)) >= level
                        and ec_prefixes(pred)[level - 1] == true_prefix
                        for pred in predictions
                    )
                    label_hits.append(1.0 if pred_hit else 0.0)
                if label_hits:
                    level_scores[level].append(float(np.mean(label_hits)))
        metrics["k={}".format(k)] = {
            "level_{}_accuracy".format(level): round(
                float(np.mean(level_scores[level])) * 100.0,
                4,
            )
            if level_scores[level]
            else 0.0
            for level in (4, 3, 2, 1)
        }
        metrics["k={}".format(k)].update(
            {
                "level_{}_support".format(level): int(len(level_scores[level]))
                for level in (4, 3, 2, 1)
            }
        )
    return metrics


def compute_supplemental_ranking_metrics(care_df, hit_ks=(1, 3, 5, 10, 20)):
    rank_cols = _rank_columns(care_df)
    reciprocal_ranks = []
    hits = {k: [] for k in hit_ks}

    for _, row in care_df.iterrows():
        predictions = [str(row[col]) for col in rank_cols if pd.notna(row[col])]
        true_prefix_lists = _true_prefixes(row["EC number"])
        deepest = []
        for prefixes in true_prefix_lists:
            deepest.append((len(prefixes), prefixes[-1]))
        first_rank = None
        for rank, pred in enumerate(predictions, start=1):
            pred_prefixes = ec_prefixes(pred)
            matched = False
            for depth, true_prefix in deepest:
                if len(pred_prefixes) >= depth and pred_prefixes[depth - 1] == true_prefix:
                    matched = True
                    break
            if matched:
                first_rank = rank
                break
        reciprocal_ranks.append(0.0 if first_rank is None else 1.0 / first_rank)
        for k in hit_ks:
            hits[k].append(first_rank is not None and first_rank <= k)

    return {
        "mrr": round(float(np.mean(reciprocal_ranks)), 6) if reciprocal_ranks else 0.0,
        "rank_columns": int(len(rank_cols)),
        **{
            "hit@{}".format(k): round(float(np.mean(values)) * 100.0, 4) if values else 0.0
            for k, values in hits.items()
        },
    }


def sorted_label_string(labels):
    return ";".join(sorted(labels, key=_ec_sort_key))
