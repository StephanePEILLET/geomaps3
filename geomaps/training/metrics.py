"""Metrics utilities for the Geomaps DL pipeline.

Combines:
- GeomapsMetrics: torchmetrics Metric used by SegmentationTask during train/val/test.
- torch_metrics_from_cm: derives micro/macro/per-class metrics from a confusion matrix.
- get_metrics_from_obs: helper computing metrics from TP/FP/FN/TN counts.
"""
import torch
from torchmetrics import Metric
from torchmetrics.functional import confusion_matrix


SMOOTH = 1e-6


def get_metrics_from_obs(true_pos, false_neg, false_pos, true_neg, smooth=SMOOTH, micro=False):
    if micro:
        oa = true_pos / (true_pos + false_pos + smooth)
        iou = true_pos / (true_pos + false_pos + false_neg + smooth)
        return {"OA": oa, "IoU": iou}

    accuracy = (true_pos + true_neg) / (true_pos + false_pos + true_neg + false_neg + smooth)
    precision = true_pos / (true_pos + false_pos + smooth)
    recall = true_pos / (true_pos + false_neg + smooth)
    specificity = true_neg / (true_neg + false_pos + smooth)
    fpr = false_pos / (false_pos + true_neg + smooth)
    f1_score = (2 * true_pos) / (2 * true_pos + false_pos + false_neg + smooth)
    iou = true_pos / (true_pos + false_pos + false_neg + smooth)

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "F1-Score": f1_score,
        "IoU": iou,
        "FPR": fpr,
    }


def torch_metrics_from_cm(cm_macro, class_labels):
    """Derive per-class, micro and macro metrics from a confusion matrix tensor."""
    metrics_collection = {"cm_macro": cm_macro}

    stats_macro = torch.zeros(cm_macro.shape[0], 4).to(cm_macro.device)
    stats_macro[:, 0] = torch.diag(cm_macro)
    stats_macro[:, 1] = cm_macro.sum(0) - torch.diag(cm_macro)
    stats_macro[:, 2] = cm_macro.sum(1) - torch.diag(cm_macro)
    stats_macro[:, 3] = cm_macro.sum() - (stats_macro[:, 0:3].sum(1))

    cm_micro = stats_macro.sum(0)
    metrics_micro = get_metrics_from_obs(
        true_pos=cm_micro[0],
        false_pos=cm_micro[1],
        false_neg=cm_micro[2],
        true_neg=cm_micro[3],
        micro=True,
    )

    metrics_collection["cm_micro"] = cm_micro.reshape(2, 2)
    metrics_collection["Overall/Accuracy"] = metrics_micro["OA"] * 100
    metrics_collection["Overall/Precision"] = metrics_micro["IoU"] * 100

    metrics_by_class = get_metrics_from_obs(
        true_pos=stats_macro[:, 0],
        false_pos=stats_macro[:, 1],
        false_neg=stats_macro[:, 2],
        true_neg=stats_macro[:, 3],
    )

    for metric_name, metric_per_class in metrics_by_class.items():
        metrics_collection["Average/" + metric_name] = metric_per_class.mean() * 100
        for class_idx, class_label in enumerate(class_labels):
            metrics_collection[class_label + "/" + metric_name] = metric_per_class[class_idx] * 100

    return metrics_collection


class GeomapsMetrics(Metric):
    """torchmetrics Metric accumulating a confusion matrix and computing
    micro/macro/per-class segmentation metrics."""

    def __init__(self, num_classes, class_labels, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.num_classes = num_classes
        self.class_labels = class_labels
        default = torch.zeros(num_classes, num_classes, dtype=torch.long)
        self.add_state("confmat", default=default, dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape
        self.confmat += confusion_matrix(
            preds, target, num_classes=self.num_classes, normalize=None
        )

    def compute(self):
        return torch_metrics_from_cm(cm_macro=self.confmat, class_labels=self.class_labels)
