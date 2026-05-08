import torch

from geomaps.training.metrics import GeomapsMetrics, get_metrics_from_obs, torch_metrics_from_cm


def test_get_metrics_from_obs_perfect_prediction():
    metrics = get_metrics_from_obs(true_pos=10, false_neg=0, false_pos=0, true_neg=10)
    assert metrics["Accuracy"] > 0.999
    assert metrics["IoU"] > 0.999
    assert metrics["Precision"] > 0.999
    assert metrics["Recall"] > 0.999


def test_torch_metrics_from_cm_returns_expected_keys():
    cm = torch.tensor([[5, 1, 0], [0, 4, 1], [1, 0, 6]], dtype=torch.float32)
    out = torch_metrics_from_cm(cm, ["a", "b", "c"])
    for key in ["cm_macro", "cm_micro", "Overall/Accuracy", "Average/IoU", "a/IoU"]:
        assert key in out


def test_geomaps_metrics_update_and_compute():
    """Smoke test against pinned torchmetrics API (>=0.7, <1.0).

    Newer torchmetrics (>=1.0) renamed confusion_matrix to require a `task`
    argument. The code targets the pinned version listed in environment.yml.
    """
    import torchmetrics
    if tuple(int(p) for p in torchmetrics.__version__.split(".")[:2]) >= (1, 0):
        import pytest as _pytest
        _pytest.skip("torchmetrics>=1.0 changed confusion_matrix signature; pin to 0.x")

    metric = GeomapsMetrics(num_classes=3, class_labels=["a", "b", "c"])
    preds = torch.randint(0, 3, (4, 8, 8))
    target = preds.clone()
    metric.update(preds, target)
    out = metric.compute()
    assert out["Average/Accuracy"].item() > 99.0
