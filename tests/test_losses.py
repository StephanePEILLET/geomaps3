import torch

from geomaps.training.losses import BCEWithLogitsLoss, CrossEntropyWithLogitsLoss


def test_bce_loss_returns_scalar(torch_logits):
    targets = torch.randint(0, 2, torch_logits.shape, dtype=torch.float32)
    loss = BCEWithLogitsLoss()(torch_logits, targets)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_cross_entropy_with_one_hot_targets(torch_logits):
    # mask is C-channel one-hot: shape (B, C, H, W)
    cls_idx = torch.randint(0, torch_logits.shape[1], torch_logits.shape[2:])
    one_hot = torch.nn.functional.one_hot(cls_idx, num_classes=torch_logits.shape[1])
    one_hot = one_hot.permute(2, 0, 1).unsqueeze(0).repeat(torch_logits.shape[0], 1, 1, 1)
    loss = CrossEntropyWithLogitsLoss()(torch_logits, one_hot)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
