"""Hand-computed foreground scores and batch-invariant aggregation."""

import pytest
import torch

from brain_mri_segmentation.evaluation.metrics import (
    ForegroundOverlap,
    binary_confusion,
    overlap_scores,
)


def test_hand_computed_overlap_excludes_background() -> None:
    prediction = torch.tensor([[1, 1], [0, 0]])
    truth = torch.tensor([[1, 0], [1, 0]])
    counts = binary_confusion(prediction, truth).tolist()
    assert counts == [1, 1, 1, 1]
    assert overlap_scores(counts) == {"dice": 0.5, "iou": 1 / 3, "pixel_accuracy": 0.5}


@pytest.mark.parametrize("value", [0, 1])
def test_perfect_masks_and_empty_foreground(value: int) -> None:
    mask = torch.full((1, 2, 2), value)
    scores = overlap_scores(binary_confusion(mask, mask).tolist())
    assert scores == {"dice": 1.0, "iou": 1.0, "pixel_accuracy": 1.0}


def test_empty_target_false_positive_is_zero() -> None:
    counts = binary_confusion(torch.ones(2, 2), torch.zeros(2, 2)).tolist()
    assert overlap_scores(counts) == {"dice": 0.0, "iou": 0.0, "pixel_accuracy": 0.0}


def test_pooling_is_batch_invariant_and_not_sample_average() -> None:
    prediction = torch.tensor([[[1, 1], [0, 0]], [[0, 0], [0, 0]]])
    truth = torch.tensor([[[1, 0], [1, 0]], [[0, 0], [0, 0]]])
    for score, expected in (("dice", 0.5), ("iou", 1 / 3)):
        whole, batches = ForegroundOverlap(score), ForegroundOverlap(score)
        whole.update(prediction, truth)
        for p, t in zip(prediction, truth, strict=True):
            batches.update(p, t)
        assert whole.compute().item() == pytest.approx(expected)
        assert batches.compute().item() == pytest.approx(expected)
        batches.reset()
        assert batches.counts.tolist() == [0, 0, 0, 0]


def test_invalid_input_rejected() -> None:
    with pytest.raises(ValueError, match="only 0 and 1"):
        binary_confusion(torch.tensor([[2]]), torch.tensor([[1]]))
    with pytest.raises(ValueError, match="matching"):
        binary_confusion(torch.zeros(2, 2), torch.zeros(1, 2, 2))
    with pytest.raises(ValueError, match="empty"):
        overlap_scores([0, 0, 0, 0])
