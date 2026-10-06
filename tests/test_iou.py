"""Our IoU family vs torchvision's reference implementations."""
import torch
from torchvision.ops import complete_box_iou_loss, distance_box_iou_loss, generalized_box_iou_loss

from src.losses.iou import bbox_iou, iou_loss, pairwise_iou


def rand_boxes(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    xy = torch.rand(n, 2, generator=g) * 100
    wh = torch.rand(n, 2, generator=g) * 50 + 1
    return torch.cat([xy, xy + wh], 1)


def test_giou_matches_torchvision():
    a, b = rand_boxes(64, 1), rand_boxes(64, 2)
    ours = iou_loss(a, b, "giou")
    ref = generalized_box_iou_loss(a, b, reduction="none")
    assert torch.allclose(ours, ref, atol=1e-5), (ours - ref).abs().max()


def test_diou_matches_torchvision():
    a, b = rand_boxes(64, 3), rand_boxes(64, 4)
    ours = iou_loss(a, b, "diou")
    ref = distance_box_iou_loss(a, b, reduction="none")
    assert torch.allclose(ours, ref, atol=1e-5), (ours - ref).abs().max()


def test_ciou_matches_torchvision():
    a, b = rand_boxes(64, 5), rand_boxes(64, 6)
    ours = iou_loss(a, b, "ciou")
    ref = complete_box_iou_loss(a, b, reduction="none")
    assert torch.allclose(ours, ref, atol=1e-4), (ours - ref).abs().max()


def test_identical_boxes_have_zero_loss():
    a = rand_boxes(16, 7)
    for kind in ("iou", "giou", "diou", "ciou"):
        assert iou_loss(a, a, kind).abs().max() < 1e-5


def test_pairwise_iou_shape_and_values():
    a = torch.tensor([[0., 0., 10., 10.]])
    b = torch.tensor([[0., 0., 10., 10.], [5., 5., 15., 15.], [20., 20., 30., 30.]])
    m = pairwise_iou(a, b)
    assert m.shape == (1, 3)
    assert abs(m[0, 0] - 1.0) < 1e-6
    assert abs(m[0, 1] - 25.0 / 175.0) < 1e-6   # inter 25, union 100+100-25
    assert m[0, 2] < 1e-6


def test_disjoint_giou_is_worse_than_iou():
    a = torch.tensor([[0., 0., 10., 10.]])
    b = torch.tensor([[50., 50., 60., 60.]])
    assert bbox_iou(a, b, "iou").item() == 0.0
    assert bbox_iou(a, b, "giou").item() < 0.0      # GIoU stays informative when IoU dies
