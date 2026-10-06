"""Hand-checkable assigner behaviour."""
import torch

from src.losses.assigner import center_prior_assign, tal_assign
from src.models.detector import make_points


def grid(size=64, strides=(8, 16, 32)):
    shapes = [(size // s, size // s) for s in strides]
    return make_points(shapes, strides, torch.device("cpu"))


def test_center_prior_picks_points_inside_the_box():
    points, stride = grid()
    gt = torch.tensor([[16., 16., 48., 48.]])
    labels = torch.tensor([3])
    fg, gi, tboxes, tscores = center_prior_assign(points, stride, gt, labels, 10, topk=9)
    assert fg.sum() > 0
    sel = points[fg]
    assert (sel[:, 0] > 16).all() and (sel[:, 0] < 48).all()
    assert (sel[:, 1] > 16).all() and (sel[:, 1] < 48).all()
    # every positive carries the GT's class as a one-hot target
    assert (tscores[fg].argmax(1) == 3).all()
    assert torch.allclose(tboxes[fg], gt.expand(int(fg.sum()), 4))


def test_no_gt_gives_no_positives():
    points, stride = grid()
    gt = torch.zeros(0, 4)
    labels = torch.zeros(0, dtype=torch.long)
    fg, _, _, tscores = center_prior_assign(points, stride, gt, labels, 10)
    assert fg.sum() == 0 and tscores.sum() == 0


def test_smaller_gt_wins_contested_point():
    """A point inside both boxes must go to the smaller object, which has fewer
    candidate points to spare."""
    points, stride = grid()
    big = [10., 10., 54., 54.]
    small = [28., 28., 36., 36.]
    gt = torch.tensor([big, small])
    labels = torch.tensor([0, 1])
    fg, gi, _, tscores = center_prior_assign(points, stride, gt, labels, 10, topk=9)
    centre = ((points[:, 0] - 32).abs() < 4) & ((points[:, 1] - 32).abs() < 4) & fg
    if centre.any():
        assert (tscores[centre].argmax(1) == 1).all()


def test_tal_prefers_the_point_whose_prediction_is_good():
    """Two in-box points; the one whose predicted box matches the GT should win."""
    points = torch.tensor([[20., 20.], [40., 40.]])
    stride = torch.tensor([8., 8.])
    gt = torch.tensor([[10., 10., 50., 50.]])
    labels = torch.tensor([2])
    scores = torch.full((2, 10), -4.0)
    scores[0, 2] = 4.0                      # confident, correct class
    pred = torch.tensor([[10., 10., 50., 50.],      # perfect
                         [45., 45., 48., 48.]])     # tiny, poor overlap
    fg, gi, _, tscores = tal_assign(points, stride, gt, labels, scores, pred, 10, topk=1)
    assert fg[0] and not fg[1]
    assert tscores[0].argmax().item() == 2
    assert tscores[0, 2] > 0                # soft target carries IoU quality


def test_tal_always_assigns_something_when_a_gt_is_in_frame():
    """Even with garbage predictions, a visible GT must get at least one positive,
    otherwise it contributes no gradient at all."""
    points, stride = grid()
    gt = torch.tensor([[16., 16., 48., 48.]])
    labels = torch.tensor([5])
    scores = torch.full((points.shape[0], 10), -10.0)
    pred = torch.zeros(points.shape[0], 4)
    fg, _, _, _ = tal_assign(points, stride, gt, labels, scores, pred, 10, topk=10)
    assert fg.sum() >= 1
