from einops import rearrange
import torch
from torch.nn import functional as F
from torchvision.ops import generalized_box_iou_loss, sigmoid_focal_loss, box_iou

from . import box_coder
from ..data_container import BoxList


def select_anchors_highest_iou(anchors: torch.Tensor, targets: torch.Tensor, allow_duplicate_targets_per_anchor: bool = True):
    """
    Select anchor for each ground truth for a single image.

    Parameters
    ----------
    anchors : torch.Tensor
        Anchors, sized [N, 4]. The format is (x1, y1, x2, y2).
    targets : torch.Tensor
        Ground truth bboxes, sized [M, 4]. The format is (x1, y1, x2, y2).
    allow_duplicate_targets_per_anchor : bool
        If True, allow a anchor to be matched to multiple targets.
        If False, it is not guaranteed that each anchor will be matched to at least one target.

    Returns
    -------
    positive_anchors_idx : torch.Tensor
        Index of selected anchors, sized [M].
    """
    N = len(anchors)
    M = len(targets)
    if not allow_duplicate_targets_per_anchor:
        assert N >= M, ("If allow_duplicate_targets_per_anchor is False,"
                        "the number of anchors must be greater than or equal to the number of targets.")

    ious = box_iou(anchors, targets)  # [N, M]
    if allow_duplicate_targets_per_anchor:
        positive_anchors_idx = ious.argmax(dim=0)
        return positive_anchors_idx
    else:
        positive_anchors_idx = torch.zeros(M, dtype=torch.long, device=ious.device) - 1
        for _ in range(M):
            i = ious.flatten().argmax()
            i_anc, i_tar = i // M, i % M
            positive_anchors_idx[i_tar] = i_anc
            ious[i_anc] = -1
            ious[:, i_tar] = -1
        return positive_anchors_idx


@torch.jit.script
def select_anchors_atss(anchors: torch.Tensor, targets: torch.Tensor, num_anchors_per_level: list[int], atss_topk: int = 9, allow_duplicate_targets_per_anchor: bool = True):
    """
    Select anchor for each ground truth for a single image.
    A difference from original ATSS is that this function selects at least one anchor for each ground truth.
    (Original ATSS may not select any anchor for some ground truths)

    Parameters
    ----------
    anchors : torch.Tensor
        Anchors, sized [N, 4]. The format is (x1, y1, x2, y2).
    targets : torch.Tensor
        Ground truth bboxes, sized [M, 4]. The format is (x1, y1, x2, y2).
    num_anchors_per_level : List[int]
        Number of anchors per level.
    atss_topk : int
        Number of anchors to be selected for each level.
    allow_duplicate_targets_per_anchor : bool
        If True, allow a anchor to be matched to multiple targets.
        If False, it is not guaranteed that each anchor will be matched to at least one target.

    Returns
    -------
    positive_anchors_idx : torch.Tensor
        Index of selected anchors. ndim is 1.
    matched_target_idx : torch.Tensor
        Index of matched ground truth. ndim is 1.
    """

    # Select anchors with topk smallest distance for each ground truth for each level
    anchors_center = (anchors[:, 2:] + anchors[:, :2]) / 2
    targets_center = (targets[:, 2:] + targets[:, :2]) / 2
    distance = torch.cdist(targets_center, anchors_center)  # [M, N]

    start = 0
    candidate = []
    for num_anchors in num_anchors_per_level:
        end = start + num_anchors
        distance_per_level = distance[:, start:end]
        _, topk_idx = distance_per_level.topk(min(atss_topk, num_anchors), dim=1, largest=False)  # [M, atss_topk]
        candidate.append(topk_idx + start)
        start = end
    candidate = torch.cat(candidate, dim=1)  # [M, atss_topk * num_levels]
    candidate_anchors = anchors[candidate]  # [M, atss_topk * num_levels, 4]

    # Select anchor based on mean and variance of IoUs from selected anchors
    xy_min = torch.max(candidate_anchors[:, :, :2], targets[:, None, :2])
    xy_max = torch.min(candidate_anchors[:, :, 2:], targets[:, None, 2:])
    xy = torch.clamp(xy_max - xy_min, min=0)
    area_i = xy[:, :, 0] * xy[:, :, 1]
    area_a = (candidate_anchors[:, :, 2] - candidate_anchors[:, :, 0]) \
           * (candidate_anchors[:, :, 3] - candidate_anchors[:, :, 1])
    area_t = (targets[:, 2] - targets[:, 0]) \
           * (targets[:, 3] - targets[:, 1])
    ious = area_i / (area_a + area_t[:, None] - area_i)
    higher_iou = ious >= ious.mean(dim=1, keepdim=True) + ious.std(dim=1, keepdim=True)  # [M, atss_topk * num_levels]
    highest_iou_idx = ious.argmax(dim=1)  # [M]

    # Select anchors which center is in the ground truth
    anchors_center = (candidate_anchors[:, :, 2:] + candidate_anchors[:, :, :2]) / 2
    is_in_xy = torch.logical_and(
        anchors_center - targets[:, None, :2] > 0,
        targets[:, None, 2:] - anchors_center > 0
    )
    is_in = torch.all(is_in_xy, dim=2)  # [M, atss_topk * num_levels]

    # Choose final matched anchors
    is_pos = higher_iou & is_in
    no_pos = torch.logical_not(is_pos.any(dim=1))     # Avoid targets with no positive anchors
    is_pos[no_pos, highest_iou_idx[no_pos]] = True    # by selecting the anchor with highest IoU
    positive_anchors_idx = candidate[is_pos]  # [P]
    matched_target_idx = is_pos.nonzero()[:, 0]  # [P]
    positive_ious = ious[is_pos]  # [P]

    # If there are anchors matched with multiple ground truth, choose the one with highest IoU
    if not allow_duplicate_targets_per_anchor:
        iou_argsort = torch.argsort(positive_anchors_idx * 2 + positive_ious)  # first sort by indexes of anchors, then by ious in ascending order
        positive_anchors_idx = positive_anchors_idx[iou_argsort]
        matched_target_idx = matched_target_idx[iou_argsort]
        positive_anchors_idx, counts = torch.unique_consecutive(positive_anchors_idx, return_counts=True)
        matched_target_idx = matched_target_idx[counts.cumsum(dim=0) - 1]
    
    return positive_anchors_idx, matched_target_idx


@torch.jit.script
def regression_loss_func(
    box_regression: torch.Tensor,
    objectness_for_nms: torch.Tensor,
    anchors: list[torch.Tensor],
    targets: list[torch.Tensor],
    positive_anchors_idx: list[torch.Tensor],
    matched_target_idx: list[torch.Tensor],
    alpha: float = 0.25,
    gamma: float = 2.0
):
    """
    Compute the loss for box regression and objectness.

    Parameters
    ----------
    box_regression : torch.Tensor
        Predicted box regression values for all anchors, sized [B, num_anchors, 4].
    objectness : torch.Tensor
        Predicted objectness values for all anchors, sized [B, num_anchors].
    anchors : List[torch.Tensor]
        Anchors for each image, each is sized [num_anchors, 4].
    targets : List[torch.Tensor]
        Ground truth boxes present in the image. Each element is a tensor of size [num_boxes, 4].
    positive_anchors_idx : List[torch.tensor]
        List of positive anchors index for each image. ndim of each tensor is 1.
    matched_target_idx : List[torch.tensor]
        List of matched ground truth index for each image. ndim of each tensor is 1.
    alpha : float
        Alpha parameter for focal loss.
    gamma : float
        Gamma parameter for focal loss.

    Returns
    -------
    regression_loss : torch.Tensor
        Loss for box regression. Scalar.
    objectness_loss : torch.Tensor
        Loss for objectness. Scalar.
    """
    objectness_losses_for_nms = []

    pos_box_regression = []
    pos_anchors = []
    pos_targets = []

    for i in range(len(positive_anchors_idx)):
        # focal loss for objectness_for_nms
        gt_objectness = torch.zeros_like(objectness_for_nms[i])
        gt_objectness[positive_anchors_idx[i]] = 1
        objectness_losses_for_nms.append(sigmoid_focal_loss(objectness_for_nms[i], gt_objectness, alpha=alpha, gamma=gamma, reduction="sum"))

        pos_box_regression.append(box_regression[i, positive_anchors_idx[i]])
        pos_anchors.append(anchors[i][positive_anchors_idx[i]])
        pos_targets.append(targets[i][matched_target_idx[i]])

    objectness_loss_for_nms = torch.stack(objectness_losses_for_nms).sum()
    pos_box_regression = torch.cat(pos_box_regression, dim=0)
    pos_anchors = torch.cat(pos_anchors, dim=0)
    pos_targets = torch.cat(pos_targets, dim=0)
    
    # Compute box regression loss
    pos_box_pred = box_coder.decode(pos_box_regression, pos_anchors)
    regression_loss = generalized_box_iou_loss(pos_box_pred, pos_targets).sum()

    return regression_loss, objectness_loss_for_nms


## not used
def boxclassification_loss_func(
    objlogits: torch.Tensor,
    box_labels: list[torch.Tensor],
    positive_anchors_idx: list[torch.Tensor],
    matched_target_idx: list[torch.Tensor]
):
    """
    Compute the loss for box classification.

    Parameters
    ----------
    objlogits: torch.Tensor
        Predicted box classification logits for all anchors, sized [batchsize, num_anchors, num_classes].
    box_labels : List[torch.Tensor]
        Ground truth box labels present in the image. Each element is a tensor of size [num_boxes].
    positive_anchors_idx : List[torch.tensor]
        List of positive anchors index for each image. ndim of each tensor is 1.
    matched_target_idx : List[torch.tensor]
        List of matched ground truth index for each image. ndim of each tensor is 1.

    Returns
    -------
    boxclassification_loss : torch.Tensor
        Sigmoid cross entropy loss for box classification. Scalar.
    """
    batchsize, num_anchors, _ = objlogits.shape
    objlogits = objlogits.view(batchsize * num_anchors, -1)
    positive_anchors_idx = torch.cat([i_posanchor + i * num_anchors for i, i_posanchor in enumerate(positive_anchors_idx)], dim=0)
    matched_target_labels = torch.cat([b_lbl[mt_idx] for b_lbl, mt_idx in zip(box_labels, matched_target_idx)])

    # Make one-hot labels whose shape is as same as `objlogits`
    boxclassification_labels = torch.zeros_like(objlogits)
    boxclassification_labels[positive_anchors_idx, matched_target_labels] = 1

    # Compute box classification loss
    boxclassification_loss = F.binary_cross_entropy_with_logits(objlogits, boxclassification_labels, reduction='sum')

    return boxclassification_loss