# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from typing import List
import einops
import torch
from torchvision.ops import nms

from . import box_coder
from ..data_container import BoxList


def boxlist_nms(
    boxlist: BoxList,
    nms_thresh: float,
    max_proposals: int = -1):
    """
    Performs non-maximum suppression on a boxlist, with scores specified
    in a boxlist field via score_field.

    Parameters
    ----------
    boxlist : BoxList
        BoxList to perform NMS on
    nms_thresh : float
        NMS threshold
    max_proposals : int, optional
        if > 0, then only the top max_proposals are kept after non-maximum suppression
    
    Returns
    -------
    boxlist : BoxList
        BoxList after NMS
    """
    if nms_thresh <= 0:
        return boxlist
    mode = boxlist.mode
    boxlist = boxlist.convert("xyxy")
    boxes = boxlist.bbox
    scores = boxlist.get_field("scores")
    labels = boxlist.get_field("labels")

    keep = []
    unique_labels = torch.unique(labels)
    for j in unique_labels:
        inds = (labels == j).nonzero().view(-1)

        scores_j = scores[inds]
        boxes_j = boxes[inds, :].view(-1, 4)
        keep_j = nms(boxes_j, scores_j, nms_thresh)

        keep += inds[keep_j].tolist()

    if max_proposals > 0:
        keep = keep[:max_proposals]
    boxlist = boxlist[keep]

    return boxlist.convert(mode)


def remove_small_boxes(boxlist, min_size):
    """
    Only keep boxes with both sides >= min_size

    Parameters
    ----------
    boxlist : BoxList
        BoxList to perform filtering on
    min_size : int
        minimum size of the box

    Returns
    -------
    boxlist : BoxList
        BoxList after filtering
    """
    # WORK AROUND: work around unbind using split + squeeze.
    xywh = boxlist.convert("xywh").bbox
    ws = xywh[:, 2]
    hs = xywh[:, 3]
    keep = ((ws >= min_size) & (hs >= min_size)).nonzero().squeeze(1)
    return boxlist[keep]


class GLIPPostProcessor:
    """ Compute boxes and scores from a list of box regression, centerness, anchors, and dot product logits. """

    def __init__(
            self,
            pre_nms_thresh = 0.05,  # cfg.MODEL.ATSS.INFERENCE_TH
            pre_nms_top_n = 100,  # cfg.MODEL.ATSS.DETECTIONS_PER_IMG
            nms_thresh = 0.6,  # cfg.MODEL.ATSS.NMS_TH
            fpn_post_nms_top_n = 100,  # cfg.MODEL.ATSS.DETECTIONS_PER_IMG
            min_size = 0,
            mdetr_style_aggregate_class_num=100  #cfg.TEST.MDETR_STYLE_AGGREGATE_CLASS_NUM
    ):
        super().__init__()
        self.pre_nms_thresh = pre_nms_thresh
        self.pre_nms_top_n = pre_nms_top_n
        self.nms_thresh = nms_thresh
        self.fpn_post_nms_top_n = fpn_post_nms_top_n
        self.min_size = min_size
        self.mdetr_style_aggregate_class_num = mdetr_style_aggregate_class_num

    def call_for_single_feature_map(self, box_regression, centerness, anchors, clslogits):
        box_regression = einops.rearrange(box_regression, "B C H W -> B (H W) C", C=4)
        centerness = einops.rearrange(centerness, "B C H W -> B (H W) C", C=1)
        centerness = centerness.sigmoid()
        clslogits = einops.rearrange(clslogits, "B C H W -> B (H W) C")
        box_cls = clslogits.sigmoid()

        ### multiply the classification scores with centerness scores ###
        box_cls *= centerness

        results = []

        # for each image
        for per_box_cls, per_box_regression, per_anchors in zip(box_cls, box_regression, anchors):
            ### remove low scoring boxes
            per_candidate_inds = per_box_cls > self.pre_nms_thresh  # (H * W, C)
            per_box_cls = per_box_cls[per_candidate_inds]  # 1D
            per_candidate_nonzeros = per_candidate_inds.nonzero()
            # ^-- `nonzero` returns (N, 2), where the second dimension is (box, class)

            ### remove boxes overflowing the maximum number of boxes
            per_pre_nms_top_n = torch.sum(per_candidate_inds).clamp(max=self.pre_nms_top_n)
            per_box_cls, top_k_indices = per_box_cls.topk(per_pre_nms_top_n, sorted=False)
            per_candidate_nonzeros = per_candidate_nonzeros[top_k_indices, :]

            ### decode the boxes
            per_box_loc = per_candidate_nonzeros[:, 0]
            per_class = per_candidate_nonzeros[:, 1]  # class index starts from 0
            detections = box_coder.decode(
                per_box_regression[per_box_loc, :],
                per_anchors.bbox[per_box_loc, :])

            ### create the boxlist
            boxlist = BoxList(detections, per_anchors.size, mode="xyxy")
            boxlist.add_field("labels", per_class)
            boxlist.add_field("scores", torch.sqrt(per_box_cls))  # not sure why sqrt is used here
            boxlist = boxlist.clip_to_image(remove_empty=False)
            boxlist = remove_small_boxes(boxlist, self.min_size)
            results.append(boxlist)

        return results

    def __call__(
            self,
            box_regression : List[torch.Tensor],
            centerness : List[torch.Tensor],
            anchors : List[List[BoxList]],
            clslogits : List[torch.Tensor]
            ) -> List[BoxList]:
        """ Box decoder for a list of feature maps. It perform NMS.

        Parameters
        ----------
        box_regression: List[Tensor]
            A list of Tensors of box regression for each feature level. Each tensor has shape `(N, 4, H, W)`.
        centerness: List[Tensor]
            A list of Tensors of centerness for each feature level. Each tensor has shape `(N, 1, H, W)`.
        anchors: List[List[BoxList]]
            A list of N BoxLists, one for each feature level.
        dot_product_logits: List[Tensor]
            A list of Tensors of dot product logits for each feature level. Each tensor has shape `(N, seq_len, H, W)`.
        positive_map: List[List[int]]
            A dict of positive map for each object.

        Returns
        -------
        boxlists: List[BoxList]
            A list of BoxLists, one for each image, containing the result of the computation
        """
        sampled_boxes = []
        anchors = list(zip(*anchors))  # Convert from each image to each feature level
        for b, c, a, cls in zip(box_regression, centerness, anchors, clslogits):
            sampled_boxes.append(
                self.call_for_single_feature_map(b, c, a, cls)
            )

        boxlists = list(zip(*sampled_boxes))  # Convert from each feature level to each image
        boxlists = [BoxList.concate_box_list(boxlist) for boxlist in boxlists]
        boxlists = self.select_over_all_levels(boxlists)

        return boxlists

    # TODO very similar to filter_results from PostProcessor
    # but filter_results is per image
    # TODO Yang: solve this issue in the future. No good solution
    # right now.
    # TODO Otani: Concatenating box_cls, box_regression, and centerness first,
    # and then executing postprocessor is better.
    def select_over_all_levels(self, boxlists):
        num_images = len(boxlists)
        results = []
        for i in range(num_images):
            # multiclass nms
            result = boxlist_nms(boxlists[i], self.nms_thresh)
            number_of_detections = len(result)

            # Limit to max_per_image detections **over all classes**
            if number_of_detections > self.fpn_post_nms_top_n > 0:
                cls_scores = result.get_field("scores")
                image_thresh, _ = torch.kthvalue(
                    # TODO: confirm with Pengchuan and Xiyang, torch.kthvalue is not implemented for 'Half'
                    # cls_scores.cpu(),
                    cls_scores.cpu().float(),
                    number_of_detections - self.fpn_post_nms_top_n + 1
                )
                keep = cls_scores >= image_thresh.item()
                keep = torch.nonzero(keep).squeeze(1)
                result = result[keep]
            results.append(result)
        return results