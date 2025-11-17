from typing import Optional
from einops import rearrange
import torch
import torch.nn as nn
from torchvision.transforms.functional import normalize
from torchvision.ops import nms, box_iou

from .backbone import SwinFPN
from .rpn import RPN
from .relationship_extractor import RelationshipFeatureExtractor, RelationshipExtractor
from ..box_tools.anchor import AnchorGenerator
from ..box_tools import box_coder
from ..box_tools.loss import select_anchors_atss, select_anchors_highest_iou
from ..data_container import ImageList, BoxList


class RELRCNN(nn.Module):
    def __init__(
        self,
        n_objcat:int = 150,
        n_relcat:int = 50,
        n_scene_cat:int = 0,
        n_proposals:int = 1024,
        nms_thresh:float = 0.5,
        num_transformer_layers:int = 4,
        pixel_mean = [0.406, 0.456, 0.485],
        pixel_std = [0.225, 0.224, 0.229],
        pe_type:str = 'random_fourier'
        ):
        super().__init__()
        self.backbone = SwinFPN()
        self.rpn = RPN(channels = 256, pe_type=pe_type)
        self.relationship_feature_extractor = RelationshipFeatureExtractor(in_dim=512, out_dim=256, num_layers=num_transformer_layers)
        self.relationship_extractor = RelationshipExtractor(n_objcat, n_relcat, in_dim=256)
        self.anchor_generator = AnchorGenerator()
        if n_scene_cat > 0:
            self.scene_classifier = nn.Linear(512, n_scene_cat)
        else:
            self.scene_classifier = None

        self.n_proposals = n_proposals
        self.nms_thresh = nms_thresh
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std

    def core(self, images: ImageList, boxes: Optional[list[BoxList]] = None, without_predicted_boxes:bool = False, interbatch:bool = False):
        """
        Parameters
        ----------
        images: ImageList
            The input images.
        boxes: Optional[list[BoxList]]
            If provided, the boxes are always output as the predicted boxes.
            It is used for training and evaluation that requires ground truth boxes.
        without_predicted_boxes: bool
            If True, the relationship and object classification are performed only for the boxes provided.

        Returns
        -------
        bbox_reg: torch.Tensor
            The predicted box regression. The shape is (B, HW, 4).
        objectness: torch.Tensor
            The predicted objectness for NMS. The shape is (B, HW, 1).
        object_classes: Union[list[torch.Tensor], torch.Tensor]
            The predicted object classes.
            If interbatch is False, returns a list of tensors, each tensor has the shape (N, n_objcat).
            If interbatch is True, returns a single tensor, the shape is (N_0 + N_1 + ... + N_B, n_objcat).
            If boxes are provided, the head is always output as the ground truth object classes.
            if interbatch is true and boxes are provided,
            the head is output as the ground truth object classes for the all boxes in the batch.
        relationship_classes: Union[list[torch.Tensor], torch.Tensor]
            The predicted relationship classes.initial_nms_idxs = nms(decoded_bboxes, objectness_scores, self.nms_thresh)
            The anchors for loss calculation. The length for each level is (H * W).
        pos_anchors: list[torch.Tensor]
            The positive anchors for ATSS loss calculation.
        target_idxs: list[torch.Tensor]
            The target indices for ATSS loss calculation.
        nmsed_indices: list[torch.Tensor]
            The indices of anchors that are selected by NMS. The shape for each tensor is (N,).
        """
        assert boxes is not None or not without_predicted_boxes, "If without_predicted_boxes is True, boxes must be provided."
        self_device = next(self.parameters()).device

        img_tensor = normalize(images.tensors.cuda(non_blocking=True).to(self_device), self.pixel_mean, self.pixel_std)

        image_features = self.backbone(img_tensor)
        bbox_reg, objectness, features = self.rpn(image_features)

        if self.scene_classifier is not None:
            scene_classes = self.scene_classifier(features[-1].mean(dim=(2, 3)))
        else:
            scene_classes = None

        anchors = self.anchor_generator(images, objectness)
        num_anchors_per_level = [len(anchors_per_level) for anchors_per_level in anchors[0]]
        anchors = [BoxList.concate_box_list(anc).to(self_device) for anc in anchors]

        bbox_reg   = torch.cat([rearrange(x, "B C H W -> B (H W) C") for x in bbox_reg  ], dim=1)
        objectness = torch.cat([rearrange(x, "B C H W -> B (H W) C") for x in objectness], dim=1)[..., 0]
        features   = torch.cat([rearrange(x, "B C H W -> B (H W) C") for x in features  ], dim=1)

        pos_anchors = []
        target_idxs = []
        nmsed_indices = []
        relationship_features = []

        if boxes is not None:
            for b in range(bbox_reg.shape[0]):
                pos_ancs, targ_idxs = select_anchors_atss(
                        anchors[b].bbox,
                        boxes[b].bbox,
                        num_anchors_per_level)
                pos_anchors.append(pos_ancs)
                target_idxs.append(targ_idxs)
                decoded_bboxes = box_coder.decode(bbox_reg[b], anchors[b].bbox)

                objectness_scores = objectness[b].detach().sigmoid()
                if self.training:
                    # Randomly switch two mode
                    if torch.rand(()) < 0.5:
                        # If nmsed_boxes include boxes that have high IoU with the ground truth boxes,
                        # the boxes are selected as positive anchors.
                        # Otherwise, randomly select positive anchors for each object.
                        initial_nms_idxs = nms(decoded_bboxes, objectness_scores, self.nms_thresh)
                        initial_selected_anchors = initial_nms_idxs[:self.n_proposals]
                        initial_selected_bboxes = decoded_bboxes[initial_selected_anchors]
                        initial_ious = box_iou(boxes[b].bbox, initial_selected_bboxes)
                        max_ious, max_idxs = initial_ious.max(dim=1)
                        max_idxs = initial_selected_anchors[max_idxs]
                        selected_anchors = []
                        for i in torch.unique(targ_idxs):
                            if max_ious[i] > 0.7:
                                selected_anchors.append(max_idxs[i])
                            else:
                                candidate_anchors = pos_ancs[targ_idxs == i]
                                i_selected = torch.randint(len(candidate_anchors), device="cuda", size=())
                                selected_anchors.append(candidate_anchors[i_selected])
                    else:
                        # Randomly select positive anchors for each object
                        selected_anchors = []
                        for i in torch.unique(targ_idxs):
                            candidate_anchors = pos_ancs[targ_idxs == i]
                            i_selected = torch.randint(len(candidate_anchors), device="cuda", size=())
                            selected_anchors.append(candidate_anchors[i_selected])
                    if len(selected_anchors) > 0:
                        selected_anchors = torch.stack(selected_anchors, dim=0)
                    else:
                        selected_anchors = torch.tensor([], device=self_device, dtype=torch.long)
                else:
                    selected_anchors = select_anchors_highest_iou(
                        anchors[b].bbox,
                        boxes[b].bbox,
                        allow_duplicate_targets_per_anchor=False)

                if not without_predicted_boxes:
                    # Set objectness for selected anchors as 1.0
                    # Set objectness for other anchors as 0.0
                    objectness_scores[pos_ancs] = 0.0
                    objectness_scores[selected_anchors] = 1.0

                    nms_idxs = nms(decoded_bboxes, objectness_scores, self.nms_thresh)
                    nms_idxs = nms_idxs[torch.all(selected_anchors[:, None] != nms_idxs, dim=0)]
                    nms_idxs = torch.cat([selected_anchors, nms_idxs])
                    nms_idxs = nms_idxs[:self.n_proposals]
                else:
                    nms_idxs = selected_anchors

                rel_feat = self.relationship_feature_extractor(features[b][nms_idxs])
                relationship_features.append(rel_feat)
                nmsed_indices.append(nms_idxs)

        else:
            for b in range(bbox_reg.shape[0]):
                objectness_scores = objectness[b].detach().sigmoid()
                decoded_bboxes = box_coder.decode(bbox_reg[b], anchors[b].bbox)
                nms_idxs = nms(decoded_bboxes, objectness_scores, self.nms_thresh)
                nms_idxs = nms_idxs[:self.n_proposals]
                rel_feat = self.relationship_feature_extractor(features[b][nms_idxs])
                relationship_features.append(rel_feat)
                pos_anchors.append(torch.tensor([], device=self_device))
                target_idxs.append(torch.tensor([], device=self_device))
                nmsed_indices.append(nms_idxs)

        if interbatch:
            if boxes is not None:
                relationship_features_with_gt = []
                relationship_features_wo_gt = []
                for rel_feat, box in zip(relationship_features, boxes):
                    relationship_features_with_gt.append(rel_feat[:len(box)])
                    relationship_features_wo_gt.append(rel_feat[len(box):])
                
                relationship_features = torch.cat(relationship_features_with_gt + relationship_features_wo_gt, dim=0)
            else:
                relationship_features = torch.cat(relationship_features, dim=0)
            object_classes, relationship_classes = self.relationship_extractor(relationship_features)
        else:
            object_classes = []
            relationship_classes = []
            for rel_feat in relationship_features:
                obj_cls, rel_cls = self.relationship_extractor(rel_feat)
                object_classes.append(obj_cls)
                relationship_classes.append(rel_cls)

        return (
            bbox_reg,        # For box regression
            objectness,      # For NMS
            object_classes,  # For object classification
            relationship_classes, # For relationship classification
            scene_classes,   # For scene classification
            anchors,         # For loss calculation
            pos_anchors,     # For atss loss calculation
            target_idxs,     # For atss loss calculation
            nmsed_indices    # For evaluation
        )
    
    def forward(self, images: ImageList, boxes: list[BoxList], interbatch:bool = False):
        """For training"""

        (
            bbox_reg,        # For box regression
            objectness,      # For NMS
            object_classes,  # For object classification
            relationship_classes, # For relationship classification
            scene_classes,   # For scene classification
            anchors,         # For loss calculation
            pos_anchors,     # For atss loss calculation
            target_idxs,     # For atss loss calculation
            nmsed_indices    # For evaluation
        ) = self.core(images, boxes, interbatch=interbatch)

        return (
            bbox_reg,        # For box regression
            objectness,      # For NMS
            object_classes,  # For object classification
            relationship_classes, # For relationship classification
            scene_classes,   # For scene classification
            anchors,         # For loss calculation
            pos_anchors,     # For atss loss calculation
            target_idxs      # For atss loss calculation
        )
    
    def inference(self, images: ImageList, boxes: Optional[list[BoxList]] = None):
        """For evaluation"""

        (
            bbox_reg,        # For box regression
            objectness,      # For NMS
            object_classes,  # For object classification
            relationship_classes, # For relationship classification
            scene_classes,   # For scene classification
            anchors,         # For loss calculation
            pos_anchors,     # For atss loss calculation
            target_idxs,     # For atss loss calculation
            nmsed_indices    # For evaluation
        ) = self.core(images, boxes)

        if boxes is None:
            boxes = []
            for b in range(len(nmsed_indices)):
                allboxes_per_image = box_coder.decode(
                        bbox_reg[b],
                        anchors[b].bbox
                    )
                boxes.append(
                    BoxList(
                        allboxes_per_image[nmsed_indices[b]],
                        images.image_sizes[b],
                        "xyxy"
                    ).clip_to_image(remove_empty=False)
                )

            return (
                object_classes,  # For object classification
                relationship_classes, # For relationship classification
                scene_classes, # For scene classification
                boxes
            )

        else:
            object_classes = [objcls[:len(b)] for objcls, b in zip(object_classes, boxes)]
            relationship_classes = [relcls[:len(b), :len(b)] for relcls, b in zip(relationship_classes, boxes)]
            return (
                object_classes,  # For object classification
                relationship_classes,  # For relationship classification,
                scene_classes,  # For scene classification
                boxes
            )