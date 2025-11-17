import numpy as np
import torch
from ..data_container import BoxList, ImageList
from ..graphical_model.beliefprop import LogSumProdVariable, LogSumProdFactor, get_sum_of_logZ
from ..graphical_model.factorfuncs import ObjClsFactor, RelClsFactor, DenseMatFactor
from ..box_tools.loss import regression_loss_func


def belief_propagation_loss_func(
    object_classes, relationship_classes,
    clslabels, attributes, rels, preds
):
    """
    Parameters
    ----------
    object_classes : torch.Tensor
        The K head logits are corresponding to the K ground truth boxes.
        Shape is (N, C)
    relationship_classes : torch.Tensor
        The K head logits are corresponding to the K ground truth boxes.
        Shape is (N, N, V)
    clslabels : torch.Tensor
        Index of the ground truth class for each box. Shape is (K,)
    attributes : list of list of int or None
        Each element is a list of attribute indices for each box.
    rels : torch.Tensor
        Shape is (R, 2)
    preds : torch.Tensor
        Shape is (R,)
    """
    N = len(object_classes)
    K = len(clslabels)

    variables = {}
    energy = 0.0

    for i_obj in range(K):
        variable = LogSumProdVariable(N)
        clslabel = clslabels[i_obj]
        f = ObjClsFactor(object_classes[:, clslabel])
        LogSumProdFactor(f, [variable])
        energy = energy + f.get_conditional_energy([i_obj])
        variables[i_obj] = variable
    
    if attributes is not None:
        for i_obj, attrs in enumerate(attributes):
            for attr in attrs:
                f = ObjClsFactor(object_classes[:, attr])
                LogSumProdFactor(f, [variables[i_obj]])
                energy = energy + f.get_conditional_energy([i_obj])

    for (s, o), pred in zip(rels, preds):
        s = s.item()
        o = o.item()
        f = RelClsFactor(relationship_classes[..., pred])
        LogSumProdFactor(f, [variables[s], variables[o]])
        energy = energy + f.get_conditional_energy([s, o])

    logZ = get_sum_of_logZ(variables.values())
    
    nll = logZ - energy
    return nll


def relationship_detection_loss(
    object_classes, relationship_classes,
    clslabels, rels, preds
):
    """
    Parameters are the same as `belief_propagation_loss_func`.
    """
    if len(rels) == 0:
        return (
            torch.tensor(0.0, device=object_classes.device, dtype=torch.float32),
            torch.tensor(0.0, device=object_classes.device, dtype=torch.float32)
        )

    N = len(object_classes)            # Number of boxes
    K = len(clslabels)                 # Number of ground truth boxes
    C = object_classes.shape[1]        # Number of object categories
    V = relationship_classes.shape[2]  # Number of relationship categories

    s_variable = LogSumProdVariable(N)
    o_variable = LogSumProdVariable(N)
    s_label_variable = LogSumProdVariable(C).to("cuda")
    o_label_variable = LogSumProdVariable(C).to("cuda")
    v_label_variable = LogSumProdVariable(V).to("cuda")

    s_label_factor = DenseMatFactor(object_classes)
    o_label_factor = DenseMatFactor(object_classes)
    v_label_factor = DenseMatFactor(relationship_classes)

    LogSumProdFactor(s_label_factor, [s_variable, s_label_variable])
    LogSumProdFactor(o_label_factor, [o_variable, o_label_variable])
    LogSumProdFactor(v_label_factor, [s_variable, o_variable, v_label_variable])

    logZ = s_variable.get_logZ()  # This graph is a tree, so LBP is not needed.
    
    s_anchor = rels[:, 0]
    o_anchor = rels[:, 1]
    s_energy = object_classes[s_anchor, clslabels[s_anchor]]
    o_energy = object_classes[o_anchor, clslabels[o_anchor]]
    v_energy = relationship_classes[s_anchor, o_anchor, preds]
    energy = s_energy + o_energy + v_energy
    mean_energy = energy.mean()

    nll = logZ - mean_energy
    pseudo_nll_of_v_label = torch.mean(relationship_classes[s_anchor, o_anchor].logsumexp(dim=-1) - v_energy)
    return nll, pseudo_nll_of_v_label


class CalculateLosses:
    def __init__(self, model, is_object_cat, category_drop_rate, interbatch=False, box_classification_loss_func="softmax_cross_entropy"):
        self.model = model
        self.is_object_cat = is_object_cat
        self.category_drop_rate = category_drop_rate
        self.interbatch = interbatch
        self.box_classification_loss_func = box_classification_loss_func

    def __call__(self, b_imgs: ImageList, b_boxes: list[BoxList], b_rels, b_preds, b_scenelabels):
        ###################################################################
        ### 1. Forward the network and prepare for the loss calculation ###
        ###################################################################
        b_imgs = b_imgs.to("cuda")
        b_boxes = [boxes.to("cuda") for boxes in b_boxes]

        (
            b_bbox_reg,        # For box regression
            b_objectness,      # For NMS
            b_object_classes,  # For object classification
            b_relationship_classes, # For relationship classification
            b_scene_classes,   # For scene classification
            b_anchors,         # For loss calculation
            b_pos_anchors,     # For atss loss calculation
            b_target_idxs,     # For atss loss calculation
        ) = self.model(b_imgs, b_boxes, interbatch=self.interbatch)

        b_clslabels = [boxes.get_field("labels") for boxes in b_boxes]
        b_attributes = []

        for boxes in b_boxes:
            if boxes.has_field("attributes"):
                if self.box_classification_loss_func == "softmax_cross_entropy":
                    raise ValueError("Attribute classification is not supported with softmax_cross_entropy loss.")
                b_attributes.append(boxes.get_field("attributes"))
            else:
                b_attributes.append(None)

        losses = {}

        #################################################
        ### 2. Loss for the scene classification part ###
        #################################################
        if b_scene_classes is not None:
            k_hot_labels = torch.zeros_like(b_scene_classes)
            for i, labels in enumerate(b_scenelabels):
                for label in labels:
                    k_hot_labels[i, label] = 1.0
            scene_classification_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                b_scene_classes, k_hot_labels, reduction="mean")
        else:
            scene_classification_loss = 0.0
        
        losses["scene_classification"] = scene_classification_loss

        ###############################################
        ### 3. Losses for the object detection part ###
        ###############################################
        regression_loss, objectness_loss = regression_loss_func(
            b_bbox_reg,
            b_objectness,
            [anchors.bbox for anchors in b_anchors],
            [boxes.bbox for boxes in b_boxes],
            b_pos_anchors,
            b_target_idxs
        )

        del b_bbox_reg, b_objectness, b_anchors, b_pos_anchors, b_target_idxs

        ### If interbatch is True, all data are concatenated to a single batch.
        if self.interbatch:
            batch_starts_from = np.cumsum([0] + [len(boxes) for boxes in b_boxes])[:-1]
            b_object_classes = b_object_classes[None]  # b_object_classes is already concatenated
            b_clslabels = [torch.cat(b_clslabels, dim=0)]
            b_relationship_classes = b_relationship_classes[None]  # b_relationship_classes is already concatenated
            b_rels = [torch.cat([rels + start for rels, start in zip(b_rels, batch_starts_from)], dim=0)]
            b_preds = [torch.cat(b_preds, dim=0)]

        #################################################
        ### 4. The loss for the object classification ###
        #################################################
        if self.box_classification_loss_func == "softmax_cross_entropy":
            boxclassification_loss = 0.0
            # Ignore `b_attributes` because it is not supported with softmax_cross_entropy loss
            for object_classes, clslabels in zip(b_object_classes, b_clslabels):
                n_candidates, n_classes = object_classes.shape
                object_classes = torch.cat([object_classes, torch.zeros(len(object_classes), 1, device=object_classes.device)], dim=1)
                clslabels = torch.cat([
                    clslabels,
                    torch.full([n_candidates - len(clslabels)], n_classes, dtype=clslabels.dtype, device=clslabels.device)
                    ])
                boxclassification_loss = boxclassification_loss + \
                    torch.nn.functional.cross_entropy(
                        object_classes,
                        clslabels,
                        reduction="mean")

        elif self.box_classification_loss_func == "sigmoid_cross_entropy":
            boxclassification_loss = 0.0
            for object_classes, clslabels, attributes in zip(b_object_classes, b_clslabels, b_attributes):
                n_candidates, n_classes = object_classes.shape
                onehot_clslabels = torch.nn.functional.one_hot(clslabels, num_classes=n_classes).float()
                if attributes is not None:
                    for i, attrs in enumerate(attributes):
                        for attr in attrs:
                            onehot_clslabels[i, attr] = 1.0
                onehot_clslabels = torch.cat([
                    onehot_clslabels,
                    torch.zeros(n_candidates - len(clslabels), n_classes, device=onehot_clslabels.device)
                    ], dim=0)
                boxclassification_loss = boxclassification_loss + \
                    torch.nn.functional.binary_cross_entropy_with_logits(
                        object_classes,
                        onehot_clslabels,
                        reduction="mean")

        losses["object_detection"] = {
            "regression": regression_loss,
            "objectness": objectness_loss,
            "boxclassification": boxclassification_loss
        }

        ##############################################
        ### 5. Losses for the graph matching part  ###
        ##############################################
        # Make last channel of `clslogits` as no-label class and change labels to no-label class randomly
        i_nolabel = b_object_classes[0].shape[1]
        b_object_classes = [
            torch.cat([object_classes, object_classes[:, self.is_object_cat].logsumexp(dim=1, keepdim=True)], dim=1)
            for object_classes in b_object_classes]
        for objlabels in b_clslabels:
            drop_cls = torch.rand(len(objlabels)) < self.category_drop_rate
            objlabels[drop_cls] = i_nolabel

        losses["graph_matching"] = {}

        for mode in ("wholebox", "boxgiven"):
            matchloss_bp = 0.0

            for b in range(len(b_clslabels)):
                object_classes = b_object_classes[b]
                attributes = b_attributes[b]
                relationship_classes = b_relationship_classes[b]
                clslabels = b_clslabels[b]
                rels = b_rels[b]
                preds = b_preds[b]

                if mode == "boxgiven":
                    object_classes = object_classes[:len(clslabels)]
                    relationship_classes = relationship_classes[:len(clslabels), :len(clslabels)]

                matchloss_bp = matchloss_bp + \
                    belief_propagation_loss_func(
                        object_classes, relationship_classes,
                        clslabels, attributes, rels, preds)

            losses["graph_matching"][mode] = {
                "belief_propagation": matchloss_bp
            }

        return losses