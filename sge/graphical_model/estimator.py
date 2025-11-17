from typing import Optional
from einops import rearrange, einsum
import numpy as np
import torch
from torchvision.transforms.functional import normalize
from torchvision.ops import nms

from .mplp import MPLPVariable, MPLPFactor
from .mplp import reset as mplp_reset
from .gibbs_sampling import GSVariable, GSFactor, gibbs_sampling
from .beliefprop import LogSumProdVariable, LogSumProdFactor
from .factorfuncs import ObjClsFactor, RelClsFactor, RelClsMaxFactor, DenseMatFactor
from .best_max_marginal_first import best_max_marginal_first
from ..data_container import BoxList
from ..data_container.imagelist import to_image_list
from ..box_tools import box_coder


class MAPEstimator:
    def __init__(
        self,
        max_mplp_iter=10,
        gibbs_iter=25, gibbs_start_temp=1.0, gibbs_end_temp=0.01
        ):
        self.max_mplp_iter = max_mplp_iter
        self.gibbs_iter = gibbs_iter
        self.gibbs_start_temp = gibbs_start_temp
        self.gibbs_end_temp = gibbs_end_temp

    def __call__(self, objness: torch.Tensor, relness: torch.Tensor, labels: torch.Tensor, rels: torch.Tensor, preds: torch.Tensor):
        torch_idx_types = (torch.int64, torch.int32, torch.int16)
        assert objness.ndim == 2
        assert relness.ndim == 3 and relness.shape[0] == relness.shape[1] and relness.shape[1] == len(objness)
        assert  labels.ndim == 1 and labels.dtype in torch_idx_types
        assert    rels.ndim == 2 and   rels.dtype in torch_idx_types and rels.shape[1] == 2
        assert   preds.ndim == 1 and  preds.dtype in torch_idx_types
        assert len(preds) == len(rels)

        obj_mplp_variables = []
        obj_gp_variables = []

        for l in labels:
            obj_variable = MPLPVariable(len(objness), device="cuda")
            obj_gp_variable = GSVariable(len(objness))
            f = ObjClsFactor(objness[:, l])
            MPLPFactor(f, [obj_variable])
            GSFactor(f, [obj_gp_variable])
            obj_mplp_variables.append(obj_variable)
            obj_gp_variables.append(obj_gp_variable)

        for (s, o), pred in zip(rels.cpu().numpy(), preds):
            fmax  = RelClsMaxFactor(relness[..., pred])
            fprob = RelClsFactor(relness[..., pred])
            MPLPFactor( fmax, [obj_mplp_variables[s], obj_mplp_variables[o]])
            GSFactor  (fprob, [  obj_gp_variables[s],   obj_gp_variables[o]])

        notsame_energy = torch.zeros(len(objness), len(objness)).to("cuda")
        notsame_energy[torch.arange(len(objness)), torch.arange(len(objness))] = float('-inf')
        notsame_factor = RelClsFactor(notsame_energy)

        for i in range(len(obj_mplp_variables)):
            obj_variable = obj_mplp_variables[i]
            obj_variable.run(self.max_mplp_iter)
            i_anchor = obj_variable.get_message().argmax()

            obj_gp_variable = obj_gp_variables[i]
            obj_gp_variable.set_initial(i_anchor)
            for j in range(len(obj_gp_variables)):
                if j > i:
                    GSFactor(notsame_factor, [obj_gp_variable, obj_gp_variables[j]])

        gibbs_sampling(obj_gp_variables, self.gibbs_iter, start_temp=self.gibbs_start_temp, end_temp=self.gibbs_end_temp)

        i_anchors = torch.tensor([obj_variable.x for obj_variable in obj_gp_variables])
        return i_anchors


class MLEstimator:
    def __init__(
        self,
        bp_iter=10,
        n_candidates=100,
        enable_not_same=False
        ):
        self.bp_iter = bp_iter
        self.n_candidates = n_candidates
        self.enable_not_same = enable_not_same

    def __call__(self, objness: torch.Tensor, relness: torch.Tensor, labels: torch.Tensor, rels: torch.Tensor, preds: torch.Tensor):
        torch_idx_types = (torch.int64, torch.int32, torch.int16)
        assert objness.ndim == 2
        assert relness.ndim == 3 and relness.shape[0] == relness.shape[1] and relness.shape[1] == len(objness)
        assert  labels.ndim == 1 and labels.dtype in torch_idx_types
        assert    rels.ndim == 2 and   rels.dtype in torch_idx_types and rels.shape[1] == 2
        assert   preds.ndim == 1 and  preds.dtype in torch_idx_types
        assert len(preds) == len(rels)

        obj_bp_variables = []

        for l in labels:
            obj_variable = LogSumProdVariable(len(objness), device="cuda")
            if l.item() >= 0:
                energy = objness[:, l]
            else:
                energy = objness.logsumexp(dim=1)
            f = ObjClsFactor(energy)
            LogSumProdFactor(f, [obj_variable])
            obj_bp_variables.append(obj_variable)

        for (s, o), pred in zip(rels.cpu().numpy(), preds):
            f = RelClsFactor(relness[..., pred])
            LogSumProdFactor(f, [obj_bp_variables[s], obj_bp_variables[o]])

        if self.enable_not_same:
            notsame_energy = torch.zeros(len(objness), len(objness)).to("cuda")
            notsame_energy[torch.arange(len(objness)), torch.arange(len(objness))] = float('-inf')
            notsame_factor = RelClsFactor(notsame_energy)

            for i in range(len(obj_bp_variables)):
                for j in range(len(obj_bp_variables)):
                    if j > i:
                        LogSumProdFactor(notsame_factor, [obj_bp_variables[i], obj_bp_variables[j]])

        i_candidates = []
        for obj_bp_variable in obj_bp_variables:
            likelihood = obj_bp_variable.get_loglikelihood()
            if likelihood is None:
                if self.bp_iter > 0:
                    obj_bp_variable.runLBP(self.bp_iter)
                    likelihood = obj_bp_variable.get_loglikelihood()
                else:
                    raise RuntimeError("BP cannot find a exact solution. Try to turn on loopy BP.")
            i_candidates.append(likelihood.argsort(descending=True)[:self.n_candidates])
        i_candidates = torch.stack(i_candidates)

        return i_candidates


def calculate_rank_of_true(pred_boxes, gt_boxes, threshold=0.5):
    assert pred_boxes.ndim == 3
    assert gt_boxes.ndim == 2
    assert pred_boxes.shape[2] == 4
    assert gt_boxes.shape[1] == 4
    assert pred_boxes.shape[0] == gt_boxes.shape[0]

    xp1, yp1, xp2, yp2 = pred_boxes[:, :, 0], pred_boxes[:, :, 1], pred_boxes[:, :, 2], pred_boxes[:, :, 3]
    x1, y1, x2, y2 = gt_boxes[:, None, 0], gt_boxes[:, None, 1], gt_boxes[:, None, 2], gt_boxes[:, None, 3]
    intersection = (torch.min(xp2, x2) - torch.max(xp1, x1)).clamp(min=0) * (torch.min(yp2, y2) - torch.max(yp1, y1)).clamp(min=0)
    union = (xp2 - xp1) * (yp2 - yp1) + (x2 - x1) * (y2 - y1) - intersection
    iou = intersection / union
    is_true = iou >= threshold

    ranks = []
    for i in range(is_true.shape[0]):
        indices = is_true[i].nonzero().flatten()
        if len(indices) == 0:
            rank = -1
        else:
            rank = indices[0].item()
        ranks.append(rank)

    return ranks


class TopMMAPEstimator:
    def __init__(
        self, model, anchor_generator,
        pixel_mean, pixel_std,
        max_mplp_iter=10,
        gibbs_iter=10, gibbs_start_temp=0.1, gibbs_end_temp=0.01,
        nms_thresh=0.7, nms_topk=1024, top_m=100
        ):
        self.model = model
        self.anchor_generator = anchor_generator
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std
        self.max_mplp_iter = max_mplp_iter
        self.gibbs_iter = gibbs_iter
        self.gibbs_start_temp = gibbs_start_temp
        self.gibbs_end_temp = gibbs_end_temp
        self.nms_thresh = nms_thresh
        self.nms_topk = nms_topk
        self.top_m = top_m

    def __call__(self, img: torch.Tensor, labels: torch.Tensor, rels: torch.Tensor, preds: torch.Tensor):
        torch_idx_types = (torch.int64, torch.int32, torch.int16)
        assert    img.ndim == 3
        assert labels.ndim == 1 and labels.dtype in torch_idx_types
        assert   rels.ndim == 2 and   rels.dtype in torch_idx_types and rels.shape[1] == 2
        assert  preds.ndim == 1 and  preds.dtype in torch_idx_types
        assert len(preds) == len(rels)

        img_tensor = normalize(img.unsqueeze(0), self.pixel_mean, self.pixel_std)
        oup_objcat = labels.unique()
        oup_relcat = preds.unique()

        with torch.inference_mode():
            (
                bbox_reg,
                objectness,
                objectness_for_nms,
                clslogits,
                proj_s,
                proj_o,
                proj_v
            ) = self.model(img_tensor, oup_objcat - 1, oup_relcat - 1)

        anchors = self.anchor_generator(to_image_list([img]), objectness)[0]
        anchors = BoxList.concate_box_list(anchors).bbox

        bbox_reg   = torch.cat([rearrange(x, "B C H W -> (B H W) C") for x in bbox_reg  ], dim=0)
        objectness = torch.cat([rearrange(x, "B C H W -> (B H W) C") for x in objectness], dim=0)
        objectness_for_nms = torch.cat([rearrange(x, "B C H W -> (B H W) C") for x in objectness_for_nms], dim=0)
        clslogits  = torch.cat([rearrange(x, "B C H W -> (B H W) C") for x in clslogits ], dim=0)
        proj_s = torch.cat([rearrange(x, "B C H W -> (B H W) C") for x in proj_s ], dim=0)
        proj_o = torch.cat([rearrange(x, "B C H W -> (B H W) C") for x in proj_o], dim=0)

        pred_boxes = box_coder.decode(bbox_reg, anchors)
        selected = nms(pred_boxes, objectness_for_nms.flatten().sigmoid(), self.nms_thresh)
        selected = selected[:self.nms_topk]
        objectness = objectness[selected]
        clslogits = clslogits[selected]
        proj_s = proj_s[selected]
        proj_o = proj_o[selected]
        proj_vso = einsum(proj_v, proj_s, proj_o, "V CS CO, S CS, O CO -> V S O")
        pred_boxes = pred_boxes[selected]

        obj_mplp_variables = []
        obj_gp_variables = []

        for l in labels:
            obj_variable = MPLPVariable(len(pred_boxes), device="cuda")
            obj_gp_variable = GSVariable(len(pred_boxes))
            if l == 0:
                f = ObjClsFactor(objectness[:, 0])
            else:
                i_label = torch.nonzero(oup_objcat == l)[0, 0].item()
                f = ObjClsFactor(objectness[:, 0] + clslogits[:, i_label])
            MPLPFactor(f, [obj_variable])
            GSFactor(f, [obj_gp_variable])
            obj_mplp_variables.append(obj_variable)
            obj_gp_variables.append(obj_gp_variable)

        for (s, o), pred in zip(rels.cpu().numpy(), preds):
            i_pred = torch.nonzero(oup_relcat == pred)[0, 0].item()
            fmax  = RelClsMaxFactor(proj_vso[i_pred])
            fprob = RelClsFactor(proj_vso[i_pred])
            MPLPFactor( fmax, [obj_mplp_variables[s], obj_mplp_variables[o]])
            GSFactor  (fprob, [  obj_gp_variables[s],   obj_gp_variables[o]])

        notsame_energy = torch.zeros(len(pred_boxes), len(pred_boxes)).to("cuda")
        notsame_energy[torch.arange(len(pred_boxes)), torch.arange(len(pred_boxes))] = float('-inf')
        notsame_factor = RelClsFactor(notsame_energy)
        for i in range(len(obj_gp_variables)):
            for j in range(len(obj_gp_variables)):
                if j > i:
                    GSFactor(notsame_factor, [obj_gp_variables[i], obj_gp_variables[j]])

        def target_function(*constraints : np.ndarray) -> tuple[float, np.ndarray]:
            mplp_reset(obj_mplp_variables)

            for constraint, obj_mplp_variable, obj_gp_variable in zip(constraints, obj_mplp_variables, obj_gp_variables):
                constraint = torch.from_numpy(constraint, device="cuda")
                obj_mplp_variable.set_mask_to_fix(constraint)
                obj_gp_variable.set_mask_to_fix(constraint)

            for obj_mplp_variable, obj_gp_variable in zip(obj_mplp_variables, obj_gp_variables):
                obj_mplp_variable.run(self.max_mplp_iter)
                obj_gp_variable.set_initial(obj_mplp_variable.get_message().argmax())

            score = gibbs_sampling(obj_gp_variables, self.gibbs_iter, start_temp=self.gibbs_start_temp, end_temp=self.gibbs_end_temp)
            assignment = np.array([obj_variable.x for obj_variable in obj_gp_variables])

            return score, assignment

        initial_constraints = [np.ones(len(pred_boxes), dtype=bool) for _ in range(len(obj_gp_variables))]
        _, bmmf_assignment = best_max_marginal_first(target_function, self.top_m, initial_constraints)

        selected_boxes = []
        for assignment in bmmf_assignment:
            i_candidates = torch.from_numpy(assignment, device="cuda")
            selected_boxes.append(pred_boxes[i_candidates])
        selected_boxes = torch.stack(selected_boxes)
        selected_boxes = rearrange(selected_boxes, "M N B -> N M B")

        # The shape of return value is (n_objs, self.top_m, 4)
        return selected_boxes