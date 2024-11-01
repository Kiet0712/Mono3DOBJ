# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import numpy as np

from utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou, box_xyxy_to_cxcywh, box_cxcylrtb_to_xyxy


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_3dcenter: float = 1, cost_bbox: float = 1, cost_giou: float = 1, cost_depth: float=1, cost_dim: float=1, cost_angle: float=1):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_3dcenter = cost_3dcenter
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cost_depth = cost_depth
        self.cost_dim = cost_dim
        self.cost_angle = cost_angle
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"
    def calc_cost_class(self, outputs, targets):
        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()  # [batch_size * num_queries, num_classes]
        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets]).long()

        # Compute the classification cost.
        alpha = 0.25
        gamma = 2.0
        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
        return cost_class
    def calc_cost_3dcenter(self, outputs, targets):
        out_3dcenter = outputs["pred_boxes"][:, :, 0: 2].flatten(0, 1)  # [batch_size * num_queries, 4]
        tgt_3dcenter = torch.cat([v["boxes_3d"][:, 0: 2] for v in targets])

        # Compute the 3dcenter cost between boxes
        cost_3dcenter = torch.cdist(out_3dcenter, tgt_3dcenter, p=1)
        return cost_3dcenter
    def calc_cost_bbox(self, outputs, targets):
        out_2dbbox = outputs["pred_boxes"][:, :, 2: 6].flatten(0, 1)  # [batch_size * num_queries, 4]
        tgt_2dbbox = torch.cat([v["boxes_3d"][:, 2: 6] for v in targets])

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_2dbbox, tgt_2dbbox, p=1)
        return cost_bbox
    def calc_cost_giou(self, outputs, targets):
        # Compute the giou cost betwen boxes
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
        tgt_bbox = torch.cat([v["boxes_3d"] for v in targets])
        cost_giou = -generalized_box_iou(box_cxcylrtb_to_xyxy(out_bbox), box_cxcylrtb_to_xyxy(tgt_bbox))
        return cost_giou
    def calc_cost_depth(self, outputs, targets):
        out_depth = outputs["pred_depth"].flatten(0,1)
        tgt_depth = torch.cat([v["depth"] for v in targets])

        out_depth, out_depth_log_variance = out_depth[:, 0:1], out_depth[:, 1:2]
        depth_cost = 1.1412 * torch.exp(-out_depth_log_variance) * torch.cdist(out_depth, tgt_depth, p = 1) + out_depth_log_variance
        return depth_cost
    def calc_cost_angle(self, outputs, targets):
        out_heading = outputs['pred_angle'].flatten(0,1)
        tgt_heading_cls = torch.cat([v['heading_bin'] for v in targets]).long()
        tgt_heading_res = torch.cat([v['heading_res'] for v in targets])

        out_heading_cls = out_heading[:, 0:12].softmax(-1)
        cls_loss = -(out_heading_cls + 1e-8).log()[:, tgt_heading_cls].squeeze(-1)

        out_heading_res = out_heading[:, 12:24]
        #cls_onehot = torch.zeros(tgt_heading_cls.shape[0], 12).cuda().scatter_(dim=1, index = tgt_heading_cls, value=1)
        out_heading_res = out_heading_res[:,tgt_heading_cls] #B, N, G
        reg_loss = (out_heading_res-tgt_heading_res.unsqueeze(0)).squeeze(-1).abs()

        return cls_loss + reg_loss
    def calc_cost_dim(self, outputs, targets):
        out_dims = outputs['pred_3d_dim'].flatten(0,1)
        tgt_dims = torch.cat([v['size_3d'] for v in targets])

        dimension = tgt_dims # G, 3
        dim_cost = torch.stack((
            torch.cdist(out_dims[0:1],tgt_dims[0:1],p=1),
            torch.cdist(out_dims[1:2],tgt_dims[1:2],p=1),
            torch.cdist(out_dims[2:3],tgt_dims[2:3],p=1)
        ), dim = -1) # Q, G, 3
        dim_cost /= dimension.unsqueeze(0)
        compensation_weight = torch.cdist(out_dims, tgt_dims, p = 1).mean() / dim_cost.mean()
        dim_cost = dim_cost*compensation_weight
        dim_cost = dim_cost.sum(-1)
        return dim_cost
    @torch.no_grad()
    def forward(self, outputs, targets, group_num=11):
        bs, num_queries = outputs["pred_boxes"].shape[:2]

        cost_class = self.calc_cost_class(outputs, targets)

        cost_3dcenter = self.calc_cost_3dcenter(outputs, targets)

        cost_bbox = self.calc_cost_bbox(outputs, targets)

        cost_giou = self.calc_cost_giou(outputs, targets)

        cost_depth = self.calc_cost_depth(outputs, targets)
        
        cost_angle = self.calc_cost_angle(outputs, targets)

        cost_dim = self.calc_cost_dim(outputs, targets)

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_3dcenter * cost_3dcenter + self.cost_class * cost_class + self.cost_giou * cost_giou + self.cost_depth * cost_depth + self.cost_angle*cost_angle + self.cost_dim*cost_dim
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        #indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        indices = []
        g_num_queries = num_queries // group_num
        C_list = C.split(g_num_queries, dim=1)
        for g_i in range(group_num):
            C_g = C_list[g_i]
            indices_g = [linear_sum_assignment(c[i]) for i, c in enumerate(C_g.split(sizes, -1))]
            if g_i == 0:
                indices = indices_g
            else:
                indices = [
                    (np.concatenate([indice1[0], indice2[0] + g_num_queries * g_i]), np.concatenate([indice1[1], indice2[1]]))
                    for indice1, indice2 in zip(indices, indices_g)
                ]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(cfg):
    return HungarianMatcher(
        cost_class=cfg['set_cost_class'],
        cost_bbox=cfg['set_cost_bbox'],
        cost_3dcenter=cfg['set_cost_3dcenter'],
        cost_giou=cfg['set_cost_giou'],
        cost_depth=cfg["set_cost_depth"],
        cost_dim=cfg['set_cost_dim'],
        cost_angle=cfg['set_cost_angle'])
