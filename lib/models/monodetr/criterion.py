import torch
import torch.nn.functional as F
from torch import nn
from .depth_predictor.ddn_loss import DDNLoss
from lib.losses.focal_loss import sigmoid_focal_loss
from utils.misc import (accuracy, get_world_size,is_dist_avail_and_initialized)
from utils import box_ops
import math

def class2angle(cls,residual):
    angle_per_class = 2 * math.pi / 12.
    angle_center = cls * angle_per_class
    angle = angle_center + residual
    angle[angle > math.pi] = angle[angle > math.pi] - 2*math.pi
    return angle

def alpha2ry(alpha,u,calibs):
    cu = calibs[:,0,2]
    fu = calibs[:,0,0]
    ry = alpha + torch.atan2(u - cu, fu)
    ry[ry>math.pi] = ry[ry>math.pi] - 2 *math.pi
    ry[ry<-math.pi] = ry[ry<-math.pi] + 2 * math.pi
    return ry

def img_to_rect(uv, depth, calibs):
    cu = calibs[:,0,2]
    cv = calibs[:,1,2]
    fu = calibs[:,0,0]
    fv = calibs[:,1,1]
    tx = calibs[:,0,3] / (-fu)
    ty = calibs[:,1,3] / (-fv)
    u = uv[:,0]
    v = uv[:,1]
    x = ((u - cu) * depth) / fu + tx
    y = ((v - cv) * depth) / fv + ty
    return torch.stack((x,y,depth),dim=-1)

class SetCriterion(nn.Module):
    """ This class computes the loss for MonoDETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, focal_alpha, focal_gamma, losses, query_self_distillation=False, group_num=11):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.ddn_loss = DDNLoss()  # for depth map
        self.group_num = group_num
        self.query_self_distillation = query_self_distillation
    def IoU3D(self, outputs, targets, indices, calibs):
        idx = self._get_src_permutation_idx(indices)

        src_boxes = outputs['pred_boxes'][:, :, :6][idx]
        target_boxes = torch.cat([t['boxes_3d'][:, :6][i] for t, (_, i) in zip(targets, indices)], dim=0)
        
        src_corner_2d = box_ops.box_cxcylrtb_to_xyxy(src_boxes)
        src_xywh_2d = box_ops.box_xyxy_to_cxcywh(src_corner_2d)

        target_corner_2d = box_ops.box_cxcylrtb_to_xyxy(target_boxes)
        target_xywh_2d = box_ops.box_xyxy_to_cxcywh(target_corner_2d)

        src_xs2d = src_xywh_2d[:,0]*torch.tensor([1280], device='cuda')
        target_xs2d = target_xywh_2d[:,0]*torch.tensor([1280], device='cuda')

        src_3dcenter = outputs['pred_boxes'][:, :, 0: 2][idx] * torch.tensor([1280, 384], device='cuda')
        target_3dcenter = torch.cat([t['boxes_3d'][:, 0: 2][i] for t, (_, i) in zip(targets, indices)], dim=0)* torch.tensor([1280, 384], device='cuda')

        src_dims = outputs['pred_3d_dim'][idx]
        target_dims = torch.cat([t['size_3d'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        src_depths = outputs['pred_depth'][idx][:,0]
        target_depths = torch.cat([t['depth'][i] for t, (_, i) in zip(targets, indices)], dim=0).squeeze()
        calibs = calibs.unsqueeze(1).repeat(1,outputs['pred_boxes'].shape[1],1,1)[idx]
        src_center3d = img_to_rect(src_3dcenter,src_depths,calibs)
        target_center3d = img_to_rect(target_3dcenter,target_depths,calibs)

        heading_input = outputs['pred_angle'][idx]
        target_heading_cls = torch.cat([t['heading_bin'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_heading_res = torch.cat([t['heading_res'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        heading_input = heading_input.view(-1, 24)
        heading_target_cls = target_heading_cls.view(-1).long()
        heading_target_res = target_heading_res.view(-1)
        heading_input_cls = torch.argmax(heading_input[:,0:12],dim=-1)
        heading_input_res = heading_input[:,12:24][torch.arange(heading_input_cls.shape[0],device="cuda"),heading_input_cls]
        src_angle = class2angle(heading_input_cls, heading_input_res)
        target_angle = class2angle(heading_target_cls, heading_target_res)
        src_ry = alpha2ry(src_angle,src_xs2d,calibs).unsqueeze(-1)
        target_ry = alpha2ry(target_angle,target_xs2d,calibs).unsqueeze(-1)

        src_boxes3d = torch.cat((src_center3d,src_dims,src_ry),dim=-1)
        target_boxes3d = torch.cat((target_center3d,target_dims,target_ry),dim=-1)
        iou3d = torch.diag(box_ops.boxes_iou3d_gpu(src_boxes3d,target_boxes3d)).detach()
        return iou3d
    
    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, calibs, log=True):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)

        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes_3d'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        ious, _ = box_ops.box_iou(box_ops.box_cxcylrtb_to_xyxy(src_boxes), box_ops.box_cxcylrtb_to_xyxy(target_boxes))
        ious = torch.diag(ious).detach()

        src_logits = outputs['pred_logits']
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o.long()
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()
        weight = self.focal_alpha * pred_score.pow(self.focal_gamma) * (1 - target) + target_score
        
        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_vfl': loss}

    def loss_labels_vfl_3DIoU(self, outputs, targets, indices, num_boxes, calibs, log=True):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)

        # src_boxes = outputs['pred_boxes'][idx]
        # target_boxes = torch.cat([t['boxes_3d'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        # ious, _ = box_ops.box_iou(box_ops.box_cxcylrtb_to_xyxy(src_boxes), box_ops.box_cxcylrtb_to_xyxy(target_boxes))
        # ious = torch.diag(ious).detach()
        ious = self.IoU3D(outputs, targets, indices, calibs)

        src_logits = outputs['pred_logits']
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o.long()
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()
        weight = self.focal_alpha * pred_score.pow(self.focal_gamma) * (1 - target) + target_score
        
        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_vfl': loss}
    
    def loss_labels(self, outputs, targets, indices, num_boxes, calibs, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)

        target_classes[idx] = target_classes_o.squeeze().long()

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=self.focal_gamma) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes, calibs):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_3dcenter(self, outputs, targets, indices, num_boxes, calibs):
        
        idx = self._get_src_permutation_idx(indices)
        src_3dcenter = outputs['pred_boxes'][:, :, 0: 2][idx]
        target_3dcenter = torch.cat([t['boxes_3d'][:, 0: 2][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_3dcenter = F.l1_loss(src_3dcenter, target_3dcenter, reduction='none')
        losses = {}
        losses['loss_center'] = loss_3dcenter.sum() / num_boxes
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes, calibs):
        
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_2dboxes = outputs['pred_boxes'][:, :, 2: 6][idx]
        target_2dboxes = torch.cat([t['boxes_3d'][:, 2: 6][i] for t, (_, i) in zip(targets, indices)], dim=0)

        # l1
        loss_bbox = F.l1_loss(src_2dboxes, target_2dboxes, reduction='none')
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        # giou
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes_3d'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcylrtb_to_xyxy(src_boxes),
            box_ops.box_cxcylrtb_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_depths(self, outputs, targets, indices, num_boxes, calibs):  

        idx = self._get_src_permutation_idx(indices)
   
        src_depths = outputs['pred_depth'][idx]
        target_depths = torch.cat([t['depth'][i] for t, (_, i) in zip(targets, indices)], dim=0).squeeze()

        depth_input, depth_log_variance = src_depths[:, 0], src_depths[:, 1] 
        depth_loss = (1.4142 * torch.exp(-depth_log_variance) * torch.abs(depth_input - target_depths) + depth_log_variance)
        losses = {}
        losses['loss_depth'] = depth_loss.sum() / num_boxes 
        return losses  
    
    def loss_dims(self, outputs, targets, indices, num_boxes, calibs):  

        idx = self._get_src_permutation_idx(indices)
        src_dims = outputs['pred_3d_dim'][idx]
        target_dims = torch.cat([t['size_3d'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        dimension = target_dims.clone().detach()
        dim_loss = torch.abs(src_dims - target_dims)
        dim_loss /= dimension
        with torch.no_grad():
            compensation_weight = F.l1_loss(src_dims, target_dims) / dim_loss.mean()
        dim_loss *= compensation_weight
        losses = {}
        losses['loss_dim'] = dim_loss.sum() / num_boxes
        return losses

    def loss_angles(self, outputs, targets, indices, num_boxes, calibs):  

        idx = self._get_src_permutation_idx(indices)
        heading_input = outputs['pred_angle'][idx]
        target_heading_cls = torch.cat([t['heading_bin'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_heading_res = torch.cat([t['heading_res'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        heading_input = heading_input.view(-1, 24)
        heading_target_cls = target_heading_cls.view(-1).long()
        heading_target_res = target_heading_res.view(-1)

        # classification loss
        heading_input_cls = heading_input[:, 0:12]
        cls_loss = F.cross_entropy(heading_input_cls, heading_target_cls, reduction='none')

        # regression loss
        heading_input_res = heading_input[:, 12:24]
        cls_onehot = torch.zeros(heading_target_cls.shape[0], 12).cuda().scatter_(dim=1, index=heading_target_cls.view(-1, 1), value=1)
        heading_input_res = torch.sum(heading_input_res * cls_onehot, 1)
        reg_loss = F.l1_loss(heading_input_res, heading_target_res, reduction='none')
        
        angle_loss = cls_loss + reg_loss
        losses = {}
        losses['loss_angle'] = angle_loss.sum() / num_boxes 
        return losses

    def loss_depth_map(self, outputs, targets, indices, num_boxes, calibs):
        depth_map_logits = outputs['pred_depth_map_logits']

        num_gt_per_img = [len(t['boxes']) for t in targets]
        gt_boxes2d = torch.cat([t['boxes'] for t in targets], dim=0) * torch.tensor([80, 24, 80, 24], device='cuda')
        gt_boxes2d = box_ops.box_cxcywh_to_xyxy(gt_boxes2d)
        gt_center_depth = torch.cat([t['depth'] for t in targets], dim=0).squeeze(dim=1)
        
        losses = dict()

        losses["loss_depth_map"] = self.ddn_loss(
            depth_map_logits, gt_boxes2d, num_gt_per_img, gt_center_depth)
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, calibs, **kwargs):
        
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'depths': self.loss_depths,
            'dims': self.loss_dims,
            'angles': self.loss_angles,
            'center': self.loss_3dcenter,
            'depth_map': self.loss_depth_map,
            'label_vfl_2d': self.loss_labels_vfl,
            'label_vfl_3d': self.loss_labels_vfl_3DIoU
        }

        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, calibs, **kwargs)

    def forward(self, outputs, targets, calibs, mask_dict=None):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k!='denoising_output'}
        group_num = self.group_num if self.training else 1

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets, group_num=group_num)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets) * group_num
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses

        losses = {}
        for loss in self.losses:
            #ipdb.set_trace()
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, calibs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            iou3d = self.IoU3D(outputs, targets, indices, calibs)
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices_i = self.matcher(aux_outputs, targets, group_num=group_num)
                for loss in self.losses:
                    if loss == 'depth_map':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_i, num_boxes, calibs, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
                if self.query_self_distillation:
                    idx_i = self._get_src_permutation_idx(indices_i)
                    idx = self._get_src_permutation_idx(indices)
                    query_i = aux_outputs['query'][idx_i]
                    query = outputs['query'][idx]
                    loss_query_self_distillation = iou3d.unsqueeze(-1) * F.smooth_l1_loss(query_i, query.detach(), beta=2.0, reduction='none')
                    loss_query_self_distillation = loss_query_self_distillation.sum() / num_boxes
                    l_dict = {'loss_query_self_distillation':loss_query_self_distillation}
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        if 'denoising_output' in outputs:
            losses.update(self.calculate_dn_loss(outputs, targets, calibs, num_boxes))
        return losses
    def calculate_dn_loss(self, outputs, targets, calibs, num_boxes):
        losses = {}
        denoising_output, denoising_groups, single_padding = (
            outputs["denoising_output"],
            outputs["denoising_groups"],
            outputs["max_gt_num_per_image"],
        )
        device = denoising_output["pred_logits"].device
        dn_idx = []
        for i in range(len(targets)):
            if len(targets[i]["labels"]) > 0:
                t = torch.arange(0, len(targets[i]["labels"])).long().to(device)
                t = t.unsqueeze(0).repeat(denoising_groups, 1)
                tgt_idx = t.flatten()
                output_idx = (
                    torch.tensor(range(denoising_groups)).to(device) * single_padding
                ).long().unsqueeze(1) + t
                output_idx = output_idx.flatten()
            else:
                output_idx = tgt_idx = torch.tensor([]).long().to(device)

            dn_idx.append((output_idx, tgt_idx))
        for loss in self.losses:
            if loss == 'depth_map':
                # Intermediate masks losses are too costly to compute, we ignore them.
                continue
            kwargs = {}
            if loss == 'labels':
                # Logging is enabled only for the last layer
                kwargs = {'log': False}
            l_dict = self.get_loss(loss, denoising_output, targets, dn_idx, num_boxes* denoising_groups // 11, calibs, **kwargs)
            l_dict = {k + f'_dn': v for k, v in l_dict.items()}
            losses.update(l_dict)
        iou3d_dn = self.IoU3D(denoising_output, targets, dn_idx, calibs)
        for i, aux_outputs in enumerate(denoising_output['aux_outputs']):
            for loss in self.losses:
                if loss == 'depth_map':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs = {'log': False}
                l_dict = self.get_loss(loss, aux_outputs, targets, dn_idx, num_boxes* denoising_groups // 11, calibs, **kwargs)
                l_dict = {k + f'_dn_{i}': v for k, v in l_dict.items()}
                losses.update(l_dict)
            if self.query_self_distillation:
                idx = self._get_src_permutation_idx(dn_idx)
                query_i = aux_outputs['query'][idx]
                query = denoising_output['query'][idx]
                loss_query_self_distillation = iou3d_dn.unsqueeze(-1) * F.smooth_l1_loss(query_i, query.detach(), beta=2.0, reduction='none')
                loss_query_self_distillation = loss_query_self_distillation.sum() / (num_boxes * denoising_groups // 11)
                l_dict = {'loss_query_self_distillation':loss_query_self_distillation}
                l_dict = {k + f'_dn_{i}': v for k, v in l_dict.items()}
                losses.update(l_dict)
        return losses
