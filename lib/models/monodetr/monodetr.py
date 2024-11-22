"""
MonoDETR: Depth-aware Transformer for Monocular 3D Object Detection
"""
import torch
import torch.nn.functional as F
from torch import nn
import math
import copy

from utils.misc import (NestedTensor, inverse_sigmoid, get_clones, MLP)

from .backbone import build_backbone
from .matcher import build_matcher
from .depthaware_transformer import build_depthaware_transformer
from .depth_predictor import DepthPredictor
from .criterion import SetCriterion
from .denoising import build_generate_DN_queries


class MonoDETR(nn.Module):
    """ This is the MonoDETR module that performs monocualr 3D object detection """
    def __init__(self, backbone, depthaware_transformer, depth_predictor, num_classes, num_queries, num_feature_levels,
                 aux_loss=True, query_self_distillation=False, generate_DN_queries = None, group_num=11):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            depthaware_transformer: depth-aware transformer architecture. See depth_aware_transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For KITTI, we recommend 50 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
 
        self.num_queries = num_queries
        self.depthaware_transformer = depthaware_transformer
        self.depth_predictor = depth_predictor
        hidden_dim = depthaware_transformer.d_model
        self.hidden_dim = hidden_dim
        self.num_feature_levels = num_feature_levels
        # prediction heads
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value

        self.bbox_embed = MLP(hidden_dim, hidden_dim, 6, 3)
        self.dim_embed_3d = MLP(hidden_dim, hidden_dim, 3, 2)
        self.angle_embed = MLP(hidden_dim, hidden_dim, 24, 2)
        self.depth_embed = MLP(hidden_dim, hidden_dim, 2, 2)  # depth and deviation

        self.query_embed = nn.Embedding(num_queries * group_num, hidden_dim*2)

        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.num_classes = num_classes

        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
        num_pred = depthaware_transformer.decoder.num_layers
        self.class_embed = get_clones(self.class_embed, num_pred)
        self.bbox_embed = get_clones(self.bbox_embed, num_pred)
        nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
        # hack implementation for iterative bounding box refinement
        self.depthaware_transformer.decoder.bbox_embed = self.bbox_embed
        self.dim_embed_3d = get_clones(self.dim_embed_3d, num_pred)
        self.depthaware_transformer.decoder.dim_embed = self.dim_embed_3d  
        self.angle_embed = get_clones(self.angle_embed, num_pred)
        self.depth_embed = get_clones(self.depth_embed, num_pred)
        self.depthaware_transformer.decoder.depth_embed = self.depth_embed
        self.query_self_distillation = query_self_distillation
        if query_self_distillation:
            self.proj_self_distillation = nn.Linear(hidden_dim, hidden_dim)
        self.generate_DN_queries = generate_DN_queries
        self.group_num = group_num
    def forward(self, images, calibs, target, img_sizes):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        """

        if self.training and self.generate_DN_queries is not None:
            gt_labels_list = [t["labels"].long() for t in target]
            gt_boxes_list = [t["boxes_3d"] for t in target]
            gt_depth_list = [t["depth"] for t in target]
            gt_dim_list = [t["size_3d"] for t in target]
            gt_heading_bin_list = [t["heading_bin"] for t in target]

        features, pos = self.backbone(images)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None

        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = torch.zeros(src.shape[0], src.shape[2], src.shape[3]).to(torch.bool).to(src.device)
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)
        attn_mask = None
        noised_label_queries = None
        noised_box_queries = None
        noised_query_embed = None
        if self.training:
            query_embeds = self.query_embed.weight
            if self.generate_DN_queries is not None:
                (
                    noised_label_queries,
                    noised_box_queries,
                    attn_mask,
                    denoising_groups,
                    max_gt_num_per_image,
                ) = self.generate_DN_queries(gt_labels_list, gt_boxes_list, gt_depth_list, gt_dim_list, gt_heading_bin_list)
                noised_query_embed, noised_label_queries = torch.split(noised_label_queries, self.hidden_dim, dim=-1)

        else:
            # only use one group in inference
            query_embeds = self.query_embed.weight[:self.num_queries]

        pred_depth_map_logits, depth_pos_embed, weighted_depth = self.depth_predictor(srcs, masks[1], pos[1])
        
        hs, init_reference, inter_references, inter_references_dim, inter_references_depths = self.depthaware_transformer(
            srcs, masks, pos, query_embeds, noised_label_queries, noised_box_queries, noised_query_embed, depth_pos_embed, attn_mask)

        outputs_coords = []
        outputs_classes = []
        outputs_3d_dims = []
        outputs_depths = []
        outputs_angles = []

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)

            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 6:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference

          
            # 3d center + 2d box
            outputs_coord = tmp.sigmoid()
            outputs_coords.append(outputs_coord)

            # classes
            outputs_class = self.class_embed[lvl](hs[lvl])
            outputs_classes.append(outputs_class)

            # 3D sizes
            size3d = inter_references_dim[lvl]
            outputs_3d_dims.append(size3d)

            # depth_geo
            box2d_height_norm = outputs_coord[:, :, 4] + outputs_coord[:, :, 5]
            box2d_height = torch.clamp(box2d_height_norm * img_sizes[:, 1: 2], min=1.0)
            depth_geo = size3d[:, :, 0] / box2d_height * calibs[:, 0, 0].unsqueeze(1)

            # depth_reg
            depth_reg = inter_references_depths[lvl]

            # depth_map
            outputs_center3d = ((outputs_coord[..., :2] - 0.5) * 2).unsqueeze(2).detach()
            depth_map = F.grid_sample(
                weighted_depth.unsqueeze(1),
                outputs_center3d,
                mode='bilinear',
                align_corners=True).squeeze(1)

            # depth average + sigma
            depth_ave = torch.cat([((1. / (depth_reg[:, :, 0: 1].sigmoid() + 1e-6) - 1.) + depth_geo.unsqueeze(-1) + depth_map) / 3,
                                    depth_reg[:, :, 1: 2]], -1)
            outputs_depths.append(depth_ave)

            # angles
            outputs_angle = self.angle_embed[lvl](hs[lvl])
            outputs_angles.append(outputs_angle)

        outputs_coord = torch.stack(outputs_coords)
        outputs_class = torch.stack(outputs_classes)
        outputs_3d_dim = torch.stack(outputs_3d_dims)
        outputs_depth = torch.stack(outputs_depths)
        outputs_angle = torch.stack(outputs_angles)

        if self.generate_DN_queries is not None and self.training:
            out = {
                "denoising_groups": torch.tensor(denoising_groups).cuda(),
                "max_gt_num_per_image": torch.tensor(max_gt_num_per_image).cuda(),
            }

            padding_size = out["max_gt_num_per_image"] * out["denoising_groups"]

            outputs_known_class = outputs_class[:, :, :padding_size, :]
            outputs_known_coord = outputs_coord[:, :, :padding_size, :]
            outputs_known_3d_dim = outputs_3d_dim[:, :, :padding_size, :]
            outputs_known_depth = outputs_depth[:, :, :padding_size, :]
            outputs_known_angle = outputs_angle[:, :, :padding_size, :]
            if self.query_self_distillation:
                if self.training:
                    known_hs = hs[:, :, :padding_size, :]
            output_denoising = {'pred_logits': outputs_known_class[-1], 'pred_boxes': outputs_known_coord[-1]}
            output_denoising['pred_3d_dim'] = outputs_known_3d_dim[-1]
            output_denoising['pred_depth'] = outputs_known_depth[-1]
            output_denoising['pred_angle'] = outputs_known_angle[-1]

            if self.query_self_distillation:
                if self.training:
                    output_denoising['query'] = known_hs[-1]

            if self.aux_loss:
                if not self.query_self_distillation:
                    output_denoising['aux_outputs'] = self._set_aux_loss(
                        outputs_known_class, outputs_known_coord, outputs_known_3d_dim, outputs_known_angle, outputs_known_depth)
                else:
                    if self.training:
                        output_denoising['aux_outputs'] = self._set_aux_loss(
                            outputs_known_class, outputs_known_coord, outputs_known_3d_dim, outputs_known_angle, outputs_known_depth, known_hs)
                    else:
                        output_denoising['aux_outputs'] = self._set_aux_loss(
                            outputs_known_class, outputs_known_coord, outputs_known_3d_dim, outputs_known_angle, outputs_known_depth)
            
            outputs_class = outputs_class[:, :, padding_size:, :]
            outputs_coord = outputs_coord[:, :, padding_size:, :]
            outputs_3d_dim = outputs_3d_dim[:, :, padding_size:, :]
            outputs_depth = outputs_depth[:, :, padding_size:, :]
            outputs_angle = outputs_angle[:, :, padding_size:, :]
            if self.query_self_distillation:
                if self.training:
                    hs = hs[:, :, padding_size:, :]

        if self.generate_DN_queries is None:
            out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        else:
            if self.training:
                out['pred_logits'] = outputs_class[-1]
                out['pred_boxes'] = outputs_coord[-1]
            else:
                out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.generate_DN_queries is not None and self.training:
            out['denoising_output'] = output_denoising
        out['pred_3d_dim'] = outputs_3d_dim[-1]
        out['pred_depth'] = outputs_depth[-1]
        out['pred_angle'] = outputs_angle[-1]
        out['pred_depth_map_logits'] = pred_depth_map_logits
        if self.query_self_distillation:
            if self.training:
                out['query'] = hs[-1]
        if self.aux_loss:
            if not self.query_self_distillation:
                out['aux_outputs'] = self._set_aux_loss(
                        outputs_class, outputs_coord, outputs_3d_dim, outputs_angle, outputs_depth)
            else:
                if self.training:
                    out['aux_outputs'] = self._set_aux_loss(
                        outputs_class, outputs_coord, outputs_3d_dim, outputs_angle, outputs_depth, hs)
                else:
                    out['aux_outputs'] = self._set_aux_loss(
                        outputs_class, outputs_coord, outputs_3d_dim, outputs_angle, outputs_depth)
        return out #, mask_dict

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_3d_dim, outputs_angle, outputs_depth, hs=None):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if hs==None:
            return [{'pred_logits': a, 'pred_boxes': b, 
                    'pred_3d_dim': c, 'pred_angle': d, 'pred_depth': e}
                    for a, b, c, d, e in zip(outputs_class[:-1], outputs_coord[:-1],
                                            outputs_3d_dim[:-1], outputs_angle[:-1], outputs_depth[:-1])]
        else:
            return [{'pred_logits': a, 'pred_boxes': b, 
                    'pred_3d_dim': c, 'pred_angle': d, 'pred_depth': e, 'query': self.proj_self_distillation(f)}
                    for a, b, c, d, e, f in zip(outputs_class[:-1], outputs_coord[:-1],
                                            outputs_3d_dim[:-1], outputs_angle[:-1], outputs_depth[:-1], hs[:-1])]
                



def build(cfg):
    # backbone
    backbone = build_backbone(cfg)

    # detr
    depthaware_transformer = build_depthaware_transformer(cfg)

    # depth prediction module
    depth_predictor = DepthPredictor(cfg)
    generate_DN_queries = None
    if cfg['use_dn']:
        generate_DN_queries = build_generate_DN_queries(cfg)
    model = MonoDETR(
        backbone,
        depthaware_transformer,
        depth_predictor,
        num_classes=cfg['num_classes'],
        num_queries=cfg['num_queries'],
        aux_loss=cfg['aux_loss'],
        num_feature_levels=cfg['num_feature_levels'],
        query_self_distillation=cfg['query_self_distillation'],
        generate_DN_queries = generate_DN_queries)

    # matcher
    matcher = build_matcher(cfg)

    # loss
    weight_dict = {'loss_ce': cfg['cls_loss_coef'], 'loss_bbox': cfg['bbox_loss_coef']}
    weight_dict['loss_giou'] = cfg['giou_loss_coef']
    weight_dict['loss_dim'] = cfg['dim_loss_coef']
    weight_dict['loss_angle'] = cfg['angle_loss_coef']
    weight_dict['loss_depth'] = cfg['depth_loss_coef']
    weight_dict['loss_center'] = cfg['3dcenter_loss_coef']
    weight_dict['loss_depth_map'] = cfg['depth_map_loss_coef']
    weight_dict['loss_query_self_distillation'] = cfg['query_self_distillation_loss_coef']
    weight_dict['loss_vfl'] = cfg['cls_loss_coef']

    weight_dict['loss_ce_dn']= cfg['cls_loss_coef']
    weight_dict['loss_bbox_dn'] = cfg['bbox_loss_coef']
    weight_dict['loss_giou_dn'] = cfg['giou_loss_coef']
    weight_dict['loss_angle_dn'] = cfg['angle_loss_coef']
    weight_dict['loss_center_dn'] = cfg['3dcenter_loss_coef']
    weight_dict['loss_dim_dn'] = cfg['dim_loss_coef']
    weight_dict['loss_depth_dn'] = cfg['depth_loss_coef']
    weight_dict['loss_query_self_distillation_dn'] = cfg['query_self_distillation_loss_coef']
    # TODO this is a hack
    if cfg['aux_loss']:
        aux_weight_dict = {}
        for i in range(cfg['dec_layers'] - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality', 'depths', 'dims', 'angles', 'center', 'depth_map']
    if cfg['use_vfl']:
        if cfg['use_vfl_with_3dIoU']:
            losses[0] = 'label_vfl_3d'
        else:
            losses[0] = 'label_vfl_2d'
    criterion = SetCriterion(
        cfg['num_classes'],
        matcher=matcher,
        weight_dict=weight_dict,
        focal_alpha=cfg['focal_alpha'],
        focal_gamma=cfg['focal_gamma'],
        losses=losses,
        query_self_distillation=cfg['query_self_distillation'])

    device = torch.device(cfg['device'])
    criterion.to(device)
    
    return model, criterion
