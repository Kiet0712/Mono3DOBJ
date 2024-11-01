import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_


from .ops.modules import MSDeformAttn
from .encoder import VisualEncoder, VisualEncoderLayer
from .decoder import DepthAwareDecoder, DepthAwareDecoderLayer


class DepthAwareTransformer(nn.Module):
    def __init__(
            self,
            d_model=256,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=1024,
            dropout=0.1,
            activation="relu",
            return_intermediate_dec=False,
            num_feature_levels=4,
            dec_n_points=4,
            enc_n_points=4,
            two_stage=False,
            two_stage_num_proposals=50,
            group_num=11,
            use_dab=False,
            two_stage_dino=False):

        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals
        self.use_dab = use_dab
        self.two_stage_dino = two_stage_dino
        self.group_num = group_num

        encoder_layer = VisualEncoderLayer(
            d_model, dim_feedforward, dropout, activation, num_feature_levels, nhead, enc_n_points)
        self.encoder = VisualEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = DepthAwareDecoderLayer(
            d_model, dim_feedforward, dropout, activation, num_feature_levels, nhead, dec_n_points, group_num=group_num)
        self.decoder = DepthAwareDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec, d_model, use_dab=use_dab, two_stage_dino=two_stage_dino)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        if two_stage:
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(d_model * 2)
        elif two_stage_dino:
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)      
            self.tgt_embed = nn.Embedding(self.two_stage_num_proposals*group_num, d_model)
            nn.init.normal_(self.tgt_embed.weight.data)
            self.two_stage_wh_embedding = None
            self.enc_out_class_embed = None
            self.enc_out_bbox_embed = None
        else:
            if not self.use_dab:
                self.reference_points = nn.Linear(d_model, 2)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if not self.two_stage and not self.use_dab and not self.two_stage_dino:
            xavier_uniform_(self.reference_points.weight.data, gain=1.0)
            constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed)

    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            
            lr = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            tb = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            wh = torch.cat((lr, tb), -1)

            proposal = torch.cat((grid, wh), -1).view(N_, -1, 6)
            proposals.append(proposal)
            _cur += (H_ * W_)
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, masks, pos_embeds, query_embed=None, depth_pos_embed=None, depth_pos_embed_ip=None, attn_mask=None):
        if not self.two_stage_dino:
            assert self.two_stage or query_embed is not None

        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)

            mask = mask.flatten(1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)

        src_flatten = torch.cat(src_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=srcs[0].device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)
        #enc_intermediate_output, enc_intermediate_refpoints = None
        # prepare input for decoder
        bs, _, c = memory.shape
        ####DINO_pos
        if self.two_stage:
            ###share
            output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)
            output_memory = self.enc_output_norm(self.enc_output(output_memory))
            # hack implementation for two-stage Deformable DETR
            enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
            enc_outputs_coord_unact = self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals
            # enc_outputs_class = self.decoder.class_embed(output_memory)
            # enc_outputs_coord_unact = self.decoder.bbox_embed(output_memory) + output_proposals
            topk = self.two_stage_num_proposals
            ####share_end
            topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]
            topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 6))
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid()
            init_reference_out = reference_points
            
            topk_coords_unact_input = torch.cat((topk_coords_unact[..., 0: 2], topk_coords_unact[..., 2: : 2] + topk_coords_unact[..., 3: : 2]), dim=-1)

            pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact_input)))
            query_embed, tgt = torch.split(pos_trans_out, c, dim=2)
        elif self.use_dab:
            reference_points = query_embed[..., self.d_model:].sigmoid() 
            tgt = query_embed[..., :self.d_model]
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
            ##for dn
            init_reference_out = reference_points
        elif self.two_stage_dino:
            output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)
            output_memory = self.enc_output_norm(self.enc_output(output_memory))
            enc_outputs_class_unselected = self.enc_out_class_embed(output_memory)
            enc_outputs_coord_unselected = self.enc_out_bbox_embed(output_memory) + output_proposals # (bs, \sum{hw}, 4) unsigmoid
            if self.training:
                topk = self.two_stage_num_proposals*self.group_num
            else:
                topk = self.two_stage_num_proposals
            topk_proposals = torch.topk(enc_outputs_class_unselected.max(-1)[0], topk, dim=1)[1] # bs, nq

            # gather boxes
            refpoint_embed_undetach = torch.gather(enc_outputs_coord_unselected, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 6)) # unsigmoid
            refpoint_embed_ = refpoint_embed_undetach.detach()
            init_box_proposal = torch.gather(output_proposals, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 6)).sigmoid() # sigmoid
            if self.training:
                tgt_ = self.tgt_embed.weight[:, None, :].repeat(1, bs, 1) # nq, bs, d_model
            else:
                tgt_ = self.tgt_embed.weight[:self.two_stage_num_proposals, None, :].repeat(1, bs, 1) # nq, bs, d_model
            reference_points,tgt=refpoint_embed_,tgt_
            init_reference_out = reference_points
        else:
            query_embed, tgt = torch.split(query_embed, c, dim=1)
            query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
            reference_points = self.reference_points(query_embed).sigmoid()
            init_reference_out = reference_points

        depth_pos_embed = depth_pos_embed.flatten(2).permute(2, 0, 1)
        depth_pos_embed_ip = depth_pos_embed_ip.flatten(2).permute(2, 0, 1)
        mask_depth = masks[1].flatten(1)

        # decoder
        #ipdb.set_trace()
        hs, inter_references, inter_references_dim = self.decoder(
            tgt,                 #.transpose(1,0), for DINO
            reference_points,
            memory,
            spatial_shapes,
            level_start_index,
            valid_ratios,
            query_embed if (not self.use_dab and not self.two_stage_dino) else None, #,INFo
            mask_flatten,
            depth_pos_embed,
            mask_depth, bs=bs, depth_pos_embed_ip=depth_pos_embed_ip, pos_embeds=pos_embeds,attn_mask=attn_mask)

        inter_references_out = inter_references
        inter_references_out_dim = inter_references_dim

        if self.two_stage:
            return hs, init_reference_out, inter_references_out, inter_references_out_dim, enc_outputs_class, enc_outputs_coord_unact
        return hs, init_reference_out, inter_references_out, inter_references_out_dim, None, None


def build_depthaware_transformer(cfg):
    return DepthAwareTransformer(
        d_model=cfg['hidden_dim'],
        dropout=cfg['dropout'],
        activation="relu",
        nhead=cfg['nheads'],
        dim_feedforward=cfg['dim_feedforward'],
        num_encoder_layers=cfg['enc_layers'],
        num_decoder_layers=cfg['dec_layers'],
        return_intermediate_dec=cfg['return_intermediate_dec'],
        num_feature_levels=cfg['num_feature_levels'],
        dec_n_points=cfg['dec_n_points'],
        enc_n_points=cfg['enc_n_points'],
        two_stage=cfg['two_stage'],
        two_stage_num_proposals=cfg['num_queries'],
        use_dab= cfg['use_dab'],
        two_stage_dino = cfg['two_stage_dino'])
