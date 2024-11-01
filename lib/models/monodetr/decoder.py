from .ops.modules import MSDeformAttn
import math
import torch
import torch.nn.functional as F
from torch import nn
from utils.misc import inverse_sigmoid, get_clones, get_activation_fn, MLP


def gen_sineembed_for_position(pos_tensor):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / 128)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    elif pos_tensor.size(-1) == 6:
        for i in range(2, 6):         # Compute sine embeds for l, r, t, b
            embed = pos_tensor[:, :, i] * scale
            pos_embed = embed[:, :, None] / dim_t
            pos_embed = torch.stack((pos_embed[:, :, 0::2].sin(), pos_embed[:, :, 1::2].cos()), dim=3).flatten(2)
            if i == 2:  # Initialize pos for the case of size(-1)=6
                pos = pos_embed
            else:       # Concatenate embeds for l, r, t, b
                pos = torch.cat((pos, pos_embed), dim=2)
        pos = torch.cat((pos_y, pos_x, pos), dim=2) 
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos


class DepthAwareDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4, group_num=1):
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # depth cross attention
        self.cross_attn_depth = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout_depth = nn.Dropout(dropout)
        self.norm_depth = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self.group_num = group_num
        
        # Decoder Self-Attention
        self.sa_qcontent_proj = nn.Linear(d_model, d_model)
        self.sa_qpos_proj = nn.Linear(d_model, d_model)
        self.sa_kcontent_proj = nn.Linear(d_model, d_model)
        self.sa_kpos_proj = nn.Linear(d_model, d_model)
        self.sa_v_proj = nn.Linear(d_model, d_model)
        self.nhead = n_heads
        

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self,
                tgt,
                query_pos,
                reference_points,
                src,
                src_spatial_shapes,
                level_start_index,
                src_padding_mask,
                depth_pos_embed,
                mask_depth, 
                bs,
                query_sine_embed=None,
                is_first=None,
                depth_pos_embed_ip=None,
                pos_embeds=None,
                self_attn_mask=None,
                query_pos_un=None):

        # depth cross attention
        tgt2 = self.cross_attn_depth(tgt.transpose(0, 1),
                                     depth_pos_embed,
                                     depth_pos_embed,
                                     key_padding_mask=mask_depth)[0].transpose(0, 1)
       
        tgt = tgt + self.dropout_depth(tgt2)
        tgt = self.norm_depth(tgt)

        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        
        q_content = self.sa_qcontent_proj(q)
        q_pos = self.sa_qpos_proj(q)
        k_content = self.sa_kcontent_proj(k)
        k_pos = self.sa_kpos_proj(k)
        v = self.sa_v_proj(tgt)
        q = q_content + q_pos
        k = k_content + k_pos
        
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = tgt.transpose(0, 1)
        num_queries = q.shape[0]
       
        if self.training:
            num_noise = num_queries-self.group_num * 50
            num_queries = self.group_num * 50
            q_noise = q[:num_noise].repeat(1,self.group_num, 1)
            k_noise = k[:num_noise].repeat(1,self.group_num, 1)
            v_noise = v[:num_noise].repeat(1,self.group_num, 1)
            q = q[num_noise:]
            k = k[num_noise:]
            v = v[num_noise:]
            q = torch.cat(q.split(num_queries // self.group_num, dim=0), dim=1)
            k = torch.cat(k.split(num_queries // self.group_num, dim=0), dim=1)
            v = torch.cat(v.split(num_queries // self.group_num, dim=0), dim=1)
            q = torch.cat([q_noise,q], dim=0)
            k = torch.cat([k_noise,k], dim=0)
            v = torch.cat([v_noise,v], dim=0)
        
        tgt2 = self.self_attn(q, k, v)[0]
        if self.training:
            tgt2 = torch.cat(tgt2.split(bs, dim=1), dim=0).transpose(0, 1)
            
        else:
            tgt2 = tgt2.transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
      
        
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               src, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


class DepthAwareDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False, d_model=None, use_dab=False, two_stage_dino=False):
        super().__init__()
        self.layers = get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.dim_embed = None
        self.class_embed = None
        self.use_dab=use_dab
        self.two_stgae_dino = two_stage_dino
        if use_dab:
            self.query_scale = MLP(d_model, d_model, d_model, 2)
            self.query_scale_bbox = MLP(d_model, 2, 2, 2)
            self.ref_point_head = MLP(3 * d_model, d_model, d_model, 2)
        elif two_stage_dino:
            self.ref_point_head = MLP(3 * d_model, d_model, d_model, 2)
            #self.query_scale = None
            self.query_scale = MLP(d_model, d_model, d_model, 2)
            self.query_pos_sine_scale = None
            self.ref_anchor_head = None
        else:
            self.query_scale = MLP(d_model, d_model, d_model, 2)
            self.ref_point_head = MLP(d_model, d_model, 2, 2)
        #conditional
        # for layer_id in range(num_layers - 1):
        #     self.layers[layer_id + 1].ca_qpos_proj = None
        ###

    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None, depth_pos_embed=None, mask_depth=None, bs=None, depth_pos_embed_ip=None, pos_embeds=None, attn_mask=None):
        output = tgt

        intermediate = []
        intermediate_reference_points = []
        intermediate_reference_dims = []
        bs = src.shape[0]
        ###for dn
        if self.use_dab:
            reference_points = reference_points[None].repeat(bs, 1, 1)
        elif self.two_stgae_dino:
            reference_points = reference_points.sigmoid()

        
        for lid, layer in enumerate(self.layers):
            
            if reference_points.shape[-1] == 6:
                
                reference_points_input = reference_points[:, :, None] * torch.cat([src_valid_ratios, src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]
            if self.two_stgae_dino:
                
                query_sine_embed = gen_sineembed_for_position(reference_points_input[:, :, 0, :])
                raw_query_pos = self.ref_point_head(query_sine_embed) # nq, bs, 256
                
                pos_scale = self.query_scale(output) if lid != 0 else 1
                query_pos = pos_scale * raw_query_pos
            ###conditional
            #ipdb.set_trace()
            query_pos_un=None
            if self.use_dab:
                
                query_sine_embed = gen_sineembed_for_position(reference_points_input[:, :, 0, :]) # bs, nq, 256*2 
                raw_query_pos = self.ref_point_head(query_sine_embed) # bs, nq, 256
                pos_scale = self.query_scale(output) if lid != 0 else 1
                #pos_scale  = 1
                query_pos = pos_scale * raw_query_pos
                
            output = layer(output,
                           query_pos,
                           reference_points_input,
                           src,
                           src_spatial_shapes,
                           src_level_start_index,
                           src_padding_mask,
                           depth_pos_embed,
                           mask_depth, bs,query_sine_embed=None,
                           is_first=(lid == 0), depth_pos_embed_ip=depth_pos_embed_ip, pos_embeds=pos_embeds, self_attn_mask=attn_mask,query_pos_un=query_pos_un)

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output)
                if reference_points.shape[-1] == 6:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    # print(reference_points.shape)
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.dim_embed is not None:
                reference_dims = self.dim_embed[lid](output)

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)
                intermediate_reference_dims.append(reference_dims)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points), torch.stack(intermediate_reference_dims)

        return output, reference_points