import torch
import torch.nn as nn
from utils.misc import inverse_sigmoid, MLP
import functools
import math

def apply_label_noise(
    labels: torch.Tensor,
    label_noise_prob: float = 0.2,
    num_classes: int = 80,
):
    """
    Args:
        labels (torch.Tensor): Classification labels with ``(num_labels, )``.
        label_noise_prob (float): The probability of the label being noised. Default: 0.2.
        num_classes (int): Number of total categories.

    Returns:
        torch.Tensor: The noised labels the same shape as ``labels``.
    """
    if label_noise_prob > 0:
        p = torch.rand_like(labels.float())
        noised_index = torch.nonzero(p < label_noise_prob).view(-1)
        new_labels = torch.randint_like(noised_index, 0, num_classes)
        noised_labels = labels.scatter_(0, noised_index, new_labels)
        return noised_labels
    else:
        return labels
def apply_box_3d_noise(
    boxes: torch.Tensor,
    box_noise_scale: float = 0.4,
):
    """
    Args:
        boxes (torch.Tensor): Bounding boxes in format ``(x_c, y_c, w, h)`` with
            shape ``(num_boxes, 4)``
        box_noise_scale (float): Scaling factor for box noising. Default: 0.4.
    """
    if box_noise_scale > 0:
        diff = torch.zeros_like(boxes)
        diff[:, 0] = boxes[:, 3] / 2
        diff[:, 1:] = boxes[:, 1:]
        boxes += torch.mul((torch.rand_like(boxes) * 2 - 1.0), diff).cuda() * box_noise_scale
        boxes[:, 0] = boxes[:, 0].clamp(min=1e-3, max=65.0)
    return boxes

def apply_box_noise(
    boxes: torch.Tensor,
    box_noise_scale: float = 0.4,
):
    """
    Args:
        boxes (torch.Tensor): Bounding boxes in format ``(x_c, y_c, w, h)`` with
            shape ``(num_boxes, 4)``
        box_noise_scale (float): Scaling factor for box noising. Default: 0.4.
    """
    if box_noise_scale > 0:
        diff = torch.zeros_like(boxes)
        diff[:, 0] = (boxes[:, 2] + boxes[:, 3]) / 2
        diff[:, 1] = (boxes[:, 4] + boxes[:, 5]) / 2
        diff[:, 2:] = boxes[:, 2:]
        boxes += torch.mul((torch.rand_like(boxes) * 2 - 1.0), diff).cuda() * box_noise_scale
        boxes = boxes.clamp(min=0.0, max=1.0)
    return boxes

@functools.lru_cache  # use lru_cache to avoid redundant calculation for dim_t
def get_dim_t(num_pos_feats: int, temperature: int, device: torch.device):
    dim_t = torch.arange(num_pos_feats // 2, dtype=torch.float32, device=device)
    dim_t = temperature**(dim_t * 2 / num_pos_feats)
    return dim_t  # (0, 2, 4, ..., ⌊n/2⌋*2)


def get_sine_pos_embed(
    pos_tensor,
    num_pos_feats: int = 128,
    temperature: int = 10000,
    scale: float = 2 * math.pi
):
    """Generate sine position embedding for a position tensor

    :param pos_tensor: shape as (..., 2*n).
    :param num_pos_feats: projected shape for each float in the tensor, defaults to 128
    :param temperature: the temperature used for scaling the position embedding, defaults to 10000
    :param exchange_xy: exchange pos x and pos. For example,
        input tensor is [x, y], the results will be [pos(y), pos(x)], defaults to True
    :return: position embedding with shape (None, n * num_pos_feats)
    """
    dim_t = get_dim_t(num_pos_feats, temperature, pos_tensor.device)
    pos_res = pos_tensor.unsqueeze(-1) * scale / dim_t
    pos_res = torch.stack((pos_res.sin(), pos_res.cos()), dim=-1).flatten(-2)
    pos_res = pos_res.flatten(-2)
    return pos_res
class GenerateDNQueries(nn.Module):
    def __init__(
        self,
        num_queries: int = 300,
        num_classes: int = 80,
        label_embed_dim: int = 256,
        denoising_groups: int = 5,
        label_noise_prob: float = 0.2,
        box_noise_scale: float = 0.4,
        angle_noise_prob: float = 0.2,
        boxes3d_noise_scale: float = 0.4,
        with_indicator: bool = False,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.label_embed_dim = label_embed_dim
        self.denoising_groups = denoising_groups
        self.label_noise_prob = label_noise_prob
        self.box_noise_scale = box_noise_scale
        self.angle_noise_prob = angle_noise_prob
        self.boxes3d_noise_scale = boxes3d_noise_scale
        self.with_indicator = with_indicator

        if with_indicator:
            self.label_encoder = nn.Embedding(num_classes, label_embed_dim - 1)
        else:
            self.label_encoder = nn.Embedding(num_classes, label_embed_dim)
        self.bboxes_3d_encoder = MLP(label_embed_dim*6, label_embed_dim*2, label_embed_dim*2, 2)
        self.heading_bin_encoder = nn.Embedding(12, label_embed_dim)

    def generate_query_masks(self, max_gt_num_per_image, device):
        noised_query_nums = max_gt_num_per_image * self.denoising_groups // 11
        tgt_size = noised_query_nums + self.num_queries
        attn_mask = torch.ones(tgt_size, tgt_size).to(device) < 0
        # match query cannot see the reconstruct
        attn_mask[noised_query_nums:, :noised_query_nums] = True
        for i in range(self.denoising_groups//11):
            if i == 0:
                attn_mask[
                    max_gt_num_per_image * i : max_gt_num_per_image * (i + 1),
                    max_gt_num_per_image * (i + 1) : noised_query_nums,
                ] = True
            if i == self.denoising_groups - 1:
                attn_mask[
                    max_gt_num_per_image * i : max_gt_num_per_image * (i + 1),
                    : max_gt_num_per_image * i,
                ] = True
            else:
                attn_mask[
                    max_gt_num_per_image * i : max_gt_num_per_image * (i + 1),
                    max_gt_num_per_image * (i + 1) : noised_query_nums,
                ] = True
                attn_mask[
                    max_gt_num_per_image * i : max_gt_num_per_image * (i + 1),
                    : max_gt_num_per_image * i,
                ] = True
        return attn_mask


    def forward(
        self,
        gt_labels_list,
        gt_boxes_list,
        gt_depth_list,
        gt_dim_list,
        gt_heading_bin_list
    ):

        # concat ground truth labels and boxes in one batch
        # e.g. [tensor([0, 1, 2]), tensor([2, 3, 4])] -> tensor([0, 1, 2, 2, 3, 4])
        gt_labels = torch.cat(gt_labels_list)
        gt_boxes = torch.cat(gt_boxes_list)
        gt_depths = torch.cat(gt_depth_list)
        gt_dims = torch.cat(gt_dim_list)
        gt_heading_bins = torch.cat(gt_heading_bin_list)

        # For efficient denoising, repeat the original ground truth labels and boxes to
        # create more training denoising samples.
        # e.g. tensor([0, 1, 2, 2, 3, 4]) -> tensor([0, 1, 2, 2, 3, 4, 0, 1, 2, 2, 3, 4]) if group = 2.
        gt_labels = gt_labels.repeat(self.denoising_groups, 1).flatten()
        gt_boxes = gt_boxes.repeat(self.denoising_groups, 1)
        gt_depths = gt_depths.repeat(self.denoising_groups, 1)
        gt_dims = gt_dims.repeat(self.denoising_groups, 1)
        gt_heading_bins = gt_heading_bins.repeat(self.denoising_groups, 1).flatten()

        device = gt_labels.device
        assert len(gt_labels_list) == len(gt_boxes_list)

        batch_size = len(gt_labels_list)

        # the number of ground truth per image in one batch
        # e.g. [tensor([0, 1]), tensor([2, 3, 4])] -> gt_nums_per_image: [2, 3]
        # means there are 2 instances in the first image and 3 instances in the second image
        gt_nums_per_image = [x.numel() for x in gt_labels_list]

        # Add noise on labels and boxes
        noised_labels = apply_label_noise(gt_labels, self.label_noise_prob, self.num_classes)
        noised_boxes = apply_box_noise(gt_boxes, self.box_noise_scale)
        noised_depth = gt_depths
        noised_dim = gt_dims
        noised_boxes_3d = apply_box_3d_noise(torch.cat((noised_depth, noised_dim),dim=-1), self.boxes3d_noise_scale)
        noised_heading_bin = apply_label_noise(gt_heading_bins, self.angle_noise_prob, 12)
        #noised_boxes = inverse_sigmoid(noised_boxes)

        # encoding labels
        labels_embedding = self.label_encoder(noised_labels)
        depth_and_size_3d_embedding = get_sine_pos_embed(noised_boxes_3d, self.label_embed_dim)
        heading_bin_embedding = self.heading_bin_encoder(noised_heading_bin)
        query_num = labels_embedding.shape[0]

        # add indicator to label encoding if with_indicator == True
        if self.with_indicator:
            labels_embedding = torch.cat([labels_embedding, torch.ones([query_num, 1]).to(device)], 1)
        label_embedding = self.bboxes_3d_encoder(torch.cat((labels_embedding, depth_and_size_3d_embedding, heading_bin_embedding),dim=-1))
        # calculate the max number of ground truth in one image inside the batch.
        # e.g. gt_nums_per_image = [2, 3] which means
        # the first image has 2 instances and the second image has 3 instances
        # then the max_gt_num_per_image should be 3.
        max_gt_num_per_image = max(gt_nums_per_image)

        # the total denoising queries is depended on denoising groups and max number of instances.
        noised_query_nums = max_gt_num_per_image * self.denoising_groups

        # initialize the generated noised queries to zero.
        # And the zero initialized queries will be assigned with noised embeddings later.
        noised_label_queries = (
            torch.zeros(noised_query_nums, self.label_embed_dim*2).to(device).repeat(batch_size, 1, 1)
        )
        noised_box_queries = torch.zeros(noised_query_nums, 6).to(device).repeat(batch_size, 1, 1)

        # batch index per image: [0, 1, 2, 3] for batch_size == 4
        batch_idx = torch.arange(0, batch_size)

        # e.g. gt_nums_per_image = [2, 3]
        # batch_idx = [0, 1]
        # then the "batch_idx_per_instance" equals to [0, 0, 1, 1, 1]
        # which indicates which image the instance belongs to.
        # cuz the instances has been flattened before.
        batch_idx_per_instance = torch.repeat_interleave(
            batch_idx, torch.tensor(gt_nums_per_image).long()
        )

        # indicate which image the noised labels belong to. For example:
        # noised label: tensor([0, 1, 2, 2, 3, 4, 0, 1, 2, 2, 3, 4])
        # batch_idx_per_group: tensor([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1])
        # which means the first label "tensor([0])"" belongs to "image_0".
        batch_idx_per_group = batch_idx_per_instance.repeat(self.denoising_groups, 1).flatten()

        # Cuz there might be different numbers of ground truth in each image of the same batch.
        # So there might be some padding part in noising queries.
        # Here we calculate the indexes for the valid queries and
        # fill them with the noised embeddings.
        # And leave the padding part to zeros.
        if len(gt_nums_per_image):
            valid_index_per_group = torch.cat(
                [torch.tensor(list(range(num))) for num in gt_nums_per_image]
            )
            valid_index_per_group = torch.cat(
                [
                    valid_index_per_group + max_gt_num_per_image * i
                    for i in range(self.denoising_groups)
                ]
            ).long()
        if len(batch_idx_per_group):
            noised_label_queries[(batch_idx_per_group, valid_index_per_group)] = label_embedding
            noised_box_queries[(batch_idx_per_group, valid_index_per_group)] = noised_boxes
        # generate attention masks for transformer layers
        attn_mask = self.generate_query_masks(max_gt_num_per_image, device)

        return (
            noised_label_queries,
            noised_box_queries,
            attn_mask,
            self.denoising_groups,
            max_gt_num_per_image,
        )

def build_generate_DN_queries(cfg):
    return GenerateDNQueries(
        num_queries = cfg['num_queries'],
        num_classes = cfg['num_classes'] + 1,
        label_embed_dim = cfg['hidden_dim'],
        denoising_groups = 55,
        label_noise_prob = cfg['label_noise_prob'],
        box_noise_scale = cfg['box_noise_scale'],
        angle_noise_prob = cfg['angle_noise_prob'],
        boxes3d_noise_scale = cfg['boxes3d_noise_scale'],
        with_indicator= cfg['with_indicators']
    )