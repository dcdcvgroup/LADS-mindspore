import src.resnet as resnet
import math
import os
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
import mindspore as ms
from mindspore.ops import functional as F

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Utilities for bounding box manipulation and GIoU.
"""


def box_cxcywh_to_xyxy(x):  # N,4
    x_c, y_c, w, h = ops.unstack(x, -1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return ops.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = ops.unstack(x, -1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return ops.stack(b, dim=-1)


def box_area(boxes: Tensor) -> Tensor:
    """calculate the area of the bboxes(x1,y1,x2,y2)"""

    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


# modified from torchvision to also return the union
# def box_iou(boxes1, boxes2):
#     area1 = box_area(boxes1)
#     area2 = box_area(boxes2)

#     lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
#     rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

#     wh = (rb - lt).clamp(min=0)  # [N,M,2]
#     inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

#     union = area1[:, None] + area2 - inter

#     iou = inter / union
#     return iou, union

# def generalized_box_iou(boxes1, boxes2):
#     """
#     Generalized IoU from https://giou.stanford.edu/

#     The boxes should be in [x0, y0, x1, y1] format

#     Returns a [N, M] pairwise matrix, where N = len(boxes1)
#     and M = len(boxes2)
#     """
#     # degenerate boxes gives inf / nan results
#     # so do an early check
#     if not (boxes1[:, 2:] >= boxes1[:, :2]).all():
#         for i in range(boxes1.size(0)):
#             if not (boxes1[i, 2:] >= boxes1[i, :2]).all():
#                 print(f'bug in box_iou: {boxes1[i]}')
#     assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
#     assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
#     iou, union = box_iou(boxes1, boxes2)

#     lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
#     rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

#     wh = (rb - lt).clamp(min=0)  # [N,M,2]
#     area = wh[:, :, 0] * wh[:, :, 1]

#     return iou - (area - union) / area


def box_ious(boxes1, boxes2, types='iou'):
    """
    Calculate the ious of two boxes array.

    Args:
        boxes1(Tensor): a list of boxes, (N,(cx,cy,w,h)).
        boxes2(Tensor): a list of boxes, (N,(cx,cy,w,h)).
        types(str): the types of iou.
    """

    types = [types] if isinstance(types, str) else types
    results = {}
    boxes2=boxes2.astype(boxes1.dtype)

    b1_cxcy = boxes1[:, :2]  # center x and center y of boxes1, (N,(cx,cy))
    b2_cxcy = boxes2[:, :2]  # center x and center y of boxes2, (N,(cx,cy))
    # half of the width and height of boxes1, (N,(0.5w,0.5h))
    b1_wh_half = 0.5 * boxes1[:, 2:]
    # half of the width and height of boxes2, (N,(0.5w,0.5h))
    b2_wh_half = 0.5 * boxes2[:, 2:]

    b1_lt = b1_cxcy - b1_wh_half  # left top of boxes1, (N,(lt_x,lt_y))
    b1_rb = b1_cxcy + b1_wh_half  # right bottom of boxes1, (N,(rb_x,rb_y))
    b2_lt = b2_cxcy - b2_wh_half  # left top of boxes2, (N,(lt_x,lt_y))
    b2_rb = b2_cxcy + b2_wh_half  # right bottom of boxes2, (N,(rb_x,rb_y))

    # dependence: iou, giou, diou, ciou
    _, intersect_lt = ops.max(ops.stack([b1_lt, b2_lt]))  # (N,(x,y))
    _, intersect_rb = ops.min(ops.stack([b1_rb, b2_rb]))  # (N,(x,y))
    dif = intersect_rb - intersect_lt  # (N,(dif_x,dif_y))
    clip_max_value = 10
    intersect_wh = ops.clip_by_value(dif, clip_value_min=0, clip_value_max=clip_max_value)  # (N,(w,h))
    intersect_area = ops.prod(intersect_wh, axis=1)  # (N,area)
    union_area = ops.prod(boxes1[:, 2:], 1) + ops.prod(boxes2[:, 2:], 1) - intersect_area  # (N,area)
    clip_max_value = 10
    # iou = intersect_area / union_area.clamp(min=1e-6)  # N
    iou = intersect_area / \
        ops.clip_by_value(union_area, clip_value_min=1e-6,
                          clip_value_max=clip_max_value)
    results['iou'] = iou
    # dependence: giou, diou, ciou

    # init parameters to avoid bug in graph mode
    enclose_lt = 0
    enclose_rb = 0
    enclose_wh = 0
    enclose_area = 0
    diou = 0
    if 'giou' in types or 'diou' in types or 'ciou' in types:
        enclose_lt = ops.min(ops.stack([b1_lt, b2_lt]))[1]
        enclose_rb = ops.max(ops.stack([b1_rb, b2_rb]))[1]
    # dependence: giou
    if 'giou' in types:
        enclose_wh = (enclose_rb - enclose_lt)  # N,2
        enclose_area = ops.prod(enclose_wh, 1)  # N
        giou = iou - (enclose_area - union_area) / enclose_area
        results['giou'] = giou
    # dependence: diou, ciou
    if 'diou' in types or 'ciou' in types:
        center_dist2 = ops.reduce_sum(ops.pow(boxes1[:, :2] - boxes2[:, :2], 2), axis=1)  # (N)
        enclose_dist2 = ops.reduce_sum(ops.pow(enclose_lt - enclose_rb, 2), axis=1)  # (N)
        diou = iou - center_dist2 / enclose_dist2
        results['diou'] = diou
    # dependence: ciou
    if 'ciou' in types:
        arctan = ops.atan(boxes1[:, 2]/boxes1[:, 3]) - \
            ops.atan(boxes2[:, 2]/boxes2[:, 3])  # N
        v = (4 / math.pi**2) * ops.pow(arctan, 2)
        alpha = v / (1 - iou + v + 1e-6)
        ciou = diou - alpha * v
        results['ciou'] = ciou
    return results


class NestedTensor(object):

    def __init__(self, tensors: Tensor, mask: Tensor):
        self.tensors = tensors
        self.mask = mask

    # def to(self, *args, **kwargs):
    #     cast_tensor = self.tensors.to(*args, **kwargs)
    #     cast_mask = self.mask.to(
    #         *args, **kwargs) if self.mask is not None else None
    #     return type(self)(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return repr(self.tensors)


def calc_resnet_flops(net: nn.Cell, HW=1.0):
    gated_flops, const_flops = [], 0
    for n, m in net.cells_and_names():
        if n.count('.') > 1:
            continue
        elif isinstance(m, nn.Conv2d):
            HW = 0.25 * HW if m.stride[0] > 1 else HW
            const_flops += m.weight.size * HW
        elif isinstance(m, nn.MaxPool2d):
            HW = 0.25 * HW if m.stride > 1 else HW
        elif isinstance(m, resnet.Bottleneck):
            if m.conv2.stride[0] > 1:
                const_flops += (m.conv1.weight.size + m.conv2.weight.size * 0.25 + m.conv3.weight.size * 0.25 +
                                m.downsample[0].weight.size * 0.25) * HW
                HW = 0.25 * HW
            else:
                if m.downsample is None:
                    shortcut_flops = 0
                else:
                    shortcut_flops = m.downsample[0].weight.size
                gated_flops.append([0, 0])
                gated_flops[-1][0] += (m.conv2.weight.size + m.conv3.weight.size) * HW
                gated_flops[-1][1] += (m.conv1.weight.size + shortcut_flops) * HW

    gated_flops = Tensor(gated_flops)
    total_flops = float(gated_flops.sum()) + const_flops

    return gated_flops, const_flops, total_flops


def calc_batch_flops(filter_gates, layer_gates, gated_flops, const_flops, total_flops):
    '''
    Calculate batch flops.
    Args:
        filter_gates: [0,1], shape: (B, L)
        layer_gates: {0,1}, shape: (B, L)
        gated_flops: shape: (L, 2)
        const_flops: float
        total_flops: float
    Returns:
        batch_gated_flops: shape: (B,)
        flops_ratio: shape: (1,)
    '''
    gated_flops = gated_flops.to(filter_gates.device)
    filter_gated_flops = filter_gates * gated_flops[:, 0] + gated_flops[:, 1]
    batch_gated_flops = (filter_gated_flops * layer_gates).sum(1) + const_flops
    total_gated_flops = batch_gated_flops.sum()
    batch_size = batch_gated_flops.size(0)
    flops_ratio = total_gated_flops / (batch_size * total_flops)

    return batch_gated_flops, flops_ratio


# class STE(Function):
#     @staticmethod
#     def forward(ctx, x):
#         rand = torch.rand_like(x) if x.requires_grad else 0.5
#         return (rand <= x).float()

#     @staticmethod
#     def backward(ctx, grad_outputs):
#         return grad_outputs

# class STE(Function):
#     @staticmethod
#     def forward(ctx, x):
#         rand = torch.rand_like(x) if x.requires_grad else 0.5
#         output = rand <= x
#         ctx.save_for_backward(output)
#         return output.float()

#     @staticmethod
#     def backward(ctx, grad_outputs):
#         output, = ctx.saved_tensors
#         grad_outputs = grad_outputs.clone()
#         grad_outputs[~output] = 0.0
#         return grad_outputs

# class STE(Function):
#     @staticmethod
#     def forward(ctx, x):
#         return (0.5 <= x).float()

#     @staticmethod
#     def backward(ctx, grad_outputs):
#         return grad_outputs

# ste = STE.apply


def pt_flatten(input: Tensor, start_dim: int = 0, end_dim: int = -1) -> Tensor:
    """
    Flatten a Tensor like pytorch.

    Args:
        input: the tensor to be flattend;
        start_dim: the dim where we start the flatten operation;
        end_dim: the dim where we end the flatten operation;

    """
    input_shape = input.shape
    shape_len = len(input_shape)
    output_shape = []
    for i in range(0, start_dim):
        output_shape.append(input_shape[i])

    if end_dim == -1:
        output_shape.append(-1)
        return F.reshape(input, tuple(output_shape))
    else:
        shape_mul = 1
        for i in range(start_dim, end_dim + 1):
            shape_mul *= input_shape[i]
        output_shape.append(shape_mul)
        for i in range(end_dim, shape_len):
            output_shape.append(input_shape[i])
        return F.reshape(input, tuple(output_shape))


def pt_topk(input: Tensor, k, axis=-1, sorted=True):
    """
    Finds values and indices of the `k` largest entries along the axis like pytorch.

    """

    if axis == -1 or axis == len(input.shape) - 1:
        return ops.top_k(input, k, sorted)
    else:
        input_copied = input.copy()
        last_dim = len(input.shape) - 1
        dim_list = [i for i in range(len(input.shape))]
        dim_list[axis] = last_dim
        dim_list[last_dim] = axis
        dim_tuple = tuple(dim_list)
        input_copied = ops.transpose(input_copied, tuple(dim_tuple))
        values, indices = ops.top_k(input_copied, k, sorted)
        return ops.transpose(values, dim_tuple), ops.transpose(indices, dim_tuple)


def get_backbone_ckpt_path(cache_name=None):
    resnet_ckpt_root = './weights/pretrain-weights'
    return os.path.join(resnet_ckpt_root, cache_name)


def show_cells_and_names(net: nn.Cell):
    """
    Returns:
        Iteration, all the child cells and corresponding names in the cell.
    """
    for name, cell in net.cells_and_names():
        print(name)
        print(cell)
        print(cell.requires_grad)
        print("\n")


def show_name_cells(net: nn.Cell):
    """
    Returns an iterator over all immediate cells in the network.

    include name of the cell and cell itself.

    Returns:
        Dict, all the child cells and corresponding names in the cell.
    """
    for name, cell in net.name_cells().items():
        print(name)
        # print(cell)
        print(cell.requires_grad)
        print("\n")


def show_params_and_names(net: nn.Cell):
    for name, params in net.parameters_and_names():
        print(name)
        print(params.requires_grad)
        print("\n")


def show_type(object):
    print(type(object))


def entropy(probs, eps=1e-5):
    return -probs * ops.log(probs + eps) - (1 - probs) * ops.log(1 - probs + eps)

def count_useful_length(x):
    assist = (x != 0).sum(1)
    _, max_len = ops.max(assist, 0)
    return max_len.astype(ms.int64)

def remove_zeros(x):
    """Remove redundant zeros in the end of tensor.

    Args:
        x (ms.Tensor): Tensor.

    Returns:
        _type_: _description_
    """
    # if type(x) != Tensor:
    #     new_dict = dict()
    #     for key, value in x.items():
    #         assist = (value != 0).sum(1)
    #         _, max_len = ops.max(assist, 0)
    #         res = value[:, :int(max_len)]
    #         new_dict[key] = res
    #     return new_dict
    # else:
    assist = (x != 0).sum(1)
    _, max_len = ops.max(assist, 0)
    res = x[:, :max_len.astype(ms.int64)]
    return res
