# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""

import src.resnet as resnet
import mindspore.nn as nn
from mindspore import Tensor
from src.xenv import console
from src.utils import get_backbone_ckpt_path, pt_flatten
import mindspore as ms
import mindspore.ops as ops
import math



class Backbone(nn.Cell):
    """ResNet backbone with frozen BatchNorm"""

    def __init__(self, net_name: str, dilation=False, trainable_layers=3, return_layers=1, norm_layer=None, cache_name=None, use_selector=True):
        """
        Initialize the Backbone.

        Args:
            net_name(str): The name of the backbone net(resnet50/resnet101/...).
            dilation(bool): Whether to replace stride with dilation in the backbone, default: False.
            trainable_layers(int): The number of layers to train(requires_grad = True), default: 3.
            return_layers(int): The number of layers to return, default: 1.
            norm_layer(nn.Cell): The type of norm layers used in backbone.
            cache_name(str): The name of cache.
            use_selector(bool): Whether to use selector, default: True.
        """
        super(Backbone, self).__init__()

        assert 1 <= return_layers <= 4
        assert 0 <= trainable_layers <= 5
        self.use_selector = use_selector
        self.num_channels = 512 if net_name in ("resnet18", "resnet34") else 2048

        self.body = getattr(resnet, net_name)(replace_stride_with_dilation=[False, False, dilation], norm_layer=norm_layer, use_selector=self.use_selector)

        if trainable_layers != 5:
            # frozen some layers
            layer_names = ['conv1', 'layer1', 'layer2', 'layer3', 'layer4']
            layers_to_train = layer_names[-trainable_layers:]
            # # frozen BN
            # for name, parameter in self.parameters_and_names():
            #     if 'bn' in name or 'downsample.1' in name:
            #         parameter.requires_grad = False
            # for name, cell in self.cells_and_names():
            #     if 'bn' in name or 'downsample.1' in name:
            #         cell.set_train(False)
            for name, param in self.body.parameters_and_names():
                if all([not name.startswith(layer) for layer in layers_to_train]):
                    param.requires_grad = False

        if cache_name is not None:
            # load checkpoint.
            ckpt_path = get_backbone_ckpt_path(cache_name)
            state_dict = ms.load_checkpoint(ckpt_path)
            missing_list = ms.load_param_into_net(self.body, state_dict)
            print("====== Loading pre-trained weights into ResNet ======")
            print("====== Missing weights are: ======")
            print(missing_list)
            print("======")

    def construct(self, tensor_list, filter_layer_selection):
        if self.use_selector:
            xs = self.body(tensor_list[0], filter_layer_selection)
        else:
            xs = self.body(tensor_list[0])
        mask = tensor_list[1][None]

        _mask = ops.cast(ops.interpolate(mask, sizes=xs.shape[-2:], mode='bilinear'), ms.bool_)[0]

        out = [xs, _mask]
        return out


class PositionEmbeddingSine(nn.Cell):

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None, eps=1e-6):
        super(PositionEmbeddingSine, self).__init__()
        self.half_pos_feats = num_pos_feats // 2
        self.temperature = temperature
        self.normalize = normalize
        self.eps = eps
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        self.scale = 2 * math.pi if scale is None else scale
        self.flatten_function = pt_flatten
        self.arange = ms.numpy.arange

    def construct(self, tensor_list: list):
        feat, mask = tensor_list[0], tensor_list[1]

        y_embed = mask.cumsum(1, dtype=ms.float16)
        x_embed = mask.cumsum(2, dtype=ms.float16)
        if self.normalize:
            y_embed = y_embed / (y_embed[:, -1:, :] + self.eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + self.eps) * self.scale

        dim_t = self.arange(self.half_pos_feats, dtype=feat.dtype)
        # dim_t = dim_t.to(feat.device)
        dim_t[1::2] = dim_t[::2]
        dim_t = self.temperature**(dim_t / self.half_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = self.flatten_function(ops.stack((ops.sin(pos_x[:, :, :, 0::2]), ops.cos(pos_x[:, :, :, 1::2])), axis=4), 3)
        pos_y = self.flatten_function(ops.stack((ops.sin(pos_y[:, :, :, 0::2]), ops.cos(pos_y[:, :, :, 1::2])), axis=4), 3)
        pos = ops.transpose(input_x=ops.concat((pos_y, pos_x), axis=3), input_perm=(0, 3, 1, 2))

        return pos.astype(feat.dtype)


class Joiner(nn.Cell):
    """Join the backbone and position_embedding"""

    def __init__(self, backbone, position_embedding):
        super().__init__()
        self.backbone = backbone
        self.num_channels = backbone.num_channels
        self.position_embedding = position_embedding
        self.cast = ops.Cast()
        self.expand_dims = ops.ExpandDims()
        self.squeeze = ops.Squeeze(axis=0)

    def construct(self, tensor_list: list, filter_layer_selection):
        nts = self.backbone(tensor_list, filter_layer_selection)

        out = nts
        pos = self.position_embedding(nts)

        # out=self.cast(out,tensor_list.tensors.dtype)
        # pos=self.cast(pos,tensor_list.mask.dtype)
        return out, pos  # out:[[x,_mask],[x,_mask],...]


NET_CACHE = {
    'resnet50': 'resnet50-19c8e357.pth',
    'resnet101': 'resnet101-5d3b4d8f.pth',
    'resnet50-coco': 'resnet50_fpn_coco-258fb6c6.pth',
    'resnet50-imagenet': 'resnet50-imagenet.pth',
    'resnet101-imagenet': 'resnet101-imagenet.pth',
    'resnet50-unc': 'selector-resnet50-unc.ckpt',
    'resnet101-unc': 'resnet101-unc.ckpt',
    'resnet50-gref': 'selector-resnet50-gref.ckpt',
    'resnet101-gref': 'resnet101-gref.ckpt',
    'resnet50-referit': 'selector-resnet50-referit.ckpt',
    'resnet101-referit': 'resnet101-referit.ckpt',
}


def build_backbone(net_name, pos_embed_dim=256, dilation=False, trainable_layers=3, norm_layer=None, cache_name=None, use_selector=True):
    if net_name not in NET_CACHE and cache_name is None:
        console.print("Backbone was randomly initialized.", style="yellow")
    elif net_name in NET_CACHE and cache_name is None:
        cache_name = NET_CACHE[net_name]

    net_name = net_name.split('-')[0]

    position_embedding = PositionEmbeddingSine(pos_embed_dim, normalize=True)
    backbone = Backbone(net_name, dilation, trainable_layers, norm_layer=norm_layer, cache_name=cache_name, use_selector=use_selector)
    model = Joiner(backbone, position_embedding)
    return model


def test0():
    # test the class Backbone
    res50 = build_backbone('resnet50-unc', norm_layer="frozen_bn")
    print('res50 built!')
    res50.set_train(True)
    for name, cell in res50.cells_and_names():
        if 'bn1' in name:
            cell.set_train(False)
            print(cell.momentum)
            print(cell.moving_mean)
            print(cell.moving_variance)
            print(cell.requires_grad)
            print(cell.training)
    # test0 passed!


if __name__ == '__main__':
    test0()
