"""ResNet modified"""
import mindspore as ms
import math
import numpy as np
from scipy.stats import truncnorm
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
# from ResNet.src.model_utils.config import config
# import resnet.config as config
from typing import Type, Any, Callable, Union, List, Optional
from mindspore.common.initializer import initializer, HeNormal, HeUniform, One, Zero


def conv_variance_scaling_initializer(in_channel, out_channel, kernel_size):
    fan_in = in_channel * kernel_size * kernel_size
    scale = 1.0
    scale /= max(1., fan_in)
    stddev = (scale**0.5) / .87962566103423978
    # if config.net_name == "resnet152":
    #     stddev = (scale ** 0.5)
    mu, sigma = 0, stddev
    weight = truncnorm(-2, 2, loc=mu, scale=sigma).rvs(out_channel * in_channel * kernel_size * kernel_size)
    weight = np.reshape(weight, (out_channel, in_channel, kernel_size, kernel_size))
    return Tensor(weight, dtype=mstype.float16)


def _weight_variable(shape, factor=0.01):
    init_value = np.random.randn(*shape).astype(np.float16) * factor
    return Tensor(init_value)


def calculate_gain(nonlinearity, param=None):
    """calculate_gain"""
    linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d']
    res = 0
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        res = 1
    elif nonlinearity == 'tanh':
        res = 5.0 / 3
    elif nonlinearity == 'relu':
        res = math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        if param is None:
            neg_slope = 0.01
        elif not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
            neg_slope = param
        else:
            raise ValueError("neg_slope {} not a valid number".format(param))
        res = math.sqrt(2.0 / (1 + neg_slope**2))
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))
    return res


def _calculate_fan_in_and_fan_out(tensor):
    """_calculate_fan_in_and_fan_out"""
    dimensions = len(tensor)
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
    if dimensions == 2:  # Linear
        fan_in = tensor[1]
        fan_out = tensor[0]
    else:
        num_input_fmaps = tensor[1]
        num_output_fmaps = tensor[0]
        receptive_field_size = 1
        if dimensions > 2:
            receptive_field_size = tensor[2] * tensor[3]
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size
    return fan_in, fan_out


def _calculate_correct_fan(tensor, mode):
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError("Unsupported mode {}, please use one of {}".format(mode, valid_modes))
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == 'fan_in' else fan_out


def kaiming_normal(inputs_shape, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    fan = _calculate_correct_fan(inputs_shape, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    return np.random.normal(0, std, size=inputs_shape).astype(np.float16)


def kaiming_uniform(inputs_shape, a=0., mode='fan_in', nonlinearity='leaky_relu'):
    fan = _calculate_correct_fan(inputs_shape, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    # Calculate uniform bounds from standard deviation
    bound = math.sqrt(3.0) * std
    return np.random.uniform(-bound, bound, size=inputs_shape).astype(np.float16)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    weight_shape = (out_planes, in_planes, 1, 1)
    # weight = Tensor(kaiming_normal(weight_shape, mode='fan_out', nonlinearity='relu'))
    weight = initializer(HeNormal(mode='fan_out', nonlinearity='relu'), weight_shape)

    # if config.net_name == "resnet152":
    #     weight = _weight_variable(weight_shape)

    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, pad_mode='pad', weight_init=weight, has_bias=False)
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, has_bias=False)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    weight_shape = (out_planes, in_planes, 3, 3)
    # weight = Tensor(kaiming_normal(weight_shape, mode='fan_out', nonlinearity='relu'))
    weight = initializer(HeNormal(mode='fan_out', nonlinearity='relu'), weight_shape)

    # if config.net_name == "resnet152":
    #     weight = _weight_variable(weight_shape)

    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, pad_mode='pad', group=groups, has_bias=False, dilation=dilation, weight_init=weight)


def _bn1d(channel):
    return nn.BatchNorm1d(channel, gamma_init=One(), beta_init=Zero(), moving_mean_init=Zero(), moving_var_init=One())


def _bn2d(channel):
    return nn.BatchNorm2d(channel, gamma_init=1, beta_init=0, moving_mean_init=0, moving_var_init=1)


def _bn_last(channel):
    return nn.BatchNorm2d(channel, momentum=0.9, gamma_init=0, beta_init=0, moving_mean_init=0, moving_var_init=1)


class FrozenBatchNorm2d(nn.layer.normalization._BatchNorm):

    def __init__(self,
                 num_features,
                 eps=1e-5,
                 momentum=1.0,
                 affine=True,
                 gamma_init='ones',
                 beta_init='zeros',
                 moving_mean_init='zeros',
                 moving_var_init='ones',
                 use_batch_statistics=None,
                 data_format='NCHW'):
        super().__init__(num_features, eps, momentum, affine, gamma_init, beta_init, moving_mean_init, moving_var_init, use_batch_statistics, data_format)
        self.requires_grad = False
        self.training = False
    def construct(self, x):
        return self.bn_infer(x, self.gamma,self.beta,self.moving_mean,self.moving_variance)[0]

def _frozen_bn2d(channel):
    return FrozenBatchNorm2d(channel, gamma_init=1, beta_init=0, moving_mean_init=0, moving_var_init=1)


def _fc(in_planes, out_planes, has_bias=True, bias_init=0):
    weight_shape = (out_planes, in_planes)
    weight = Tensor(kaiming_uniform(weight_shape, a=math.sqrt(5)))

    return nn.Dense(in_planes, out_planes, has_bias=has_bias, weight_init=weight, bias_init=bias_init)


class BasicBlock(nn.Cell):
    expansion: int = 1

    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride: int = 1,
                 downsample: Optional[nn.Cell] = None,
                 groups: int = 1,
                 base_width: int = 64,
                 dilation: int = 1,
                 norm_layer: Optional[Callable[..., nn.Cell]] = None) -> None:
        super(BasicBlock, self).__init__()

        if norm_layer is None:
            norm_layer = _bn2d

        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        # zero init
        self.bn2 = _bn_last(planes)
        self.downsample = downsample
        self.stride = stride
        ##### skip flag########
        self.skip = False

    def construct(self, x: Tensor) -> Tensor:
        if not self.training and self.skip:
            return x

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Cell):
    expansion: int = 4

    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride: int = 1,
                 downsample: Optional[nn.Cell] = None,
                 groups: int = 1,
                 base_width: int = 64,
                 dilation: int = 1,
                 norm_layer: Optional[Callable[..., nn.Cell]] = None) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = _bn2d

        width = int(planes * (base_width / 64.)) * groups

        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        # zero init
        self.bn3 = _bn_last(planes * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride
        ######### skip flag ########
        self.skip = False
        ############################

    def construct(self, x: Tensor) -> Tensor:
        if not self.training and self.skip:
            return x

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # ========== hook ==========
        # if self.conv2.stride[0] == 1:
        #     out_channel = self.conv2.out_channels
        #     mix_gates = filter_layer_selection[index][2]
        #     mix_gates = mix_gates.view(-1, out_channel, 1, 1)
        #     out = out * mix_gates
        # ========== hook ==========
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        # ========== hook ==========
        # if self.conv2.stride[0] == 1:
        #     mix_gates = filter_layer_selection[index][-1]
        #     mix_gates = mix_gates.view(-1, 1, 1, 1)
        #     out = out * mix_gates
        #     index += 1
        # ========== hook ==========

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BottleneckWithHook(nn.Cell):
    expansion: int = 4

    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride: int = 1,
                 downsample: Optional[nn.Cell] = None,
                 groups: int = 1,
                 base_width: int = 64,
                 dilation: int = 1,
                 norm_layer: Optional[Callable[..., nn.Cell]] = None) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = _bn2d

        width = int(planes * (base_width / 64.)) * groups

        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        # zero init
        self.bn3 = _bn_last(planes * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride
        ######### skip flag ########
        self.skip = False
        ############################

    def construct(self, x: Tensor, filter_layer_selection, index) -> Tensor:
        if not self.training and self.skip:
            return x

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # ========== hook ==========
        if self.conv2.stride[0] == 1:
            out_channel = self.conv2.out_channels
            mix_gates = filter_layer_selection[index][2].astype(out.dtype)
            mix_gates = mix_gates.view(-1, out_channel, 1, 1)
            out = out * mix_gates
        # ========== hook ==========
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        # ========== hook ==========
        if self.conv2.stride[0] == 1:
            mix_gates = filter_layer_selection[index][-1].astype(out.dtype)
            mix_gates = mix_gates.view(-1, 1, 1, 1)
            out = out * mix_gates
            index += 1
        # ========== hook ==========

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out, index


class ResLayerList(nn.Cell):

    def __init__(self,
                 block: Type[Union[BasicBlock, Bottleneck, BottleneckWithHook]],
                 planes,
                 blocks: int,
                 stride: int = 1,
                 dilate=False,
                 norm_layer=_bn2d,
                 previous_dilation=None,
                 inplanes=None,
                 base_width=None,
                 groups=None):
        super().__init__()
        self.dilation = previous_dilation
        downsample = None
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.SequentialCell([conv1x1(inplanes, planes * block.expansion, stride), norm_layer(planes * block.expansion)])

        self.layers = nn.CellList()
        self.layers.append(block(inplanes, planes, stride, downsample, groups, base_width, previous_dilation, norm_layer))
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            self.layers.append(block(inplanes, planes, groups=groups, base_width=base_width, dilation=self.dilation, norm_layer=norm_layer))

    def construct(self, x, filter_layer_selection, index):
        output = x
        for layer in self.layers:
            output, index = layer(output, filter_layer_selection, index)
        return output, index


class ResNet(nn.Cell):

    def __init__(self,
                 block: Type[Union[BasicBlock, Bottleneck]],
                 layers: List[int],
                 num_classes: int = 1000,
                 zero_init_residual: bool = False,
                 groups: int = 1,
                 width_per_group: int = 64,
                 replace_stride_with_dilation: Optional[List[bool]] = None,
                 norm_layer: Optional[Callable[..., nn.Cell]] = None) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = _bn2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=self.inplanes,
                               kernel_size=7,
                               stride=2,
                               padding=3,
                               pad_mode='pad',
                               has_bias=False,
                               weight_init=initializer(HeNormal(mode='fan_out', nonlinearity='relu'), shape=(self.inplanes, 3, 7, 7)))
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = _fc(512 * block.expansion, num_classes)

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int, stride: int = 1, dilate: bool = False) -> nn.SequentialCell:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation

        if dilate:
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.SequentialCell([conv1x1(self.inplanes, planes * block.expansion, stride), norm_layer(planes * block.expansion)])

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))

        return nn.SequentialCell(layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # pad before maxpool
        pad = ops.Pad(((0, 0), (0, 0), (1, 0), (1, 0)))
        x = pad(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.avgpool(x)
        # x = x.flatten()
        # x = self.fc(x)

        return x

    def construct(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


class ResNetWithHook(nn.Cell):
    """
    ResNet with hook.
    """

    def __init__(self,
                 block: Type[Union[BasicBlock, Bottleneck]],
                 layers: List[int],
                 num_classes: int = 1000,
                 zero_init_residual: bool = False,
                 groups: int = 1,
                 width_per_group: int = 64,
                 replace_stride_with_dilation: Optional[List[bool]] = None,
                 norm_layer: Optional[Callable[..., nn.Cell]] = None) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = _bn2d
        elif norm_layer == "frozen_bn":
            norm_layer = _frozen_bn2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=self.inplanes,
                               kernel_size=7,
                               stride=2,
                               padding=3,
                               pad_mode='pad',
                               has_bias=False,
                               weight_init=initializer(HeNormal(mode='fan_out', nonlinearity='relu'), (self.inplanes, 3, 7, 7)))
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='valid')

        #layer1
        self.layer1 = ResLayerList(block, 64, layers[0], norm_layer=norm_layer, previous_dilation=self.dilation, inplanes=self.inplanes, base_width=self.base_width, groups=self.groups)
        # self.dilation
        self.inplanes = 64 * block.expansion

        # layer2
        self.layer2 = ResLayerList(block,
                                   128,
                                   layers[1],
                                   stride=2,
                                   dilate=replace_stride_with_dilation[0],
                                   norm_layer=norm_layer,
                                   previous_dilation=self.dilation,
                                   inplanes=self.inplanes,
                                   base_width=self.base_width,
                                   groups=self.groups)
        self.dilation = self.dilation * 2 if replace_stride_with_dilation[0] else self.dilation
        self.inplanes = 128 * block.expansion

        # layer3
        self.layer3 = ResLayerList(block,
                                   256,
                                   layers[2],
                                   stride=2,
                                   dilate=replace_stride_with_dilation[1],
                                   previous_dilation=self.dilation,
                                   norm_layer=norm_layer,
                                   inplanes=self.inplanes,
                                   base_width=self.base_width,
                                   groups=self.groups)
        self.dilation = self.dilation * 2 if replace_stride_with_dilation[1] else self.dilation
        self.inplanes = 256 * block.expansion

        # layer4
        self.layer4 = ResLayerList(block,
                                   512,
                                   layers[3],
                                   stride=2,
                                   dilate=replace_stride_with_dilation[2],
                                   previous_dilation=self.dilation,
                                   norm_layer=norm_layer,
                                   inplanes=self.inplanes,
                                   base_width=self.base_width,
                                   groups=self.groups)

    def construct(self, x: Tensor, filter_layer_select, index: int = 0):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        pad = ops.Pad(((0, 0), (0, 0), (1, 0), (1, 0)))
        x = pad(x)
        x = self.maxpool(x)

        x, index = self.layer1(x, filter_layer_select, index)
        x, index = self.layer2(x, filter_layer_select, index)
        x, index = self.layer3(x, filter_layer_select, index)
        x, index = self.layer4(x, filter_layer_select, index)

        # x = self.avgpool(x)
        # x = x.flatten()
        # x = self.fc(x)
        return x


def _resnet(arch: str, block: Type[Union[BasicBlock, Bottleneck]], layers: List[int], pretrained: bool, progress: bool, use_selector: bool = True, **kwargs: Any) -> ResNet:
    if use_selector:
        model = ResNetWithHook(block, layers, **kwargs)
    else:
        model = ResNet(block, layers, **kwargs)
    # pretrained
    if pretrained:
        param_dict = ms.load_checkpoint(ckpt_file_name=ckpt_file)
        model = ms.load_param_into_net(model, param_dict)

    return model


def resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)


def resnet34(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet50(pretrained: bool = False, progress: bool = True, use_selector=True, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    if use_selector:
        return _resnet('resnet50', BottleneckWithHook, [3, 4, 6, 3], pretrained, progress, use_selector, **kwargs)
    else:
        return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, use_selector, **kwargs)


def resnet101(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)


def resnet152(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress, **kwargs)


def resnext50_32x4d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)


if __name__ == '__main__':
    model = resnet50()
