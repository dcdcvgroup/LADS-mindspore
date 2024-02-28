"""
Loss
目前主要用的是这个文件
"""
from mindspore import Parameter
import mindspore.nn as nn
import mindspore
import mindspore as ms
import src.xenv as xenv
from mindspore.ops import composite as C
from mindspore.context import ParallelMode, get_auto_parallel_context
from mindspore.communication.management import get_group_size
from mindspore.communication.management import GlobalComm, get_group_size
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
from mindspore.ops import operations as P
import mindspore.ops as ops
from mindspore.ops import functional as F
from src.utils import box_ious, entropy
from mindspore import Tensor
import math
from mindspore.parallel._utils import _get_device_num, _get_gradients_mean,\
    _get_parallel_mode, _get_enable_parallel_optimizer, _is_pynative_parallel
from mindspore.context import ParallelMode



class NetWithLoss(nn.Cell):
    """
    Build net that has loss calculation in the end of the origin net.

    Args:
        network(nn.Cell): The network that will be added the loss calculation in the end.
        xiou_loss_type(str): The type of xiou loss.
        bbox_loss_coef(float)
        xiou_loss_coef(float)
    """

    def __init__(self, network: nn.Cell, xiou_loss_type, bbox_loss_coef, xiou_loss_coef, arch_loss_coef):
        super().__init__()
        self.network = network
        self.xiou_loss_type = xiou_loss_type
        self.bbox_loss_coef = bbox_loss_coef
        self.xiou_loss_coef = xiou_loss_coef
        self.arch_loss_coef = arch_loss_coef
        self.L1loss = nn.L1Loss(reduction='sum')
        self.box_ious = box_ious
        self.scalar_summary = ops.ScalarSummary()
        self.loss_bbox = Tensor([0], mindspore.float32)
        self.loss_xiou = Tensor([0], mindspore.float32)
        self.loss_arch = Tensor([0], mindspore.float32)

    def construct(self, input, bbox, mask, phrase):

        # calculate outputs of the net
        pred_prob_boxes, filter_layer_selections, attn_ffn_selections = self.network(input, bbox, mask, phrase)

        # calculate loss
        loss_bbox = self.L1loss(pred_prob_boxes, bbox) / bbox.shape[0]
        # self.scalar_summary("loss_bbox",loss_bbox)
        loss_xiou = (1 - self.box_ious(pred_prob_boxes, bbox, self.xiou_loss_type)[self.xiou_loss_type]).mean()
        # self.scalar_summary("loss_xiou",loss_xiou)
        # not use selector
        if filter_layer_selections == None and attn_ffn_selections == None:
            total_loss = self.bbox_loss_coef * loss_bbox + self.xiou_loss_coef * loss_xiou
            # self.scalar_summary("total_loss",total_loss)
            return total_loss

        # mutual information of filters
        filter_sigma = ops.concat([selection[1] for selection in filter_layer_selections], axis=1)
        filter_local_entropy = ops.mean(entropy(filter_sigma), axis=0)
        filter_global_entropy = -entropy(ops.mean(filter_sigma, axis=0))
        loss_arch_filter = (filter_local_entropy + filter_global_entropy).mean()
        # mutual information of layers
        layer_sigma = ops.concat([selection[-2] for selection in filter_layer_selections], axis=1)
        layer_local_entropy = ops.mean(entropy(layer_sigma), axis=0)
        layer_global_entropy = -entropy(ops.mean(layer_sigma, axis=0))
        loss_arch_layer = (layer_local_entropy + layer_global_entropy).sum() * 0.075

        # mutual information of transformers(self-attn & ffn).
        trans_sigma = ops.concat(
            ([selection[1] for selection in attn_ffn_selections] + [selection[-2]
                                                                    for selection in attn_ffn_selections]),
            axis=1)
        trans_local_entropy = 0.1 * ops.mean(entropy(trans_sigma), axis=0)
        trans_global_entropy = -entropy(ops.mean(trans_sigma, axis=0))
        loss_arch_trans = (trans_local_entropy + trans_global_entropy).sum() * 0.05

        loss_arch = loss_arch_filter + loss_arch_layer + loss_arch_trans
        # self.scalar_summary("loss_arch",loss_arch)
        total_loss = (self.bbox_loss_coef * loss_bbox + self.xiou_loss_coef * loss_xiou +
                      self.arch_loss_coef * loss_arch)
        # self.scalar_summary("total_loss",total_loss)
        # self.loss_bbox = loss_bbox
        # self.loss_xiou = loss_xiou
        # self.loss_arch = loss_arch

        return total_loss, loss_bbox, loss_xiou, loss_arch


class NetWithLossV2(nn.Cell):
    """
    Build net that has loss calculation in the end of the origin net.
    From train_lads_entropy_mean_acc1_word2.py

    Args:
        network(nn.Cell): The network that will be added the loss calculation in the end.
        xiou_loss_type(str): The type of xiou loss.
        bbox_loss_coef(float)
        xiou_loss_coef(float)
    """

    def __init__(self, network: nn.Cell, xiou_loss_type, bbox_loss_coef, xiou_loss_coef, arch_loss_coef, accumulate_step=4):
        super().__init__()
        self.network = network
        self.xiou_loss_type = xiou_loss_type
        self.bbox_loss_coef = bbox_loss_coef
        self.xiou_loss_coef = xiou_loss_coef
        self.arch_loss_coef = arch_loss_coef
        self.accumulate_step = accumulate_step
        self.L1loss = nn.L1Loss(reduction='sum')
        self.box_ious = box_ious
        self.scalar_summary = ops.ScalarSummary()

    def construct(self, input, bbox, mask, phrase):

        # calculate outputs of the net
        pred_prob_boxes, filter_layer_selections, attn_ffn_selections = self.network(input, bbox, mask, phrase)
        if not self.training:
            return pred_prob_boxes

        # calculate loss
        loss_bbox = self.L1loss(pred_prob_boxes, bbox) / bbox.shape[0]
        # self.scalar_summary("loss_bbox",loss_bbox)
        loss_xiou = (1 - self.box_ious(pred_prob_boxes, bbox, self.xiou_loss_type)[self.xiou_loss_type]).mean()
        # self.scalar_summary("loss_xiou",loss_xiou)
        # not use selector
        if filter_layer_selections == None and attn_ffn_selections == None:
            total_loss = self.bbox_loss_coef * loss_bbox + self.xiou_loss_coef * loss_xiou
            # self.scalar_summary("total_loss",total_loss)
            return total_loss

        # mutual information of filters
        filter_sigma = ops.concat([selection[1] for selection in filter_layer_selections], axis=1)
        filter_local_entropy = 0.1 * ops.mean(entropy(filter_sigma), axis=0)
        filter_global_entropy = -entropy(ops.mean(filter_sigma, axis=0))
        loss_arch_filter = (filter_local_entropy + filter_global_entropy).mean()

        # mutual information of layers
        layer_sigma = ops.concat([selection[-2] for selection in filter_layer_selections], axis=1)
        layer_local_entropy = 0.1 * ops.mean(entropy(layer_sigma), axis=0)
        layer_global_entropy = -entropy(ops.mean(layer_sigma, axis=0))
        loss_arch_layer = (layer_local_entropy + layer_global_entropy).sum() * 0.075

        # mutual information of transformers(self-attn & ffn).
        trans_sigma = ops.concat(
            ([selection[1] for selection in attn_ffn_selections] + [selection[-2]
                                                                    for selection in attn_ffn_selections]),
            axis=1)
        # TODO: loss多个版本
        trans_local_entropy = 0.1 * ops.mean(entropy(trans_sigma), axis=0)
        trans_global_entropy = -entropy(ops.mean(trans_sigma, axis=0))
        loss_arch_trans = (trans_local_entropy + trans_global_entropy).sum() * 0.05

        loss_arch = loss_arch_filter + loss_arch_layer + loss_arch_trans
        # self.scalar_summary("loss_arch",loss_arch)
        total_loss = self.bbox_loss_coef * loss_bbox + self.xiou_loss_coef * loss_xiou + self.arch_loss_coef * loss_arch
        total_loss = total_loss / self.accumulate_step
        # self.scalar_summary("total_loss",total_loss)

        # active_layer_num = sum([filter_layer_selections[i][-3] for i in range(len(filter_layer_selections))])
        # active_trans_num = sum([attn_ffn_selections[i][0] for i in range(len(attn_ffn_selections))] + [attn_ffn_selections[i][-3] for i in range(len(attn_ffn_selections))])

        return total_loss, loss_bbox, loss_xiou, loss_arch  #, filter_sigma, layer_sigma, trans_sigma, active_layer_num, active_trans_num


class NetWithLossFpt(nn.Cell):
    """
    Build net that has loss calculation in the end of the origin net.
    From train_fpt.py

    Args:
        network(nn.Cell): The network that will be added the loss calculation in the end.
        xiou_loss_type(str): The type of xiou loss.
        bbox_loss_coef(float)
        xiou_loss_coef(float)
    """

    def __init__(self, network: nn.Cell, xiou_loss_type, bbox_loss_coef, xiou_loss_coef, arch_loss_coef):
        super().__init__()
        self.network = network
        self.xiou_loss_type = xiou_loss_type
        self.bbox_loss_coef = bbox_loss_coef
        self.xiou_loss_coef = xiou_loss_coef
        self.arch_loss_coef = arch_loss_coef
        self.L1loss = nn.L1Loss(reduction='sum')
        self.box_ious = box_ious
        self.scalar_summary = ops.ScalarSummary()

    def construct(self, input, bbox, mask, phrase):

        # calculate outputs of the net
        pred_prob_boxes, filter_layer_selections, attn_ffn_selections = self.network(input, bbox, mask, phrase)

        # calculate loss
        loss_bbox = self.L1loss(pred_prob_boxes, bbox) / bbox.shape[0]
        # self.scalar_summary("loss_bbox",loss_bbox)
        loss_xiou = (1 - self.box_ious(pred_prob_boxes, bbox, self.xiou_loss_type)[self.xiou_loss_type]).mean()
        # self.scalar_summary("loss_xiou",loss_xiou)
        total_loss = self.bbox_loss_coef * loss_bbox + self.xiou_loss_coef * loss_xiou
        return total_loss


class NetWithLossFptV2(nn.Cell):
    """
    Build net that has loss calculation in the end of the origin net.
    From train_fpt_word1_v2.py

    Args:
        network(nn.Cell): The network that will be added the loss calculation in the end.
        xiou_loss_type(str): The type of xiou loss.
        bbox_loss_coef(float)
        xiou_loss_coef(float)
    """

    def __init__(self, network: nn.Cell, xiou_loss_type, bbox_loss_coef, xiou_loss_coef, arch_loss_coef, accumulate_step):
        super().__init__()
        self.network = network
        self.xiou_loss_type = xiou_loss_type
        self.bbox_loss_coef = bbox_loss_coef
        self.xiou_loss_coef = xiou_loss_coef
        self.arch_loss_coef = arch_loss_coef
        self.accumulate_step = accumulate_step
        self.L1loss = nn.L1Loss(reduction='sum')
        self.box_ious = box_ious

    def construct(self, input, bbox, mask, phrase):

        # calculate outputs of the net
        pred_prob_boxes, filter_layer_selections, attn_ffn_selections = self.network(input, bbox, mask, phrase)
        if not self.training:
            return pred_prob_boxes

        # calculate loss
        loss_bbox = self.L1loss(pred_prob_boxes, bbox) / bbox.shape[0]
        # self.scalar_summary("loss_bbox",loss_bbox)
        loss_xiou = (1 - self.box_ious(pred_prob_boxes, bbox, self.xiou_loss_type)[self.xiou_loss_type]).mean()
        # not use selector
        if filter_layer_selections == None and attn_ffn_selections == None:
            total_loss = self.bbox_loss_coef * loss_bbox + self.xiou_loss_coef * loss_xiou
            # self.scalar_summary("total_loss",total_loss)
            return total_loss

        # mutual information of filters
        filter_sigma = ops.concat([selection[1] for selection in filter_layer_selections], axis=1)
        filter_local_entropy = ops.mean(entropy(filter_sigma), axis=0)
        filter_global_entropy = -entropy(ops.mean(filter_sigma, axis=0))
        loss_arch_filter = (filter_local_entropy + filter_global_entropy).mean()
        # mutual information of layers
        layer_sigma = ops.concat([selection[-2] for selection in filter_layer_selections], axis=1)
        layer_local_entropy = ops.mean(entropy(layer_sigma), axis=0)
        layer_global_entropy = -entropy(ops.mean(layer_sigma, axis=0))
        loss_arch_layer = (layer_local_entropy + layer_global_entropy).sum() * 0.075

        # mutual information of transformers(self-attn & ffn).
        trans_sigma = ops.concat(
            ([selection[1] for selection in attn_ffn_selections] + [selection[-2]
                                                                    for selection in attn_ffn_selections]),
            axis=1)
        trans_local_entropy = ops.mean(entropy(trans_sigma), axis=0)
        trans_global_entropy = -entropy(ops.mean(trans_sigma, axis=0)) * 3.0
        loss_arch_trans = (trans_local_entropy + trans_global_entropy).mean()

        loss_arch = loss_arch_filter + loss_arch_layer + loss_arch_trans
        # self.scalar_summary("loss_arch",loss_arch)
        total_loss = (self.bbox_loss_coef * loss_bbox + self.xiou_loss_coef * loss_xiou +
                      self.arch_loss_coef * loss_arch)
        # self.scalar_summary("total_loss",total_loss)

        return total_loss/self.accumulate_step, loss_bbox, loss_xiou, loss_arch


_grad_scale = ops.MultitypeFuncGraph("grad_scale")


@_grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * ops.cast(ops.Reciprocal()(scale), ops.dtype(grad))


class MyTrainOneStepCell(nn.TrainOneStepWithLossScaleCell):
    """
    Network training package class with gradient clip.

    Append an optimizer to the training network after that the construct function
    can be called to create the backward graph.

    Args:
        network (Cell): The training network.
        optimizer (Cell): Optimizer for updating the weights.
        sens (Number): The adjust parameter. Default value is 1.0.
        grad_clip (bool): Whether clip gradients. Default value is False.
    """

    def __init__(self, network, optimizer, scale_sense_value=2**5, grad_clip=False, clip_max_norm=0.15):
        loss_scale_manager = nn.DynamicLossScaleUpdateCell(loss_scale_value=scale_sense_value,
                                                           scale_factor=2,
                                                           scale_window=10000)
        super(MyTrainOneStepCell, self).__init__(network, optimizer, loss_scale_manager)
        self.grad_clip = grad_clip
        self.clip_max_norm = clip_max_norm
        # self.scale_sense = nn.DynamicLossScaleUpdateCell(loss_scale_value=scale_sense_value, scale_factor=2, scale_window=1000)

    def construct(self, image, bbox, mask, text):
        # 大多数是基类的属性和方法，详情参考对应基类API
        weights = self.weights
        loss = self.network(image, bbox, mask, text)
        scaling_sens = self.scale_sense
        # 启动浮点溢出检测。创建并清除溢出检测状态
        status, scaling_sens = self.start_overflow_check(loss, scaling_sens)
        # 给梯度乘一个scale防止梯度下溢
        scaling_sens_filled = ops.ones_like(loss) * ops.cast(scaling_sens, ops.dtype(loss))
        # grads = ops.grad(self.network, grad_position=None, weights=weights, has_aux=True)()
        grads = self.grad(self.network, weights)(image, bbox, mask, text, scaling_sens_filled)
        # 给求得的梯度除回scale计算真实的梯度值
        grads = self.hyper_map(ops.partial(_grad_scale, scaling_sens), grads)  # grads here is a tuple
        # print("grads:")
        # print(grads)
        # print("=====grads======")
        # 梯度裁剪
        if self.clip_max_norm > 0:
            grads = ops.clip_by_global_norm(grads, self.clip_max_norm)

        # 梯度聚合
        grads = self.grad_reducer(grads)

        # 获取浮点溢出状态
        cond = self.get_overflow_status(status, grads)
        # 动态loss scale时根据溢出状态计算损失缩放系数用于监控数值,如果是静态scale就什么意义
        overflow = self.process_loss_scale(cond)
        # 如果没有溢出，执行优化器更新参数
        # conv_weights_pre=self.network.network.visual_encoder.backbone.body.layer4.layers[0].conv1.weight
        # dense_weights_pre=self.network.network.bbox_pred.layers[4].weight
        if not overflow:
            self.optimizer(grads)
        # conv_weights_after=self.network.network.visual_encoder.backbone.body.layer4.layers[0].conv1.weight
        # dense_weights_after=self.network.network.bbox_pred.layers[4].weight
        # conv_weights_dif=conv_weights_after-conv_weights_pre
        # dense_weights_dif=dense_weights_after-dense_weights_pre
        return loss, cond, scaling_sens.value(), overflow


class MyTrainOneStepCellV2(nn.Cell):

    def __init__(self, network: nn.Cell, optimizer, clip_max_norm=0.15):
        super(MyTrainOneStepCellV2, self).__init__()
        self.network = network
        self.optimizer = optimizer
        self.clip_max_norm = clip_max_norm
        self.weights = self.optimizer.parameters
        self.reducer_flag = False
        self.grad_reducer = F.identity
        self.parallel_mode = _get_parallel_mode()
        self.reducer_flag = self.parallel_mode in (ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL) or \
                            _is_pynative_parallel()
        if self.reducer_flag:
            self.mean = _get_gradients_mean()
            self.degree = _get_device_num()
            from mindspore.communication.management import GlobalComm
            group = GlobalComm.WORLD_COMM_GROUP
            if isinstance(self.optimizer, (nn.AdaSumByGradWrapCell, nn.AdaSumByDeltaWeightWrapCell)):
                from mindspore.communication.management import get_group_size, create_group, get_rank
                group_number = get_group_size() // 8
                self.degree = int(self.degree / group_number)
                group_list = [list(range(x * self.degree, (x + 1) * self.degree)) for x in range(group_number)]
                current_index = get_rank() // 8
                server_group_name = "allreduce_" + str(current_index)
                create_group(server_group_name, group_list[current_index])
                group = server_group_name
            self.grad_reducer = DistributedGradReducer(self.weights, self.mean, self.degree, group=group)

    def construct(self, image, bbox, mask, text):
        weights = self.weights
        # total_loss, loss_bbox, loss_xiou, loss_arch = self.network(image, bbox, mask, text)
        (total_loss, loss_bbox, loss_xiou, loss_arch, filter_sigma, layer_sigma, trans_sigma, active_layer_num,
         active_trans_num), grads = mindspore.value_and_grad(self.network,
                                                             grad_position=None,
                                                             weights=weights,
                                                             has_aux=True)(image, bbox, mask, text)
        if self.clip_max_norm > 0:
            grads = ops.clip_by_global_norm(grads, self.clip_max_norm)
        grads = self.grad_reducer(grads)
        self.optimizer(grads)
        return total_loss, loss_bbox, loss_xiou, loss_arch, filter_sigma, layer_sigma, trans_sigma, active_layer_num, active_trans_num


class MyTrainOneStepCellAccumulator(nn.TrainOneStepWithLossScaleCell):
    """
    Network training package class with gradient clip.

    Append an optimizer to the training network after that the construct function
    can be called to create the backward graph.

    Args:
        network (Cell): The training network.
        optimizer (Cell): Optimizer for updating the weights.
        sens (Number): The adjust parameter. Default value is 1.0.
        grad_clip (bool): Whether clip gradients. Default value is False.
    """

    def __init__(self, network, optimizer, scale_sense_value=2, clip_max_norm=0.15, accumulate_step=4):
        loss_scale_manager = nn.DynamicLossScaleUpdateCell(loss_scale_value=scale_sense_value,
                                                           scale_factor=2,
                                                           scale_window=1000000)
        super().__init__(network, optimizer, loss_scale_manager)
        self.clip_max_norm = clip_max_norm
        self.inner_grads = optimizer.parameters.clone(prefix="accumulate_", init="zeros")
        self.zeros = optimizer.parameters.clone(prefix="zeros_", init="zeros")
        self.counter = Parameter(Tensor(0, ms.int32), 'counter_')
        assert accumulate_step > 0
        self.accumulate_step = Parameter(Tensor(accumulate_step, ms.int32))
        self.map = ops.HyperMap()
        # self.scale_sense = nn.DynamicLossScaleUpdateCell(loss_scale_value=scale_sense_value, scale_factor=2, scale_window=1000)

    def construct(self, image, bbox, mask, text):
        # 大多数是基类的属性和方法，详情参考对应基类API
        weights = self.weights
        loss = self.network(image, bbox, mask, text)
        if not self.training:
            return loss
        scaling_sens = self.scale_sense
        # 启动浮点溢出检测。创建并清除溢出检测状态
        status, scaling_sens = self.start_overflow_check(loss, scaling_sens)
        # 给梯度乘一个scale防止梯度下溢
        scaling_sens_filled = ops.ones_like(loss) * ops.cast(scaling_sens, ops.dtype(loss))
        grads = self.grad(self.network, weights)(image, bbox, mask, text, scaling_sens_filled)
        # 给求得的梯度除回scale计算真实的梯度值
        grads = self.hyper_map(ops.partial(_grad_scale, scaling_sens), grads)  # grads here is a tuple
        # 获取浮点溢出状态
        cond = self.get_overflow_status(status, grads)
        # 动态loss scale时根据溢出状态计算损失缩放系数用于监控数值,如果是静态scale就什么意义
        overflow = self.process_loss_scale(cond)
        # 累积梯度，先累积，再裁剪
        self.map(ops.partial(ops.assign_add), self.inner_grads, grads)
        ops.assign_add(self.counter, Tensor(1, ms.int32))
        # 到达累积步数
        if self.counter % self.accumulate_step == 0:
            # 梯度裁剪
            grads = ops.clip_by_global_norm(self.inner_grads, self.clip_max_norm)
            # 梯度聚合
            grads = self.grad_reducer(grads)
            # inner_grads 清零
            self.map(ops.partial(ops.assign), self.inner_grads, self.zeros)

            # 如果没有溢出，执行优化器更新参数
            if not overflow:
                self.optimizer(grads)
        return loss, cond, scaling_sens.value(), overflow


class MyTrainOneStepNoScaleAccumulator(nn.Cell):

    def __init__(self, network: nn.Cell, optimizer:nn.Optimizer, clip_max_norm=0.15, accumulate_step=4):
        super().__init__()
        self.network = network
        self.accumulate_step = accumulate_step
        self.optimizer = optimizer
        self.clip_max_norm = clip_max_norm
        self.weights = self.optimizer.parameters
        self.reducer_flag = False
        self.grad_reducer = F.identity
        self.parallel_mode = _get_parallel_mode()
        self.reducer_flag = self.parallel_mode in (ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL) or \
                            _is_pynative_parallel()
        if self.reducer_flag:
            self.mean = _get_gradients_mean()
            self.degree = _get_device_num()
            from mindspore.communication.management import GlobalComm
            group = GlobalComm.WORLD_COMM_GROUP
            if isinstance(self.optimizer, (nn.AdaSumByGradWrapCell, nn.AdaSumByDeltaWeightWrapCell)):
                from mindspore.communication.management import get_group_size, create_group, get_rank
                group_number = get_group_size() // 8
                self.degree = int(self.degree / group_number)
                group_list = [list(range(x * self.degree, (x + 1) * self.degree)) for x in range(group_number)]
                current_index = get_rank() // 8
                server_group_name = "allreduce_" + str(current_index)
                create_group(server_group_name, group_list[current_index])
                group = server_group_name
            self.grad_reducer = DistributedGradReducer(self.weights, self.mean, self.degree, group=group)

        # 梯度累积
        self.inner_grads = optimizer.parameters.clone(prefix="accumulate_", init="zeros")
        self.zeros = optimizer.parameters.clone(prefix="zeros_", init="zeros")
        self.counter = Parameter(Tensor(0, ms.int32), 'counter_')
        assert accumulate_step > 0
        self.accumulate_step = Tensor(accumulate_step, ms.int32)
        self.map = ops.HyperMap()

    def construct(self, image, bbox, mask, text):
        if not self.training:
            return self.network(image, bbox, mask, text)
        weights = self.weights
        # total_loss, loss_bbox, loss_xiou, loss_arch = self.network(image, bbox, mask, text)
        (total_loss, loss_bbox, loss_xiou, loss_arch), grads = mindspore.value_and_grad(self.network,
                                                                                        grad_position=None,
                                                                                        weights=weights,
                                                                                        has_aux=True)(image, bbox, mask,
                                                                                                      text)
        # 累积梯度，先累积，再裁剪, 之前的loss/4了，所以这里每一份梯度都是1/4
        self.map(ops.partial(ops.assign_add), self.inner_grads, grads)
        ops.assign_add(self.counter, Tensor(1, ms.int32))
        lr=self.optimizer.get_lr()[0]
        # 到达累积步数
        if self.counter % self.accumulate_step == 0:
            # 梯度裁剪
            grads = ops.clip_by_global_norm(self.inner_grads, self.clip_max_norm)
            # 梯度聚合
            grads = self.grad_reducer(grads)
            # inner_grads 清零
            self.map(ops.partial(ops.assign), self.inner_grads, self.zeros)
            # 每accumulate_step更新一次
            self.optimizer(grads)
        return total_loss * self.accumulate_step, loss_bbox, loss_xiou, loss_arch, lr  


expand_dims = P.ExpandDims().add_prim_attr("grad_scale", True)

get_square_sum = C.MultitypeFuncGraph("get_square_sum")


@get_square_sum.register("Tensor")
def _get_square_sum(x):
    norm = P.ReduceSum(False)(F.square(x), ())
    norm = expand_dims(F.cast(norm, mindspore.float16), 0)
    return norm


apply_global_norm = C.MultitypeFuncGraph("apply_global_norm")


@apply_global_norm.register("Tensor", "Tensor", "Tensor")
def _apply_global_norm(clip_norm, global_norm, x):
    x_dtype = F.dtype(x)
    x = x * clip_norm / global_norm
    x = F.cast(x, x_dtype)
    return x


class ClipGradNorm(nn.Cell):

    def __init__(self, clip_norm=0.15):
        super(ClipGradNorm, self).__init__()
        self.clip_norm = Tensor([clip_norm], mindspore.float16)
        self.hyper_map = C.HyperMap()
        self.greater_equal = P.GreaterEqual()

    def construct(self, x):

        square_sum = self.hyper_map(get_square_sum, x)
        global_norm = F.sqrt(F.addn(square_sum))
        cond = self.greater_equal(global_norm, self.clip_norm)
        global_norm = F.select(cond, global_norm, self.clip_norm)
        clip_x = self.hyper_map(F.partial(apply_global_norm, self.clip_norm, global_norm), x)

        return clip_x


# def clip_grad_norm_(x,clip_norm=0.15):
#     # x is a tuple
#     print(type(x))
#     print(len(x))

#     for g in x:

#     global_norm=Tensor([global_norm],mindspore.float16)
#     clip_norm=Tensor([clip_norm],mindspore.float16)

#     if float(global_norm)<float(clip_norm):
#         global_norm=clip_norm

#     clip_x=x*clip_norm/global_norm
#     clip_x=ops.cast(clip_x,mindspore.float16)

#     return clip_x
