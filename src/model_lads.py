from src.resnet import _bn1d, _fc
from mindformers.models.bert.bert import BertNetwork
import mindspore as ms
import mindspore.nn as nn
from src.xenv import console, parse_args
import src.resnet as resnet
import os
from src.backbone import build_backbone
from src.utils import calc_resnet_flops, pt_flatten, pt_topk, count_useful_length
from src.transformer import TransformerEncoder, TransformerEncoderLayer, TransformerEncoderLayerWithHook
import mindspore.ops as ops
import mindspore.numpy as msnp
from mindspore import Tensor
from src.init_weights import KaimingUniform
from mindspore.common.initializer import initializer, HeNormal, HeUniform, One, Zero, XavierUniform, XavierNormal
from collections import OrderedDict
from src.pytorch_bert import BertConfig, BertModel

args = parse_args()

WORD_ATTN = [None]


class ReferNet(nn.Cell):
    '''Baseline network for language-guided dynamic sub-net selection'''

    def __init__(self,
                 batch_size,
                 normalize_before=False,
                 dropout=0.1,
                 use_selector=True,
                 token_max_len=20,
                 use_dilation=False,
                 token_select='all',
                 cnn_model='resnet50-unc',
                 bert_model='bert_base_uncased',
                 text_encoder_layer_num=12,
                 task_encoder_layer_num=6,
                 task_encoder_dim=256,
                 task_ffn_dim=2048,
                 task_encoder_head=8,
                 trainable_layers=3,
                 norm_layer=None,
                 selector_bias = 3.0):
        super(ReferNet, self).__init__()
        self.cast_type = ms.float16
        self.N_task_token = -1
        self.use_dilation = use_dilation
        self.task_encoder_depth = task_encoder_layer_num
        self.use_selector = use_selector

        self.flatten_function = pt_flatten

        # if norm_layer=='bn2d':
        #     norm_layer=nn.BatchNorm2d
        # elif norm_layer=='sync_bn2d':
        #     norm_layer=nn.SyncBatchNorm
        # self.text_encoder = self.get_model_from_pretrained(bert_model_root="./checkpoint_download/bert",
        #                                                    bert_model=bert_model,
        #                                                    batch_size=batch_size,
        #                                                    token_max_len=token_max_len,
        #                                                    text_encoder_layer_num=text_encoder_layer_num)
        self.text_encoder = self.get_bert()
        # build backbone
        self.visual_encoder = build_backbone(cnn_model, task_encoder_dim, use_dilation, trainable_layers, use_selector=use_selector, norm_layer=norm_layer)
        # task_encoder_norm=nn.LayerNorm(task_encoder_dim) if normalize_before else None
        self.task_encoder = TransformerEncoder(task_encoder_dim,
                                               task_encoder_head,
                                               task_ffn_dim,
                                               dropout,
                                               activation='relu',
                                               num_layers=task_encoder_layer_num,
                                               normalize_before=normalize_before,
                                               use_selector=use_selector)
        # selector
        self.selector = Selector(
            self.bert_model_config.hidden_size,
            text_encoder_layer_num,
            visual_supernet=self.visual_encoder.backbone.body,
            trans_supernet=self.task_encoder,
            reduction='cls',
            selector_bias=selector_bias
        ) if use_selector else None

        # projection layers
        self.visual_proj = nn.Dense(self.visual_encoder.num_channels, task_encoder_dim, has_bias=True, weight_init=initializer(HeUniform(), (task_encoder_dim, self.visual_encoder.num_channels)))
        self.text_proj = nn.SequentialCell(nn.Dense(self.bert_model_config.hidden_size, task_encoder_dim, has_bias=True, weight_init=KaimingUniform()), nn.LayerNorm((task_encoder_dim, )))
        self.bbox_pred = MLP(task_encoder_dim, task_encoder_dim, 4, 3)
        # self.reset_trans_parameters()

    def get_bert(self):
        bert_config = BertConfig(num_hidden_layers=6)
        self.bert_model_config = bert_config
        bert = BertModel(bert_config)
        param_list = ms.load_checkpoint("./weights/pretrain-weights/bert_torch.ckpt")
        print("======Loading pretrained weight into bert======")
        missing_list = ms.load_param_into_net(bert, param_list)
        print("======Missing weights are: ======")
        print(missing_list)
        print("====================================")
        return bert

    def construct(self, images, bbox, masks, texts):
        # calculate expression feature
        # [input_ids,token_type_ids,attention_mask]
        input_ids = texts[0]  # shape = [batch_size, seq_len]
        token_type_ids = texts[1]  # shape = [batch_size, seq_len]
        attention_mask = texts[2]  # shape = [batch_size, seq_len]
        text_mask = ops.ne(attention_mask, 1)
        text_output = self.text_encoder(input_ids, attention_mask, token_type_ids)  # shape = [batch_size, seq_len, 768]

        last_hidden_state = text_output
        # all_hidden_states = text_output[1]

        # m_type=self.text_proj[0].weight.dtype
        text_feat_ = ops.transpose(last_hidden_state, (1, 0, 2))
        text_feat = self.text_proj(text_feat_)

        if self.selector is not None:
            filter_layer_selection, attn_ffn_selection = self.selector(last_hidden_state, text_mask)
        else:
            filter_layer_selection, attn_ffn_selection = None, None

        # calculate image feature
        # if self.training:
        #     images = images.astype(self.cast_type)
        visual_output, img_pos_embed = self.visual_encoder([images, masks], filter_layer_selection)
        visual_feat_raw, visual_mask = visual_output[0], visual_output[1]
        visual_feat = self.visual_proj(self.flatten_function(visual_feat_raw, 2).transpose((2, 0, 1)))
        img_pos_embed = self.flatten_function(img_pos_embed, 2).transpose((2, 0, 1))
        visual_mask = ~self.flatten_function(visual_mask, 1)

        # calculate image-language feature
        text_visual_feat = ops.concat([text_feat, visual_feat.astype(text_feat.dtype)], axis=0)
        text_visual_pos = ops.concat([ops.zeros_like(text_feat), img_pos_embed.astype(text_feat.dtype)], axis=0)

        text_visual_mask = ops.concat([text_mask, visual_mask], axis=1)
        text_visual_feat = self.task_encoder(text_visual_feat, text_visual_mask, text_visual_pos, attn_ffn_selection)
        prob_box = ops.sigmoid(self.bbox_pred(text_visual_feat[0]))
        return prob_box, filter_layer_selection, attn_ffn_selection

    # def reset_trans_parameters(self):
    #     return


class Selector(nn.Cell):
    """
    Dynamic subnet selector for the whole model
    """

    def __init__(self, input_dim: int, input_num: int, visual_supernet: nn.Cell, trans_supernet, inner_dim=32, reduction='cls', selector_bias = 3.0):
        super(Selector, self).__init__()

        assert reduction in ('cls', 'mean', 'max'), f'Not supported reduction({reduction})'
        self.cast_type = ms.float16
        self.reduction = reduction
        self.rand_function = ops.UniformReal()
        self.flatten_function = pt_flatten

        self.bmm = ops.BatchMatMul()
        # ?????????????????????
        self.proj_filter = nn.CellList()
        self.proj_filter_dict = dict()
        self.proj_layer = nn.CellList()
        self.proj_layer_dict = dict()
        self.proj_attn = nn.CellList()
        self.proj_attn_dict = dict()
        self.proj_ffn = nn.CellList()
        self.proj_ffn_dict = dict()

        self.channels = []

        # in GRAPH mode, hooks can not be used.
        # def create_hooks(key, out_channel):
        #     def filter_select_hook(module, input, output):
        #         select, sigma, mix_gates = self.selections[key][:3]
        #         mix_gates = mix_gates.view(-1, out_channel, 1, 1)
        #         output = output*mix_gates
        #         return output

        #     def layer_select_hook(module, input, output):
        #         select, sigma, mix_gates = self.selections[key][-3:]
        #         mix_gates = mix_gates.view(-1, 1, 1, 1)
        #         output = output*mix_gates
        #         return output

        #     return filter_select_hook, layer_select_hook
        index = 0
        for n, m in visual_supernet.cells_and_names():
            n = n.replace('.', '_')
            if (isinstance(m, resnet.Bottleneck) or isinstance(m, resnet.BottleneckWithHook)) and m.conv2.stride[0] == 1:
                c0 = m.conv1.in_channels
                c1 = m.conv2.in_channels
                c2 = m.conv3.in_channels
                c3 = m.conv3.out_channels
                self.channels.append([c0, c1, c2, c3])
                out_channel = m.conv2.out_channels

                net_filter = nn.Dense(inner_dim, out_channel, weight_init=initializer(XavierUniform(), (out_channel, inner_dim)), bias_init=selector_bias)
                self.proj_filter.append(net_filter)
                self.proj_filter_dict[n] = index

                net_layer = nn.Dense(inner_dim, 1, weight_init=initializer(XavierUniform(), (1, inner_dim)), bias_init=selector_bias)
                self.proj_layer.append(net_layer)
                self.proj_layer_dict[n] = index

                # register_forward_hook 在GRAPH下自动失效
                # filter_hook, layer_hook = create_hooks(n, out_channel)
                # m.bn2.register_forward_hook(filter_hook)
                # m.bn3.register_forward_hook(layer_hook)
                index += 1
        index = 0
        for n, m in trans_supernet.cells_and_names():
            n = n.replace('.', '_')
            if isinstance(m, TransformerEncoderLayer) or isinstance(m, TransformerEncoderLayerWithHook):
                net_attn = nn.Dense(inner_dim, 1, weight_init=initializer(XavierUniform(), (1, inner_dim)), bias_init=selector_bias)
                self.proj_attn.append(net_attn)
                self.proj_attn_dict[n] = index

                net_ffn = nn.Dense(inner_dim, 1, weight_init=initializer(XavierUniform(), (1, inner_dim)), bias_init=selector_bias)
                self.proj_ffn.append(net_ffn)
                self.proj_ffn_dict[n] = index
                # filter_hook, layer_hook = create_hooks(n, out_channel)
                # m.bn2.register_forward_hook(filter_hook)
                # m.bn3.register_forward_hook(layer_hook)
                index += 1

        self.num_layer = num_layer = len(self.proj_layer) + len(self.proj_attn)
        self.att_map = nn.Dense(input_dim, num_layer, weight_init=initializer(XavierUniform(), (num_layer, input_dim)), bias_init=0)

        # self.proj_pre=nn.SequentialCell(
        #     nn.BatchNorm1d(num_layer*input_dim),
        #     nn.GELU(),
        #     nn.Conv1d(num_layer*input_dim,num_layer*inner_dim,1,group=num_layer,weight_init=KaimingUniform(),has_bias=True),
        #     nn.BatchNorm1d(num_layer*inner_dim),
        #     nn.GELU(),
        #     nn.Conv1d(num_layer*inner_dim,num_layer*inner_dim*2,1,group=num_layer,weight_init=KaimingUniform(),has_bias=True),
        #     nn.BatchNorm1d(num_layer*inner_dim*2),
        #     nn.GELU()
        # )
        self.proj_pre = ProjPre(num_layer, input_dim, inner_dim)

        console.print(f'Selector:{len(self.proj_layer)}+{len(self.proj_attn)} layers in total.', style='green')

    def construct(self, text_feats, text_pad_masks):
        filter_layer_select = []
        attn_ffn_select = []

        att_mask = ops.zeros(text_pad_masks.shape, text_feats.dtype)
        # att_mask = att_mask.masked_fill(text_pad_masks, float("-inf"))
        att_mask = ops.masked_fill(att_mask, text_pad_masks, float("-inf"))
        att_mask = ops.stop_gradient(att_mask)

        # ==============
        # assistant = ops.ones(text_pad_masks.shape, text_feats.dtype)
        # assistant = assistant * float("-inf")
        # att_mask = ms.numpy.where(text_pad_masks, assistant, att_mask)
        # ==============

        att_mask = ops.expand_dims(att_mask, -1)
        att_map = self.att_map(text_feats)
        att_mask = ops.cast(att_mask, att_map.dtype)
        word_att = ops.softmax(att_map + att_mask, 1)

        inner_feats = self.bmm(ops.transpose(word_att, (0, 2, 1)), text_feats)
        # ==========
        # inner_feats = pt_flatten(inner_feats,1)
        # inner_feats = ops.expand_dims(inner_feats,-1)
        # ==========
        # inner_feats = self.proj_pre(ops.expand_dims(pt_flatten(inner_feats,1),-1))
        inner_feats = self.proj_pre(self.flatten_function(inner_feats, 1))
        inner_feats = ops.split(inner_feats, axis=1, output_num=self.num_layer * 2)

        global WORD_ATTN
        WORD_ATTN[0] = word_att

        for i in range(len(self.proj_layer)):
            filter_feat = inner_feats[2 * i]
            layer_feat = inner_feats[2 * i + 1]

            # filter_selection
            filter_logits = self.proj_filter[i](filter_feat)
            logistic_noise = -ops.log(1 / self.rand_function(filter_logits.shape) - 1) if self.training else 0
            filter_sigma = ops.sigmoid(filter_logits + logistic_noise)
            # ???data
            filter_sigma_data = filter_sigma.copy()
            filter_sigma_data = ops.stop_gradient(filter_sigma_data)
            # filter_sigma_data.set_const_arg(True)
            filter_select = ops.cast(ops.ge(filter_sigma, 0.5), filter_sigma.dtype) - filter_sigma_data + filter_sigma
            if self.training:
                mix_indices = self.rand_function(filter_select.shape) < 0.5
                filter_mix_gates = ms.numpy.where(mix_indices, filter_sigma, filter_select)
            else:
                filter_mix_gates = filter_select

            # layer selection
            layer_logits = self.proj_layer[i](layer_feat)
            logistic_noise = -ops.log(1 / self.rand_function(layer_logits.shape) - 1) if self.training else 0
            layer_sigma = ops.sigmoid(layer_logits + logistic_noise)
            # ???data
            layer_sigma_data = layer_sigma.copy()
            layer_sigma_data = ops.stop_gradient(layer_sigma_data)
            layer_select = ops.cast(ops.ge(layer_sigma, 0.5), layer_sigma.dtype) - layer_sigma_data + layer_sigma

            if self.training:
                mix_indices = self.rand_function(layer_select.shape) < 0.5
                layer_mix_gates = ms.numpy.where(mix_indices, layer_sigma, layer_select)
            else:
                layer_mix_gates = layer_select

            filter_layer_select.append([filter_select, filter_sigma, filter_mix_gates, layer_select, layer_sigma, layer_mix_gates])

        cnn_num = 2 * len(self.proj_layer)
        for i in range(len(self.proj_attn)):
            attn_feat, ffn_feat = inner_feats[2 * i + cnn_num], inner_feats[2 * i + 1 + cnn_num]

            attn_logits = self.proj_attn[i](attn_feat)
            logistic_noise = -ops.log(1 / self.rand_function(attn_logits.shape) - 1) if self.training else 0
            attn_sigma = ops.sigmoid(attn_logits + logistic_noise)
            attn_sigma_data = attn_sigma.copy()
            attn_sigma_data = ops.stop_gradient(attn_sigma_data)
            attn_select = ops.cast(ops.ge(attn_sigma, 0.5), attn_sigma.dtype) - attn_sigma_data + attn_sigma

            if self.training:
                mix_indices = self.rand_function(attn_select.shape) < 0.5
                attn_mix_gates = ms.numpy.where(mix_indices, attn_sigma, attn_select)
            else:
                attn_mix_gates = attn_select

            ffn_logits = self.proj_ffn[i](ffn_feat)
            logistic_noise = -ops.log(1 / self.rand_function(ffn_logits.shape) - 1) if self.training else 0
            ffn_sigma = ops.sigmoid(ffn_logits + logistic_noise)
            ffn_sigma_data = ffn_sigma.copy()
            ffn_sigma_data = ops.stop_gradient(ffn_sigma_data)
            ffn_select = ops.cast(ops.ge(ffn_sigma, 0.5), ffn_sigma.dtype) - ffn_sigma_data + ffn_sigma
            if self.training:
                mix_indices = self.rand_function(ffn_select.shape) < 0.5
                ffn_mix_gates = ms.numpy.where(mix_indices, ffn_sigma, ffn_select)
            else:
                ffn_mix_gates = ffn_select
            attn_ffn_select.append([attn_select, attn_sigma, attn_mix_gates, ffn_select, ffn_sigma, ffn_mix_gates])

        return filter_layer_select, attn_ffn_select


class MLP(nn.Cell):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        input_channel = [input_dim] + [hidden_dim] * (num_layers - 1)
        output_channel = [hidden_dim] * (num_layers - 1) + [output_dim]
        layers = []
        self.cast_type = ms.float32
        for i in range(num_layers - 1):
            layers.append(nn.Dense(input_channel[i], output_channel[i], weight_init=initializer(XavierUniform(), (output_channel[i], input_channel[i]))).to_float(self.cast_type))
            layers.append(nn.ReLU())
        layers.append(nn.Dense(input_channel[-1], output_channel[-1], weight_init=initializer(XavierUniform(), (output_channel[-1], input_channel[-1]))).to_float(self.cast_type))
        self.layers = nn.SequentialCell(*layers)

    def construct(self, x):
        # x = x.astype(self.cast_type)
        return self.layers(x)


class ProjPre(nn.Cell):

    def __init__(self, num_layer, input_dim, inner_dim):
        super().__init__()
        self.flatten_function = pt_flatten
        self.bn1d_0 = nn.BatchNorm1d(num_layer * input_dim)
        self.gelu_0 = nn.GELU()
        self.conv1d_0 = nn.Conv1d(num_layer * input_dim, num_layer * inner_dim, 1, group=num_layer, weight_init=KaimingUniform(), pad_mode='valid', has_bias=True)
        self.bn1d_1 = nn.BatchNorm1d(num_layer * inner_dim)
        self.gelu_1 = nn.GELU()
        self.conv1d_1 = nn.Conv1d(num_layer * inner_dim, num_layer * inner_dim * 2, 1, group=num_layer, weight_init=KaimingUniform(), pad_mode='valid', has_bias=True)
        self.bn1d_2 = nn.BatchNorm1d(num_layer * inner_dim * 2)
        self.gelu_2 = nn.GELU()

    def construct(self, x):
        output = self.bn1d_0(x)
        output = self.gelu_0(output)
        output = ops.expand_dims(output, -1)
        output = self.conv1d_0(output)
        output = self.flatten_function(output, 1)

        output = self.bn1d_1(output)
        output = self.gelu_1(output)
        output = ops.expand_dims(output, -1)
        output = self.conv1d_1(output)
        output = self.flatten_function(output, 1)

        output = self.bn1d_2(output)
        output = self.gelu_2(output)
        return output


def test0():
    ms.set_context(device_target='GPU', mode=ms.PYNATIVE_MODE)
    ms.set_auto_parallel_context(device_num=2, parallel_mode="data_parallel")
    net = ReferNet(10, use_selector=True, cnn_model='resnet50-unc')
    for name, params in net.parameters_and_names():
        if name.startswith('selector'):
            print(name)
            print(params.requires_grad)
    pass


if __name__ == '__main__':
    test0()