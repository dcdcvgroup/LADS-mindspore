"""evaluate"""
import matplotlib.pyplot as plt
import tqdm as tqdm
import numpy as np
import mindspore.nn as nn
from collections import OrderedDict
from dataset import create_ref_dataset
from xenv import parse_args
from pprint import pprint
import mindspore as ms
import mindspore.ops as ops
from mindspore import Tensor
from model_new import ReferNet
from src.loss import NetWithLossV2, MyTrainOneStepCellAccumulator
from utils import box_ious
from mindspore.communication import init, get_rank, get_group_size
import os
from mindspore.amp import auto_mixed_precision
import datetime

args = parse_args()
ms.set_seed(args.seed)


def set_parameter():
    # settings
    ms.set_context(max_call_depth=100000000)
    target = args.device_target
    if target == "CPU":
        args.run_distribute = False

    if args.mode_name == "GRAPH":
        ms.set_context(mode=ms.GRAPH_MODE, device_target=target, save_graphs=False, save_graphs_path="./graph_save/")
    else:
        ms.set_context(mode=ms.PYNATIVE_MODE, device_target=target, save_graphs=False, enable_graph_kernel=True)

    if args.run_distribute:
        if target == "Ascend":
            device_id = int(os.getenv('DEVICE_ID'))
            ms.set_context(device_id=device_id)
            ms.set_auto_parallel_context(device_num=args.device_num, parallel_mode=ms.ParallelMode.DATA_PARALLEL, gradients_mean=True)
            ms.parallel.set_algo_parameters(elementwise_op_strategy_follow=True)
            ms.set_auto_parallel_context(all_reduce_fusion_config=args.all_reduce_fusion_config)
            init()
        # GPU target
        else:
            init("nccl")
            ms.reset_auto_parallel_context()
            device_num = get_group_size()
            args.device_num = device_num
            ms.set_auto_parallel_context(device_num=device_num, parallel_mode=ms.ParallelMode.DATA_PARALLEL, parameter_broadcast=True, gradients_mean=True)


class EvalNet(nn.Cell):

    def __init__(self, network):
        super().__init__()
        self.network = network

    def construct(self, image, bbox, mask, text):
        return self.network(image, bbox, mask, text)


def eval(ckpt_path):
    if args.mode_name == "PYNATIVE":
        args.run_distribute = False

    set_parameter()
    if args.run_distribute == True:
        local_rank = get_rank()
    else:
        local_rank = 0
    time = datetime.datetime.now()
    epoch_idx = os.path.basename(ckpt_path)[:-5]
    L = len(os.path.basename(ckpt_path))
    time = ckpt_path[8:-L - 1]
    print(time)
    ckpt_root = os.path.dirname(ckpt_path)
    log_root = os.path.join(ckpt_root, 'eval_log')
    log_root = f'eval_log/{args.comments}_{time}'
    if local_rank == 0:
        if not os.path.exists(log_root):
            os.mkdir(log_root)
        log_root = os.path.join(log_root, f"{epoch_idx}")
        if not os.path.exists(log_root):
            os.mkdir(log_root)
            print(f"MKDIR {log_root}")
    # local_rank = get_rank()
    # if local_rank == 0:
    #     pprint(vars(args))

    all_reduce = ops.AllReduce()
    eval_dataset_list = create_ref_dataset(args, do_train=False, batch_size=args.batch_size, distribute=args.run_distribute)

    param_dict = ms.load_checkpoint(ckpt_path)
    # new_dict = dict()
    # for k, v in param_dict.items():
    #     # if k.startswith("optimizer"):
    #     #     # k = k[10:]
    #     #     continue
    #     # else:
    #     #     k = k[8:]
    #     # new_dict[k] = v
    #     if k.startswith("network"):
    #         new_dict[k]=v

    model = ReferNet(args.batch_size,
                     args.pre_norm,
                     args.dropout,
                     args.use_selector,
                     args.max_len,
                     args.use_dilation,
                     args.token_select,
                     args.cnn_model,
                     args.bert_model,
                     args.text_encoder_layer,
                     args.task_encoder_layer,
                     is_training=False)
    missing_list = ms.load_param_into_net(model, parameter_dict=param_dict)
    print("missing list:")
    print(missing_list)
    model.set_train(False)
    # model = NetWithLossV2(model, args.xiou_loss_type, args.bbox_loss_coef, args.xiou_loss_coef, args.arch_loss_coef)
    # params_groups = [{
    #     'params': [p for n, p in model.parameters_and_names() if n.startswith('network.visual_encoder') and p.requires_grad],
    #     'lr': args.lr_visual
    # }, {
    #     'params': [p for n, p in model.parameters_and_names() if n.startswith('network.text_encoder') and p.requires_grad],
    #     'lr': args.lr_lang
    # }, {
    #     'params': [p for n, p in model.parameters_and_names() if not n.startswith(('network.visual_encoder', 'network.text_encoder', 'network.selector')) and p.requires_grad],
    #     'lr': args.lr_base
    # }]

    # if args.use_selector:
    #     params_groups.append({'params': [p for n, p in model.parameters_and_names() if n.startswith('network.selector') and p.requires_grad], 'lr': args.lr_base, 'weight_decay': 0.0})

    # # define optimizer
    # optimizer = nn.AdamWeightDecay(params=params_groups, weight_decay=args.weight_decay)

    # model = MyTrainOneStepCellAccumulator(model, optimizer)
    # model=EvalNet(model)

    # model.to_float(ms.float16)
    # model.selector.proj_pre.bn1d_0.to_float(ms.float32)
    # model.selector.proj_pre.bn1d_1.to_float(ms.float32)
    # model.selector.proj_pre.bn1d_2.to_float(ms.float32)

    model.set_train(False)
    channel_states = {split: OrderedDict() for split in args.eval_splits}
    layer_states = {split: OrderedDict() for split in args.eval_splits}
    attn_states = {split: OrderedDict() for split in args.eval_splits}
    ffn_states = {split: OrderedDict() for split in args.eval_splits}
    total_layer_num = len(model.selector.proj_layer)
    active_filter_nums = {split: ops.zeros((total_layer_num, 64, 64), ms.float32) for split in args.eval_splits}
    active_layer_nums = {split: ops.zeros(total_layer_num, ms.float32) for split in args.eval_splits}
    total_trans_num = len(model.selector.proj_attn) * 2
    active_trans_nums = {split: ops.zeros(total_trans_num, ms.float32) for split in args.eval_splits}
    layer_index = ms.numpy.arange(1, total_layer_num + 1)
    trans_index = ms.numpy.arange(1, total_trans_num + 1)

    for split in args.eval_splits:
        for i in range(len(model.selector.proj_filter)):
            m = model.selector.proj_filter[i]
            channel_states[split][i] = ops.zeros_like(m.bias)
            layer_states[split][i] = ops.zeros(1, ms.float32)
        for i in range(len(model.selector.proj_attn)):
            attn_states[split][i] = ops.zeros(1, ms.float32)
            ffn_states[split][i] = ops.zeros(1, ms.float32)
    split_correct_total = []

    for i in range(len(eval_dataset_list)):
        split = args.eval_splits[i]
        print(f'Evalaluating {split}')
        eval_set = eval_dataset_list[i]
        eval_loader = eval_set.create_tuple_iterator()
        sent_counter = Tensor([0, 0], ms.float32)
        # loop=tqdm((enumerate(eval_loader)),total=len(enumerate(eval_loader)))
        for step_idx, data in enumerate(eval_loader):
            image = data[0]
            gt_bbox = data[1]
            mask = data[2]
            text = data[3]
            pred_bbox, filter_layer_selection, attn_ffn_selection = model(image, gt_bbox, mask, text)
            iou = box_ious(pred_bbox[:, :4], gt_bbox, 'iou')['iou']
            sent_counter[0] += (iou >= args.iou_threshold).sum()
            sent_counter[1] += image.shape[0]

            for key in channel_states[split].keys():
                channel_states[split][key] += ops.reduce_sum(filter_layer_selection[key][0], axis=0)
                layer_states[split][key] += ops.reduce_sum(filter_layer_selection[key][-3])
            for key in ffn_states[split].keys():
                attn_states[split][key] += ops.reduce_sum(attn_ffn_selection[key][0])
                ffn_states[split][key] += ops.reduce_sum(attn_ffn_selection[key][-3])

            active_layer_nums[split] += (sum([filter_layer_selection[i][-3] for i in range(len(filter_layer_selection))]) == layer_index).sum(0)
            active_trans_nums[split] += (sum([attn_ffn_selection[i][0] for i in range(len(attn_ffn_selection))] + [attn_ffn_selection[i][-3]
                                                                                                                   for i in range(len(attn_ffn_selection))]) == trans_index).sum(0)
        if args.mode_name == "GRAPH" and args.run_distribute:
            for key in channel_states[split].keys():
                all_reduce(channel_states[split][key])
                all_reduce(layer_states[split][key])
            for key in attn_states[split].keys():
                all_reduce(attn_states[split][key])
                all_reduce(ffn_states[split][key])

            all_reduce(sent_counter)
            all_reduce(active_layer_nums[split])
            all_reduce(active_trans_nums[split])
        split_correct_total.append((split, sent_counter[0], sent_counter[1]))
    # if local_rank=0:
    precision = dict()
    for split, correct, total in split_correct_total:
        print(f'correct={correct}, total={total}')
        precision[split] = correct / total
    log_str = ""
    log_str += f"Evaluating {ckpt_path}\n"
    results_dict = dict()
    results_dict['channel_states'] = channel_states
    results_dict['layer_states'] = layer_states
    results_dict['attn_states'] = attn_states
    results_dict['ffn_states'] = ffn_states
    results_dict['active_layer_nums'] = active_layer_nums
    results_dict['active_trans_nums'] = active_trans_nums
    results_dict['precision'] = precision

    # draw img
    if local_rank == 0:
        for i, split in enumerate(args.eval_splits):
            layer_select_ratios = [float(layer_state) / float(split_correct_total[i][2]) for layer_state in layer_states[split].values()]
            plt.figure()
            plt.plot(range(len(layer_select_ratios)), layer_select_ratios, '-o')
            plt.xlabel("Layer Index")
            plt.ylabel("Select Ratio")
            img_path = os.path.join(log_root, f"{split}_layer_select_ratio.jpg")
            plt.savefig(img_path)

            attn_select_ratios = [float(attn_state) / float(split_correct_total[i][2]) for attn_state in attn_states[split].values()]
            plt.figure()
            plt.plot(range(len(attn_select_ratios)), attn_select_ratios, '-o')
            plt.xlabel('Attn Index')
            plt.ylabel("Select Ratio")
            img_path = os.path.join(log_root, f"{split}_attn_select_ratio.jpg")
            plt.savefig(img_path)

            ffn_select_ratios = [float(ffn_state) / float(split_correct_total[i][2]) for ffn_state in ffn_states[split].values()]
            plt.figure()
            plt.plot(range(len(ffn_select_ratios)), ffn_select_ratios, '-o')
            plt.xlabel("Ffn Index")
            plt.ylabel("Select Ration")
            img_path = os.path.join(log_root, f"{split}_ffn_select_ratio.jpg")
            plt.savefig(img_path)

        # log_file
        for split in args.eval_splits:
            log_str += f'Eval split: {split}, precision: {precision[split]}\n'
            print(f'Eval split: {split}, precision: {precision[split]}\n')
        # with open()
        # for i in range(len(args.eval_splits)):
        #     split=
        print(log_str)
        logFile = os.path.join(log_root, f"{ckpt_path[-7:-5]}log.txt")
        # resultsFile = os.path.join(log_root, f"{ckpt_path[-8:-5]}results.npy")
        # np.save(resultsFile, results_dict)
        with open(logFile, 'a') as f:
            f.write(log_str)


def eval_list(ckpt_list: list):
    if args.mode_name == "PYNATIVE":
        args.run_distribute = False
    set_parameter()
    time = datetime.datetime.now()
    all_reduce = ops.AllReduce()
    local_rank = get_rank()
    eval_dataset_list = create_ref_dataset(args, do_train=False, batch_size=args.batch_size)

    # ckpt_path = 'train_checkpoints/07-20_03:24:42-07-20_03:40:00-epoch109.ckpt'
    model = ReferNet(args.batch_size, args.pre_norm, args.dropout, args.use_selector, args.max_len, args.use_dilation, args.token_select, args.cnn_model, args.bert_model, args.text_encoder_layer,
                     args.task_encoder_layer)
    model.set_train(False)

    for ckpt_path in ckpt_list:
        if "7-19" in ckpt_path:
            ckpt_path = "./train_checkpoints/" + ckpt_path
            log_root = f'eval_log/{ckpt_path[-8:-5]}-{time}'
            os.mkdir(log_root)
            param_dict = ms.load_checkpoint(ckpt_path)
            missing_list = ms.load_param_into_net(model, parameter_dict=param_dict)
            print("missing list:")
            print(missing_list)
            channel_states = {split: OrderedDict() for split in args.eval_splits}
            layer_states = {split: OrderedDict() for split in args.eval_splits}
            attn_states = {split: OrderedDict() for split in args.eval_splits}
            ffn_states = {split: OrderedDict() for split in args.eval_splits}
            total_layer_num = len(model.selector.proj_layer)
            active_filter_nums = {split: ops.zeros((total_layer_num, 64, 64), ms.float32) for split in args.eval_splits}
            active_layer_nums = {split: ops.zeros(total_layer_num, ms.float32) for split in args.eval_splits}
            total_trans_num = len(model.selector.proj_attn) * 2
            active_trans_nums = {split: ops.zeros(total_trans_num, ms.float32) for split in args.eval_splits}
            layer_index = ms.numpy.arange(1, total_layer_num + 1)
            trans_index = ms.numpy.arange(1, total_trans_num + 1)

            for split in args.eval_splits:
                for i in range(len(model.selector.proj_filter)):
                    m = model.selector.proj_filter[i]
                    channel_states[split][i] = ops.zeros_like(m.bias)
                    layer_states[split][i] = ops.zeros(1, ms.float32)
                for i in range(len(model.selector.proj_attn)):
                    attn_states[split][i] = ops.zeros(1, ms.float32)
                    ffn_states[split][i] = ops.zeros(1, ms.float32)
            split_correct_total = []

            for i in range(len(eval_dataset_list)):
                split = args.eval_splits[i]
                print(f'Evalaluating {split}')
                eval_set = eval_dataset_list[i]
                eval_loader = eval_set.create_tuple_iterator()
                sent_counter = Tensor([0, 0], ms.float32)
                # loop=tqdm((enumerate(eval_loader)),total=len(enumerate(eval_loader)))
                for step_idx, data in enumerate(eval_loader):
                    image = data[0]
                    gt_bbox = data[1]
                    mask = data[2]
                    text = data[3]
                    pred_bbox, filter_layer_selection, attn_ffn_selection = model(image, gt_bbox, mask, text)
                    iou = box_ious(pred_bbox[:, :4], gt_bbox, 'iou')['iou']
                    sent_counter[0] += (iou >= args.iou_threshold).sum()
                    sent_counter[1] += image.shape[0]

                    for key in channel_states[split].keys():
                        channel_states[split][key] += ops.reduce_sum(filter_layer_selection[key][0], axis=0)
                        layer_states[split][key] += ops.reduce_sum(filter_layer_selection[key][-3])
                    for key in ffn_states[split].keys():
                        attn_states[split][key] += ops.reduce_sum(attn_ffn_selection[key][0])
                        ffn_states[split][key] += ops.reduce_sum(attn_ffn_selection[key][-3])

                    active_layer_nums[split] += (sum([filter_layer_selection[i][-3] for i in range(len(filter_layer_selection))]) == layer_index).sum(0)
                    active_trans_nums[split] += (sum([attn_ffn_selection[i][0] for i in range(len(attn_ffn_selection))] + [attn_ffn_selection[i][-3]
                                                                                                                           for i in range(len(attn_ffn_selection))]) == trans_index).sum(0)
                if args.mode_name == "GRAPH":
                    for key in channel_states[split].keys():
                        all_reduce(channel_states[split][key])
                        all_reduce(layer_states[split][key])
                    for key in attn_states[split].keys():
                        all_reduce(attn_states[split][key])
                        all_reduce(ffn_states[split][key])

                    all_reduce(sent_counter)
                    all_reduce(active_layer_nums[split])
                    all_reduce(active_trans_nums[split])
                split_correct_total.append((split, sent_counter[0], sent_counter[1]))
            # if local_rank=0:
            precision = dict()
            for split, correct, total in split_correct_total:
                print(f'correct={correct}, total={total}')
                precision[split] = correct / total
            log_str = ""
            results_dict = dict()
            results_dict['channel_states'] = channel_states
            results_dict['layer_states'] = layer_states
            results_dict['attn_states'] = attn_states
            results_dict['ffn_states'] = ffn_states
            results_dict['active_layer_nums'] = active_layer_nums
            results_dict['active_trans_nums'] = active_trans_nums
            results_dict['precision'] = precision

            # draw img
            if local_rank == 0:
                for i, split in enumerate(args.eval_splits):
                    layer_select_ratios = [float(layer_state) / float(split_correct_total[i][2]) for layer_state in layer_states[split].values()]
                    plt.figure()
                    plt.plot(range(len(layer_select_ratios)), layer_select_ratios, '-o')
                    plt.xlabel("Layer Index")
                    plt.ylabel("Select Ratio")
                    img_path = os.path.join(log_root, f"{split}_layer_select_ratio.jpg")
                    plt.savefig(img_path)

                    attn_select_ratios = [float(attn_state) / float(split_correct_total[i][2]) for attn_state in attn_states[split].values()]
                    plt.figure()
                    plt.plot(range(len(attn_select_ratios)), attn_select_ratios, '-o')
                    plt.xlabel('Attn Index')
                    plt.ylabel("Select Ratio")
                    img_path = os.path.join(log_root, f"{split}_attn_select_ratio.jpg")
                    plt.savefig(img_path)

                    ffn_select_ratios = [float(ffn_state) / float(split_correct_total[i][2]) for ffn_state in ffn_states[split].values()]
                    plt.figure()
                    plt.plot(range(len(ffn_select_ratios)), ffn_select_ratios, '-o')
                    plt.xlabel("Ffn Index")
                    plt.ylabel("Select Ration")
                    img_path = os.path.join(log_root, f"{split}_ffn_select_ratio.jpg")
                    plt.savefig(img_path)

                # log_file
                for split in args.eval_splits:
                    log_str += f'Eval split: {split}, precision: {precision[split]}\n'
                    print(f'Eval split: {split}, precision: {precision[split]}\n')
                # with open()
                # for i in range(len(args.eval_splits)):
                #     split=
                print(log_str)
                logFile = os.path.join(log_root, f"{ckpt_path[-8:-5]}log.txt")
                resultsFile = os.path.join(log_root, f"{ckpt_path[-8:-5]}results.npy")
                np.save(resultsFile, results_dict)
                with open(logFile, 'a') as f:
                    f.write(log_str)


if __name__ == "__main__":
    eval(args.eval_ckpt)
    # ckpt_list=os.listdir("./train_checkpoints")
    # print(ckpt_list)
    # eval_list(ckpt_list)
    # # for ckpt_path in ckpt_list:
    # #     if "7-19" in ckpt_path:
    # #         eval(ckpt_path)