"""
train ReferNet 
use new optimizer
"""
import sys
sys.path.append("/data/songran/lads-project/new-lads/")
from mindspore import Tensor
from collections import OrderedDict
import datetime
import mindspore as ms
import mindspore.ops as ops
import mindspore.nn as nn
import src.xenv as xenv
from src.xenv import console
import os
import os.path as osp
import time
from mindspore.communication import init, get_rank, get_group_size
from src.dataset import create_ref_dataset
from src.model_lads import ReferNet
from mindspore.train.callback import LossMonitor, ModelCheckpoint, CheckpointConfig, TimeMonitor
import datetime
import time
from src.loss_new import NetWithLossV2, NetWithLossFptV2, MyTrainOneStepCellV2, box_ious, MyTrainOneStepCell, MyTrainOneStepCellAccumulator, MyTrainOneStepNoScaleAccumulator
import logging
import warnings
from src.optimizer import MyAdamWeightDecayV2
# from eval_net import eval_while_training

warnings.filterwarnings("ignore")

os.environ['GLOG_v'] = '3'
os.environ['MS_COMPILER_CACHE_PATH'] = "/data/songran/lads-project/compiler_cache"
os.environ['GLOG_log_dir'] = "./ms_log"
args = xenv.parse_args()

# set seed


class MyLossMonitor(LossMonitor):

    def __init__(self, per_print_times=1):
        super(MyLossMonitor, self).__init__()
        self._per_print_times = per_print_times
        self._start_time = time.time()
        self._loss_list = []

    def step_end(self, run_context):
        cb_params = run_context.original_args()


def set_parameter(args):
    """set parameter"""
    ms.context.set_context(max_call_depth=1000000000)
    target = args.device_target  # [GPU,Ascend]
    if target == "CPU":
        args.run_distribute = False

    # init context
    if args.mode_name == 'GRAPH':
        if target == "Ascend":
            rank_save_graphs_path = osp.join(args.save_graphs_path, "soma", str(os.getenv('DEVICE_ID')))
            ms.set_context(mode=ms.GRAPH_MODE, device_target=target, save_graphs=args.save_graphs, save_graphs_path=rank_save_graphs_path)
        else:
            ms.set_context(mode=ms.GRAPH_MODE, device_target=target, save_graphs=False, save_graphs_path="./graph_save/")
        # if target == "GPU":
        #     ms.set_context(mode=ms.GRAPH_MODE)
    else:
        ms.set_context(mode=ms.PYNATIVE_MODE, device_target=target, save_graphs=False, enable_graph_kernel=True)

    # run in server
    if args.parameter_server:
        ms.set_ps_context(enable_ps=True)

    # whether distribute
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


def filter_checkpoint_parameter_by_list(origin_dict, param_filter):
    """Remove useless parameters according to filter_list"""
    for key in list(origin_dict.keys()):
        for name in param_filter:
            if name in key:
                print("Delete parameter from checkpoint: ", key)
                del origin_dict[key]
                break


def load_pretrained_checkpoint():
    # get information about checkpoint and config checkpoint_path
    if args.resume or xenv.get_whole_life_state() > 0:
        checkpoint_path = osp.join(args.root_dir, 'checkpoint_latest.pth')
    elif args.start_epoch > 0:
        checkpoint_path = osp.join(args.root_dir, f'checkpoint_{args.start_epoch-1:03}.pth')
    else:
        checkpoint_path = None

    checkpoint_dict = None
    if checkpoint_path is not None:
        checkpoint_dict = ms.load_checkpoint(checkpoint_path)
        args.start_epoch = checkpoint_dict['epoch'] + 1
        color_tag = 'red' if xenv.get_whole_life_state() > 0 else 'yellow'
        console.print(f"resuming from epoch\[[green]{checkpoint_dict['epoch']}[/]]...", style=color_tag)

    return checkpoint_dict


def init_weight(net, param_dict=None, pre_trained: bool = False):
    """
    Init weight of the net.

    Args:
        net: the net to be init.
        param_dict: parameter dict to load into the net.
        pretrained(bool): whether to init weight from pretrained weights.
    """

    if pre_trained:
        if param_dict:
            if param_dict.get("epoch_num") and param_dict.get("step_num"):
                args.has_trained_epoch = int(param_dict['epoch_num'].data.asnumpy())
                args.has_trained_step = int(param_dict["step_num"].data.asnumpy())
            else:
                args.has_trained_epoch = 0
                args.has_trained_step = 0

            if args.filter_weight:
                filter_list = [x.name for x in net.end_point.get_parameters()]
                filter_checkpoint_parameter_by_list(param_dict, filter_list)
            ms.load_param_into_net(net, param_dict)
    else:
        # TODO
        # init weight through some metheds
        pass


def generate_steps_lr(lr_init, lr_max, step_size, max_epochs, warmup_steps, decay_epochs, gamma=0.1):
    """
    Applies three steps decay to generate learning rate array.

    Args:
        lr_init(float): initial learning rate.
        lr_max(float): max learning rate.
        step_size(int): steps that one epoch needs.
        max_epochs(int): max epochs.
        warmup_steps(int): all steps in warmup epochs.
        decay_epochs: a list or array of epochs when the learning rate decays.
        gamma(float): multiplicative factor of learning rate decay. Default:0.1.

    Returns:
        learning rate array.
    """

    if isinstance(decay_epochs, int):
        decay_epochs = [decay_epochs]
    total_steps = max_epochs * step_size
    decay_steps = [step_size * decay_epoch for decay_epoch in decay_epochs]
    decay_steps.append(total_steps)
    num_decay_epochs = len(decay_steps)
    learning_rates = [lr_max * pow(gamma, i) for i in range(num_decay_epochs)]
    lr_each_step = nn.piecewise_constant_lr(milestone=decay_steps, learning_rates=learning_rates)
    for i in range(warmup_steps):
        lr_each_step[i] = lr_init + (lr_max - lr_init) * i / warmup_steps

    return lr_each_step


def init_lr(step_size, whos_lr, max_epochs, warmup_steps, drop_epochs, drop_rate=0.1):
    """
    Initilize learning rate through AdamW.

    Args:
        step_size: steps that one epoch needs.
        whos_lr: who's lr.

    Returns:
        lr list or lr function.
    """

    base_lr = whos_lr
    if args.lr_scheduler == 'step':
        lr_list = generate_steps_lr(0.01 * base_lr, base_lr, step_size, max_epochs, warmup_steps, drop_epochs, drop_rate)
        return lr_list

    pass


def my_get_rank_and_size(run_distribute):
    if run_distribute:
        return get_rank(), get_group_size()
    else:
        return 0, 1

def save_ckpt(net, epoch_idx, log_dir, local_rank):
    if local_rank == 0:
        ckpt_path = os.path.join(log_dir, f"epoch-{epoch_idx}.ckpt")
        ms.save_checkpoint(net, ckpt_path)
    else:
        time.sleep(1)
    return 0

def write_log(log_str, logFile, local_rank):
    if local_rank == 0:
        with open(logFile, 'a') as f:
            f.write(log_str)
    else:
        time.sleep(1)
    return 0

def train_loop(net, train_dataloader, epoch_idx, logFile, local_rank, step_size):
    net.set_train(True)
    time_str = str(datetime.datetime.now().strftime('%m-%d_%H:%M:%S'))

    t0 = time.time()
    start_time = time.time()
    loss_sum = 0
    loss_bbox_sum = 0
    loss_xiou_sum = 0
    loss_arch_sum = 0
    overflow_count = 0
    for step_idx, data in enumerate(train_dataloader):
        loss, loss_bbox, loss_xiou, loss_arch, lr = net(data[0], data[1], data[2], data[3])
        if local_rank == 0:
            loss_sum += loss
            loss_bbox_sum += loss_bbox
            loss_xiou_sum += loss_xiou
            loss_arch_sum += loss_arch
            # if overflow:
            #     overflow_count += 1
            if (step_idx + 1) % 100 == 0:
                current_step = epoch_idx * step_size + step_idx + 1
                t1 = time.time()
                loss_mean = round(float(loss_sum / (step_idx + 1)),4)
                loss_bbox_mean = round(float(loss_bbox_sum / (step_idx + 1)),4)
                loss_xiou_mean = round(float(loss_xiou_sum / (step_idx + 1)),4)
                loss_arch_mean = round(float(loss_arch_sum / (step_idx + 1)),4)
                lr = round(float(lr),8)
                log_str = f'epoch: {str(epoch_idx).zfill(3)}/{args.max_epochs} | step: {step_idx}/{step_size} | loss: {loss_mean} | loss_bbox: {loss_bbox_mean} | loss_xiou: {loss_xiou_mean} | loss_arch: {loss_arch_mean} | lr: {lr} | time: {round(t1-t0)}s\n'
                write_log(log_str, logFile, local_rank)
                t0 = t1

            if step_idx == step_size - 1:
                loss_mean = round(float(loss_sum / (step_idx + 1)), 4)
                log_str = f'Whole Epoch | epoch: {str(epoch_idx).zfill(3)} | loss: {loss_mean}\n'
                write_log(log_str, logFile, local_rank)
        else:
            time.sleep(0.003)
        
def eval(net, eval_loader_list, args):
    all_reduce = ops.AllReduce()
    net.set_train(False)
    split_correct_total = []

    # ==============
    # channel_states = {split: OrderedDict() for split in args.eval_splits}
    # layer_states = {split: OrderedDict() for split in args.eval_splits}
    # attn_states = {split: OrderedDict() for split in args.eval_splits}
    # ffn_states = {split: OrderedDict() for split in args.eval_splits}
    # total_layer_num = len(net.selector.proj_layer)
    # active_filter_nums = {split: ops.zeros((total_layer_num, 64, 64), ms.float32) for split in args.eval_splits}
    # active_layer_nums = {split: ops.zeros(total_layer_num, ms.float32) for split in args.eval_splits}
    # total_trans_num = len(net.selector.proj_attn) * 2
    # active_trans_nums = {split: ops.zeros(total_trans_num, ms.float32) for split in args.eval_splits}
    # layer_index = ms.numpy.arange(1, total_layer_num + 1)
    # trans_index = ms.numpy.arange(1, total_trans_num + 1)

    # for split in args.eval_splits:
    #     for i in range(len(net.selector.proj_filter)):
    #         m = net.selector.proj_filter[i]
    #         channel_states[split][i] = ops.zeros_like(m.bias)
    #         layer_states[split][i] = ops.zeros(1, ms.float32)
    #     for i in range(len(net.selector.proj_attn)):
    #         attn_states[split][i] = ops.zeros(1, ms.float32)
    #         ffn_states[split][i] = ops.zeros(1, ms.float32)
    # ==============
    for i in range(len(eval_loader_list)):
        split = args.eval_splits[i]
        eval_loader = eval_loader_list[i]
        sent_counter = Tensor([0, 0], ms.float32)
        for step_idx, data in enumerate(eval_loader):
            image = data[0]
            gt_bbox = data[1]
            mask = data[2]
            text = data[3]
            pred_bbox,_,_ = net(image, gt_bbox, mask, text)
            iou = box_ious(pred_bbox[:, :4], gt_bbox, 'iou')['iou']
            sent_counter[0] += (iou >= args.iou_threshold).sum()
            sent_counter[1] += image.shape[0]
        all_reduce(sent_counter)
        split_correct_total.append((split, sent_counter[0], sent_counter[1]))
    precision = dict()
    for split, correct, total in split_correct_total:
        precision[split] = float(correct / total)

    return precision


def main():
    if args.mode_name == "PYNATIVE":
        args.run_distribute = False
    args.use_selector = True
    print(type(args.use_selector))
    print(args.use_selector)

    print(args.run_distribute)
    print(args.mode_name)
    # FIXME
    set_parameter(args)

    local_rank, device_num = my_get_rank_and_size(args.run_distribute)

    # create train dataset
    print("Start create dataset.")
    args.dataset_part = 'train'
    train_dataset = create_ref_dataset(args, do_train=True, batch_size=args.batch_size, distribute=args.run_distribute)
    print("dataset_len:", train_dataset.get_dataset_size())
    print("args.batch_size=", args.batch_size)
    print("batch size in dataset:", train_dataset.get_batch_size())
    step_size = train_dataset.get_dataset_size()
    print("Steps per epoch: ", step_size)

    eval_dataset_list = create_ref_dataset(args, False, args.batch_size, args.run_distribute)
    eval_loader_list = []
    args.eval_step_size=[]
    for eval_dataset in eval_dataset_list:
        args.eval_step_size.append(eval_dataset.get_dataset_size())
        eval_loader_list.append(eval_dataset.create_tuple_iterator())

    # ===========test==================

    # define the net to be trained
    print("Start create refernet")
    origin_net = ReferNet(args.batch_size, args.pre_norm, args.dropout, args.use_selector, args.max_len, args.use_dilation, args.token_select, args.cnn_model, args.bert_model, args.text_encoder_layer,
                          args.task_encoder_layer, args.task_encoder_dim, args.task_ffn_dim, args.task_encoder_head, args.trainable_layers, args.norm_layer,args.selector_bias)
    if args.pretrained_path != "":
        param_dict = ms.load_checkpoint(args.pretrained_path)
        missing_list = ms.load_param_into_net(origin_net, param_dict)
        print("====== Loading parameters into refernet ======")
        print("====== Missing weights are:")
        print(missing_list)
    
    # evaluate
    if args.is_train == False:
        precision = eval(origin_net, eval_loader_list, args)
        log_str = ""
        for split in args.eval_splits:
            log_str += f"{split}: {round(precision[split],4)} | "
            log_str +='\n'
        write_log(log_str, os.path.join(args.log_dir,f"rank_{local_rank}"))
        return

    # net with loss calculation
    args.accumulate_step = args.batch_sum / args.batch_size
    args.accumulate_step /= device_num
    if not args.finetune:
        net = NetWithLossV2(origin_net, args.xiou_loss_type, args.bbox_loss_coef, args.xiou_loss_coef, args.arch_loss_coef, accumulate_step=args.accumulate_step)
    else:
        net = NetWithLossFptV2(origin_net, args.xiou_loss_type, args.bbox_loss_coef, args.xiou_loss_coef, args.arch_loss_coef, accumulate_step=args.accumulate_step)

    # server
    if args.parameter_server:
        net.set_param_ps()

    # build learning rate groups
    lr_list = init_lr(step_size, args.lr_base, args.max_epochs, args.warmup_steps, args.drop_epochs, args.drop_rate)
    lr_base = init_lr(step_size, args.lr_base, args.max_epochs, args.warmup_steps, args.drop_epochs, args.drop_rate)
    lr_visual = init_lr(step_size, args.lr_visual, args.max_epochs, args.warmup_steps, args.drop_epochs, args.drop_rate)
    lr_lang = init_lr(step_size, args.lr_lang, args.max_epochs, args.warmup_steps, args.drop_epochs, args.drop_rate)
    args.lr_len = len(lr_list)
    args.step_size = step_size

    # args.end_drop_epoch = args.max_epochs
    # lr_base = nn.CosineDecayLR(0.1 * args.lr_base, args.lr_base, step_size * args.end_drop_epoch)
    # lr_visual = nn.CosineDecayLR(0.1 * args.lr_visual, args.lr_visual, step_size * args.end_drop_epoch)
    # lr_lang = nn.CosineDecayLR(0.1 * args.lr_lang, args.lr_lang, step_size * args.end_drop_epoch)

    # =================test======================
    # visual_encoder, text_encoder, selector的权重都已经初始化了
    # params_to_write = ""
    # for name, param in net.parameters_and_names():
    #     if name.startswith("network.task_encoder"):
    #         # if not name.startswith(("network.task_encoder","network.visual_encoder","network.selector")):
    #         params_to_write += f"{name}\n{param.shape}\n{param.requires_grad}\n------------\n"
    #         print(name)
    #         print(param.shape)
    #         print(param.requires_grad)
    #         print("-------------")
    #         # print(param.value())
    # txt_to_write = "parameters/task_encoder_mindspore.txt"
    # with open(txt_to_write, 'w') as f:
    #     f.write(params_to_write)

    # define parameters group, different groups have different lr.
    params_groups = [{
        'params': [p for n, p in net.parameters_and_names() if n.startswith('network.visual_encoder') and p.requires_grad],
        'lr': lr_visual
    }, {
        'params': [p for n, p in net.parameters_and_names() if n.startswith('network.text_encoder') and p.requires_grad],
        'lr': lr_lang
    }, {
        'params': [p for n, p in net.parameters_and_names() if not n.startswith(('network.visual_encoder', 'network.text_encoder', 'network.selector')) and p.requires_grad],
        'lr': lr_base
    }]

    if args.use_selector:
        params_groups.append({'params': [p for n, p in net.parameters_and_names() if n.startswith('network.selector') and p.requires_grad], 'lr': lr_base, 'weight_decay': 0.0})

    # define optimizer
    optimizer = MyAdamWeightDecayV2(params=params_groups, learning_rate=lr_base, weight_decay=args.weight_decay, accumulate_step=args.accumulate_step)

    # train
    print("=============Start Training============")
    # TODO
    # net = MyTrainOneStepCellV2(net, optimizer, 1, True, args.clip_max_norm)
    # net = MyTrainOneStepCell(net, optimizer, clip_max_norm=args.clip_max_norm)
    net = MyTrainOneStepNoScaleAccumulator(net, optimizer, clip_max_norm=args.clip_max_norm, accumulate_step=args.accumulate_step)
    ms.set_seed(args.seed + local_rank)
    # time_str = str(datetime.datetime.now().strftime('%m-%d_%H:%M:%S'))
    # log_dir = f"./my_logs/{time_str}_{args.comments}"
    # logFile = os.path.join(log_dir, f"rank_{local_rank}.txt")
    log_dir = args.log_dir
    logFile = os.path.join(log_dir, f"rank_{local_rank}.txt")
    
    # write logs(args information)
    if local_rank == 0:
        time_info = str(datetime.datetime.now()) + '\n'
        args_info_list = args._get_kwargs()
        args_info_str = ""
        for arg_tuple in args_info_list:
            args_info_str += f'{arg_tuple[0]}:{arg_tuple[1]}\n'
        with open(logFile, 'a') as f:
            f.write("Start time:")
            f.write(time_info)
            f.write(args_info_str)

    data_loader = train_dataset.create_tuple_iterator(num_epochs=args.max_epochs)
    # with ms.train.SummaryRecord(f'./summary_dir/train_01_{local_rank}', network=net) as summary_record:
    precision_best = dict()
    best_epoch = 0
    for split in args.eval_splits:
        precision_best[split] = 0
    # train
    net.set_train(True)
    for epoch_idx in range(args.max_epochs):
        train_loop(net, data_loader, epoch_idx, logFile, local_rank, step_size)
        # eval
        if (epoch_idx + 1)%args.eval_step == 0:
            write_log("Evaluating! | ", logFile, local_rank)
            precision = eval(net.network.network, eval_loader_list, args)
            log_str = f"Eval epoch: {epoch_idx} | "
            for split in args.eval_splits:
                log_str += f"{split}: {round(precision[split],4)} | "
            log_str+='\n'
            write_log(log_str, logFile, local_rank)
            # Best epoch
            if precision[args.eval_splits[-1]] > precision_best[args.eval_splits[-1]]:
                precision_best = precision
                best_epoch = epoch_idx
                save_ckpt(net.network.network, "best", log_dir, local_rank)
            write_log(f"Best epoch: {best_epoch}\n", logFile, local_rank) 
        # save the last epoch
        save_ckpt(net.network.network, "last", log_dir, local_rank)
        time.sleep(10)
    # log at the end    
    log_str = f"Best epoch: {best_epoch} | "
    for split in args.eval_splits:
        log_str += f"Eval split: {split}, precision: {precision_best[split]} | "
    log_str+='\n'
    write_log(log_str, logFile, local_rank)

if __name__ == "__main__":
    main()