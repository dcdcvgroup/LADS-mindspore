import os
from PIL import Image
from time import time
import src.transforms as T
from src.xenv import console
import mindspore.dataset as ds
from mindspore import Tensor
import mindspore
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.dataset import vision as vision
from src.distributed_sampler import DistributedSampler
from src.referdataset import ReferDataset


def _get_rank_info(distribute):
    """get rank size and rank id"""

    if distribute:
        init()
        rank_id = get_rank()
        device_num = get_group_size()
    else:
        rank_id = 0
        device_num = 1
    return device_num, rank_id


def cast_type(dataset: mindspore.dataset.Dataset, type: mindspore.Type, columns: list):
    type_cast_operation = ds.transforms.transforms.TypeCast(type)
    for column in columns:
        dataset = dataset.map(operations=type_cast_operation, input_columns=column, num_parallel_workers=2)

    return dataset


def create_ref_dataset(args, do_train: bool, batch_size=16, distribute=False, enable_cache=False):
    device_num, rank_id = _get_rank_info(distribute)  # get device num and rankid
    # set prefetch size
    ds.config.set_prefetch_size(10)
    ds.config.set_enable_watchdog(False)
    if do_train:
        dataset = ReferDataset(args.config_root, args.dataset_root, args.dataset_type, args.splitBy, ['train'],
                               args.use_index)

        distributed_sampler = DistributedSampler(len(dataset), device_num, rank_id, shuffle=True)

        print("Start GeneratorDataset")
        # if device_num == 1:
        #     dataset = ds.GeneratorDataset(source=dataset, column_names=['image', 'bbox', 'text'],
        #                                   num_parallel_workers=2, sampler=distributed_sampler)
        # else:
        dataset = ds.GeneratorDataset(source=dataset,
                                      column_names=['image', 'bbox', 'text'],
                                      num_parallel_workers=1,
                                      shuffle=True,
                                      num_shards=device_num,
                                      shard_id=rank_id)
        print("GeneratorDataset Finished")

        # data aug for training (random resize, random crop, random horizontal flip, totensor, normalize and pad)
        sizes = [args.img_size - 32 * r for r in range(5)]
        sizes.extend([args.img_size + 32 * r for r in range(1, 5)])

        transform_list = ds.transforms.Compose(
            [T.RandomResize(*sizes),
             T.RandomCrop(args.img_size, args.img_size),
             T.RandomHorizontalFlip()])
        dataset = dataset.map(operations=transform_list,
                              input_columns=['image', 'bbox', 'text'],
                              output_columns=['image', 'bbox', 'text'],
                              num_parallel_workers=2)

        dataset = dataset.map(operations=vision.ToTensor(),
                              input_columns=['image'],
                              output_columns=['image'],
                              num_parallel_workers=2)

        dataset = dataset.map(operations=T.NormalizeAndPad(size=args.img_size,translate=args.translate),
                              input_columns=['image', 'bbox', 'text'],
                              output_columns=['image', 'bbox', 'mask', 'text'],
                              num_parallel_workers=2)

        # set batch_size
        dataset = dataset.batch(batch_size, drop_remainder=True, num_parallel_workers=1)
        dataset = dataset.map(operations=T.MyTokenizer(bert_model=args.bert_model,
                                                       batch_size=batch_size,
                                                       token_max_len=args.max_len,
                                                       text_encoder_layer_num=args.text_encoder_layer),
                              input_columns=['text'],
                              output_columns=['text'],
                              num_parallel_workers=2)
        # set data type
        m_type = mindspore.float32
        dataset = cast_type(dataset, m_type, ['image', 'bbox', 'mask'])
        dataset = cast_type(dataset, mindspore.int32, ['text'])

        return dataset

    else:
        # eval
        print('==>Generating eval dataset list')
        eval_dataset_list = []

        for split in args.eval_splits:
            single_evalset = ReferDataset(args.config_root, args.dataset_root, args.dataset_type, args.splitBy, split,
                                          args.use_index)

            distributed_sampler = DistributedSampler(len(single_evalset), device_num, rank_id, shuffle=False)
            single_evalset = ds.GeneratorDataset(source=single_evalset,
                                                 column_names=['image', 'bbox', 'text'],
                                                 num_parallel_workers=2,
                                                 sampler=distributed_sampler,
                                                 num_shards=device_num,
                                                 shard_id=rank_id)
            # data aug (randomresize, totensor, normalizeandpad)
            single_evalset = single_evalset.map(
                operations=T.RandomResize(args.img_size),
                input_columns=['image', 'bbox', 'text'],
                output_columns=['image', 'bbox', 'text'],
                num_parallel_workers=2,
            )
            single_evalset = single_evalset.map(
                operations=vision.ToTensor(),
                input_columns=['image'],
                output_columns=['image'],
                num_parallel_workers=2,
            )
            single_evalset = single_evalset.map(
                operations=T.NormalizeAndPad(size=args.img_size),
                input_columns=['image', 'bbox', 'text'],
                output_columns=['image', 'bbox', 'mask', 'text'],
                num_parallel_workers=2,
            )
            single_evalset = single_evalset.batch(
                batch_size,
                drop_remainder=True,
                num_parallel_workers=2,
            )
            single_evalset = single_evalset.map(operations=T.MyTokenizer(
                bert_model=args.bert_model,
                batch_size=batch_size,
                token_max_len=args.max_len,
                text_encoder_layer_num=args.text_encoder_layer),
                                                input_columns=['text'],
                                                output_columns=['text'])
            m_type = mindspore.float32
            single_evalset = cast_type(single_evalset, m_type, ['image', 'bbox', 'mask'])
            single_evalset = cast_type(single_evalset, mindspore.int32, ['text'])
            eval_dataset_list.append(single_evalset)
        return eval_dataset_list


def test0():
    from xenv import parse_args
    args = parse_args()
    dataset: ds.Dataset = create_ref_dataset(args=args, do_train=True, batch_size=args.batch_size)

    data_loader = dataset.create_tuple_iterator()
    for image, bbox, mask, text in data_loader:
        print(image.shape)
        print(text)
        print(text.shape)
        print(type(text))
        break


if __name__ == '__main__':
    test0()
