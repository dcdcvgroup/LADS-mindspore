"""ReferDataset"""
import os
from PIL import Image
from time import time
import sys

sys.path.append("/data/songran/lads-project/new-lads")
import src.transforms as T
from src.xenv import console
import mindspore.dataset as ds
from mindspore import Tensor
import mindspore
import numpy as np
from mindformers import AutoTokenizer
from mindspore.communication.management import init, get_rank, get_group_size
# import matplotlib.pyplot as plt
from mindspore.dataset import vision as vision


class ReferDataset(object):
    """
    Build the refer dataset.

    Args:
        data_config_root (str): Root of data config files such as refcoco(unc).npy.
        dataset_root(str): Root of dataset.
        dataset_type(str): Type of dataset such as "refcoco".
        splitBy(str): Such as 'unc'.
        dataset_part(str): Such as 'train' or 'eval'.
        use_index(bool): Whether to use index.
    """

    def __init__(self, data_config_root, dataset_root, dataset_type, splitBy, dataset_part, use_index=True):
        start_time = time()
        self.use_index = use_index
        dataset_parts = [dataset_part] if isinstance(dataset_part, str) else dataset_part

        if splitBy is None:
            data_config_path = os.path.join(data_config_root, f'{dataset_type}.npy')
            console.print(f'Preparing {dataset_type} using dataset_part({",".join(dataset_parts)})...', style='cyan')
        else:
            data_config_path = os.path.join(data_config_root, f'{dataset_type}({splitBy}).npy')
            console.print(f'Preparing {dataset_type}({splitBy}) using dataset_part({",".join(dataset_parts)})...',
                          style='cyan')

        if os.path.exists(data_config_path):
            dataset_config = np.load(data_config_path, allow_pickle=True).item()
        else:
            raise FileNotFoundError(f"{data_config_path} not found. Run 'tools/preprocess.py' first.")

        # gather image-phrase pairs for train/test/val/[testA&testB]...
        self.index = dataset_config['index']
        self.dataset_root = dataset_root
        self.items = sum(
            [items for dataset_part, items in dataset_config['items'].items() if dataset_part in dataset_parts], [])

        assert len(self.items) > 0,\
            f"Not supported dataset_part({', '.join(dataset_parts)}), select in [{', '.join(dataset_config['items'].keys())}]"

        # print some information
        support_split_types = ', '.join(dataset_config['items'].keys())
        split_text_nums = len(self.items)
        total_text_nums = sum(len(items) for items in dataset_config['items'].values())
        console.print(f'  |- split: {support_split_types}',
                      f'  |- texts: {split_text_nums}/{total_text_nums}',
                      f'  Done({time()-start_time:.3f}s).',
                      sep='\n',
                      style='cyan')

    def __getitem__(self, index):
        item = self.items[index]

        image_path = self.get_image_path(item['image_id'])
        image = Image.open(image_path).convert("RGB")
        # image = np.asarray(image)

        xtl, ytl, w, h = item['bbox']
        cx, cy = xtl + 0.5 * w, ytl + 0.5 * h
        bbox = np.array([cx, cy, w, h])
        text = item['phrase']
        text = np.array(text)

        return (image, bbox, text)

    def __len__(self):
        return len(self.items)

    def get_image_path(self, image_id):
        if "saiaprtc12" in self.dataset_root:
            image_root = self.dataset_root
        else:
            image_root = os.path.join(self.dataset_root, 'images', 'train2014')
        if self.use_index:
            return os.path.join(image_root, os.path.basename(self.index['image'][image_id]))
        else:
            image_name = image_id
            return os.path.join(image_root, image_name)


class CountReferDataset(object):
    """
    Build the refer dataset.

    Args:
        data_config_root (str): Root of data config files such as refcoco(unc).npy.
        dataset_root(str): Root of dataset.
        dataset_type(str): Type of dataset such as "refcoco".
        splitBy(str): Such as 'unc'.
        dataset_part(str): Such as 'train' or 'eval'.
        use_index(bool): Whether to use index.
    """

    def __init__(self, data_config_root, dataset_root, dataset_type, splitBy, dataset_part, use_index=True):
        start_time = time()
        self.use_index = use_index
        dataset_parts = [dataset_part] if isinstance(dataset_part, str) else dataset_part

        if splitBy is None:
            data_config_path = os.path.join(data_config_root, f'{dataset_type}.npy')
            console.print(f'Preparing {dataset_type} using dataset_part({",".join(dataset_parts)})...', style='cyan')
        else:
            data_config_path = os.path.join(data_config_root, f'{dataset_type}({splitBy}).npy')
            console.print(f'Preparing {dataset_type}({splitBy}) using dataset_part({",".join(dataset_parts)})...',
                          style='cyan')

        if os.path.exists(data_config_path):
            dataset_config = np.load(data_config_path, allow_pickle=True).item()
        else:
            raise FileNotFoundError(f"{data_config_path} not found. Run 'tools/preprocess.py' first.")

        # gather image-phrase pairs for train/test/val/[testA&testB]...
        self.index = dataset_config['index']
        self.dataset_root = dataset_root
        self.items = sum(
            [items for dataset_part, items in dataset_config['items'].items() if dataset_part in dataset_parts], [])

        assert len(self.items) > 0,\
            f"Not supported dataset_part({', '.join(dataset_parts)}), select in [{', '.join(dataset_config['items'].keys())}]"

        # print some information
        support_split_types = ', '.join(dataset_config['items'].keys())
        split_text_nums = len(self.items)
        total_text_nums = sum(len(items) for items in dataset_config['items'].values())
        console.print(f'  |- split: {support_split_types}',
                      f'  |- texts: {split_text_nums}/{total_text_nums}',
                      f'  Done({time()-start_time:.3f}s).',
                      sep='\n',
                      style='cyan')

    def __getitem__(self, index):
        item = self.items[index]

        # image_path = self.get_image_path(item['image_id'])
        # image = Image.open(image_path).convert("RGB")
        # # image = np.asarray(image)

        # xtl, ytl, w, h = item['bbox']
        # cx, cy = xtl + 0.5 * w, ytl + 0.5 * h
        # bbox = np.array([cx, cy, w, h])
        text = item['phrase']
        # text = np.array(text)

        return text

    def __len__(self):
        return len(self.items)

    def get_image_path(self, image_id):
        image_root = os.path.join(self.dataset_root, 'images', 'mscoco', 'images', 'train2014')
        if self.use_index:
            return os.path.join(image_root, self.index['image'][image_id])
        else:
            image_name = image_id
            return os.path.join(image_root, image_name)


if __name__ == "__main__":
    from xenv import parse_args
    args = parse_args()
    args.dataset_root = "/data/songran/datasets/saiaprtc12"
    args.dataset_type = "refclef"
    args.splitBy = "berkeley"
    dataset = ReferDataset(args.config_root, args.dataset_root, args.dataset_type, args.splitBy, ['train'],
                           args.use_index)
    L = len(dataset)
    try:
        for i in range(L):
            image, text, mask =dataset[i]
            print(i)
    except:
        print("ERROR")
