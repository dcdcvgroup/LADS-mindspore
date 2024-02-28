import mindspore
import mindspore as ms
import mindspore.dataset.vision as vision
import mindspore.ops as ops
from mindspore import Tensor
import random
from PIL import ImageFilter, Image
import numpy as np
# import cv2
import os
from mindformers import AutoTokenizer

# You'd better not use network operators in mindspore like nn or ops, because they may cause bugs.


def get_image_size(img):
    """return the size of image with the form of (Width,Height)"""
    if isinstance(img, Image.Image):
        return img.size
    else:
        shape = img.shape
        if shape[0] == 3:  # (C,H,W)
            return [shape[-1], shape[-2]]
        elif shape[2] == 3:  # (H,W,C)
            return [shape[1], shape[0]]


def remove_zeros(x: dict):
    """Remove redundant zeros in the end of tensor.

    Args:
        x (ms.Tensor): Tensor.

    Returns:
        _type_: _description_
    """
    new_dict = dict()
    for key, value in x.items():
        assist = (value != 0)
        print(assist)
        print(type(assist))
        assist = assist.sum(1)
        _, max_len = ops.max(assist, 0)
        res = value[:, :int(max_len)]
        new_dict[key] = res
    return new_dict


class ColorJitter(object):

    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4, p=0.6):
        self.prob = p
        self.jitter = vision.RandomColorAdjust(brightness, contrast, saturation)

    def __call__(self, image):
        if np.random.random() < self.prob:
            image = self.jitter(image)
        return image


class RandomHorizontalFlip(object):

    def __init__(self, p=0.5):
        self.prob = p
        self.horizontalFlip = vision.HorizontalFlip()

    def __call__(self, image, bbox, text: np.ndarray):
        img=image
        if np.random.random() < self.prob:
            img = self.horizontalFlip(image)

            # When we flip the image horizontally, the texts must also change accordingly.
            text = str(text)
            text = text.replace('right', '*&^special^&*').replace('left', 'right').replace('*&^special^&*', 'left')
            bbox[..., 0] = get_image_size(img)[0] - bbox[..., 0]
            text = np.array(text)
        # return image, bbox, text  # bug
        return img, bbox, text


class RandomResize(object):

    def __init__(self, *sizes, range=False):
        assert len(sizes) > 0, 'Sizes should not be empty.'
        self.range = range
        self.resize = vision.Resize
        if not range:
            self.sizes = sizes
        else:
            self.sizes = (min(sizes), max(sizes))

    def __call__(self, image, bbox, text):
        img = image
        if self.range:
            size = random.uniform(self.sizes[0], self.sizes[1])
        else:
            size = random.choice(self.sizes)

        dir_words = ['left', 'right', 'top', 'bottom', 'middle']
        text = str(text)
        if any([wd in text for wd in dir_words]) and size > 512:
            size = 512
        img_w, img_h = get_image_size(img)
        ratio = 1.0 * size / max(img_h, img_w)
        if (bbox[2:] * ratio > 512).any():
            size = 512
            ratio = 1.0 * size / max(img_h, img_w)

        img_h, img_w = round(img_h * ratio), round(img_w * ratio)
        # plt.imsave("./test_imgs/beforeresize.jpg",np.array(img))
        img = self.resize(size=(img_h, img_w))(img)
        # plt.imsave("./test_imgs/afterresize.jpg",np.array(img))
        bbox = bbox * ratio

        return img, bbox, text


class RandomCrop(object):

    def __init__(self, min_size: int, max_size: int, crop_bbox=False):
        self.min_size = min_size
        self.max_size = max_size
        self.crop_bbox = crop_bbox

    def __call__(self, image, bbox, text):
        dir_words = ['left', 'right', 'top', 'bottom', 'middle']
        text = str(text)
        if any([wd in text for wd in dir_words]):
            return np.array(image), bbox, text

        img = image
        # plt.imsave("./test_imgs/before_crop.jpg",np.array(img))

        img_w, img_h = get_image_size(img)

        cx, cy, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
        if self.crop_bbox:
            ltx, lty, brx, bry = map(round, (cx, cy, cx, cy))
        else:
            ltx, lty, brx, bry = map(round, (cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h))

        # Avoid Rounding Problems!
        ltx, lty, brx, bry = max(ltx, 0), max(lty, 0), min(brx, img_w), min(bry, img_h)

        tw = np.random.randint(min(img_w, self.min_size), min(img_w, self.max_size) + 1, size=(1, )).item()
        th = np.random.randint(min(img_h, self.min_size), min(img_h, self.max_size) + 1, size=(1, )).item()

        max_x, max_y = min(ltx, img_w - tw), min(lty, img_h - th)
        min_x, min_y = max(0, brx - tw), max(0, bry - th)

        if max_x < min_x or max_y < min_y:
            # print("fail to random crop")
            if img_w > 512 or img_h > 512:
                print('#' * 5, (tw, th), (img_w, img_h), bbox, '<>' * 5, (max_x, max_y), (min_x, min_y), '<>' * 5,
                      (ltx, lty, brx, bry))
            return np.array(img), bbox, text

        sx = np.random.randint(min_x, max_x + 1, size=(1, )).item()
        sy = np.random.randint(min_y, max_y + 1, size=(1, )).item()

        img = vision.Crop(coordinates=(sy, sx), size=(th, tw))(img)  # np.array
        # plt.imsave("./test_imgs/after_crop.jpg",img)

        offset = np.array([sx, sy])
        lt, br = bbox[:2] - 0.5 * bbox[2:] - offset, bbox[:2] + 0.5 * bbox[2:] - offset
        # lt = torch.max(lt, mindspore.Tensor([0, 0]))
        # lt = mindspore.Tensor([max(lt[0], 0), max(lt[1], 0)])
        lt = np.array([max(lt[0], 0), max(lt[1], 0)])
        # br = torch.min(br, mindspore.Tensor([tw, th]))
        # br = mindspore.Tensor([min(br[0], int(tw)), min(
        #     br[1], int(th))], dtype=mindspore.float16)
        br = np.array([min(br[0], tw), min(br[1], th)])
        # bbox = torch.cat([0.5*(lt+br), (br-lt)])
        # bbox type : np.array -> mindspore.Tensor
        # bbox = ops.Concat()([0.5*(lt+br), (br-lt)])
        # bbox=ops.concat((0.5*(lt+br),(br-lt)))
        bbox = np.concatenate((0.5 * (lt + br), (br - lt)))

        return img, bbox, text


class ToTensor(object):
    """prepend batch_axis(image,text)"""

    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer
        self.to_tensor = vision.ToTensor()

    def __call__(self, image):
        img = image
        # vivison.totensor: change the shape from (H, W, C) to (C, H, W)
        img = self.to_tensor(img)
        return img


class NormalizeAndPad(object):
    """
    NormalizeAndPad.

    Warning: The input of this class must have gone through the vision.ToTensor().
    """

    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), size=512, translate=False):
        self.mean = mean
        self.std = std
        self.size = size
        self.translate = translate
        self.normalize=vision.Normalize(self.mean,self.std,is_hwc=False)

    def __call__(self, image, bbox, text):
        img = image
        img = self.normalize(img)  # np.array
        # img CHW
        img_w, img_h = get_image_size(img)
        max_y = self.size - img_h
        max_x = self.size - img_w

        if self.translate:
            sy = np.random.randint(0, max_y + 1, size=(1, )).item()
            sx = np.random.randint(0, max_x + 1, size=(1, )).item()
        else:
            sy = max_y // 2
            sx = max_x // 2

        # pad_img = ops.Zeros()((1,3,self.size,self.size),mindspore.float16)
        # pad_img = ops.Zeros()((self.size, self.size, 3), mindspore.float16)
        pad_img = np.zeros(shape=(3, self.size, self.size))
        # mask = ops.Zeros()((self.size, self.size), mindspore.float16)
        mask = np.zeros(shape=(self.size, self.size))

        # pad_img[..., sy:sy+img_h, sx:sx+img_w] = img
        pad_img[..., sy:sy + img_h, sx:sx + img_w] = img
        mask[sy:sy + img_h, sx:sx + img_w] = 1.0

        bbox = bbox + np.array([sx, sy, 0, 0])
        bbox = bbox / np.array([self.size] * 4)
        # input_dict['bbox'] = bbox.unsqueeze(0)

        return pad_img, bbox, mask, text


class MyTokenizer(object):
    """
    Tokenize the texts.

    Args:
        bert_model_root(str): The path of the root of the bert model.
        bert_model(str): The type of the bert model.
        batch_size(int): The batch size.
        token_max_len(int): Max length of the final token.
        text_encoder_layer_num

    """

    def __init__(self,
                 bert_model_root="./checkpoint_download/bert",
                 bert_model='bert_base_uncased',
                 batch_size=8,
                 token_max_len=15,
                 text_encoder_layer_num=6):
        super().__init__()
        custom_config_path = os.path.join(
            bert_model_root,
            bert_model + '_' + str(batch_size) + '_' + str(token_max_len) + '_' + str(text_encoder_layer_num))
        assert os.path.exists(custom_config_path), custom_config_path + \
            'not exits, please run generate_bert_config.py first.'
        self.tokenizer = AutoTokenizer.from_pretrained(yaml_name_or_path=custom_config_path)
        self.max_len = token_max_len

    def __call__(self, texts):
        texts = texts.tolist()
        texts = self.tokenizer(texts, max_length=self.max_len, padding="max_length", truncation=True)  # texts is a dict
        texts = [v for k, v in texts.items()]  # convert texts to a list
        # {'input_ids':...,'token_type_ids':...,'attention_mask':...}
        # ==>[input_ids,token_type_ids,attention_mask]
        return np.array(texts)
