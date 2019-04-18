# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import random

import torch
import torchvision
from torchvision.transforms import functional as F


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string

class RandomResizeCrop(object):
    def __init__(self, size, scale=1.5):
        #  if not isinstance(min_size, (list, tuple)):
        #    size = (size,)
        self.size = size
        self.scale = scale

    def get_params(self, boxlist):
        boxlist = boxlist.copy_with_fields([])
        while True:
            i = random.randint(0, boxlist.size[0] - self.size)
            j = random.randint(0, boxlist.size[1] - self.size)

            tb = boxlist.crop((i, j, i + self.size, j + self.size))
            keep = tb.area() > 0
            # boxlist = boxlist.clip_to_image(remove_empty=True)
            tb = tb[keep]
            # if (area > 0).any():
            if len(tb) > 0:
                return i, j

    def __call__(self, image, target):
        image = F.resize(image, int(self.size * self.scale))
        target = target.resize(image.size)
        i, j = self.get_params(target)
        t = target
        image = F.crop(image, j, i, self.size, self.size)
        target = target.crop((i, j, i + self.size, j + self.size))
        keep = target.area() > 0
        target = target[keep]
        # print(i, j, self.size, t, target)
        # print(target)
        return image, target


class Resize(object):
    def __init__(self, min_size, max_size):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = random.choice(self.min_size)
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image, target):
        size = self.get_size(image.size)
        image = F.resize(image, size)
        target = target.resize(image.size)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.hflip(image)
            target = target.transpose(0)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        return F.to_tensor(image), target


class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image, target):
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target
