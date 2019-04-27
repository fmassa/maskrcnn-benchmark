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

BEFORE = 0
AFTER = 0

class RandomResizeCrop(object):
    def __init__(self, size, scale):
        #  if not isinstance(min_size, (list, tuple)):
        #    size = (size,)
        self.size = size
        self.scale = scale

    def get_params(self, boxlist):
        boxlist = boxlist.copy_with_fields([])
        count = 0
        while True:
            count += 1
            i = random.randint(0, boxlist.size[0] - self.size)
            j = random.randint(0, boxlist.size[1] - self.size)

            tb = boxlist.crop((i, j, i + self.size, j + self.size))
            # keep = tb.area() > 0
            keep = tb.area() > 32 * 32
            if count > 10:
                keep = tb.area() > 0
            # boxlist = boxlist.clip_to_image(remove_empty=True)
            tb = tb[keep]
            # if (area > 0).any():
            if len(tb) > 0:
                return i, j

    def __call__(self, image, target):
        s = int(self.size * self.scale)
        image = F.resize(image, s)
        target = target.resize(image.size)
        i, j = self.get_params(target)
        t = target
        before = len(target)
        image = F.crop(image, j, i, self.size, self.size)
        target = target.crop((i, j, i + self.size, j + self.size))
        keep = target.area() > 0
        target = target[keep]
        after = len(target)

        #global BEFORE, AFTER
        #BEFORE += before
        #AFTER += after
        #print(before, after, BEFORE, AFTER)
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

class SquareResize(object):
    def __init__(self, size):
        self.size = size[0] if isinstance(size, (list, tuple)) else size

    def __call__(self, image, target):
        s = (self.size, self.size)
        image = F.resize(image, s)
        target = target.resize(s)
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
