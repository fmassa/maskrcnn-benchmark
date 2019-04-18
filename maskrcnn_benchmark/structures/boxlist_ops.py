# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from .bounding_box import BoxList

from maskrcnn_benchmark.layers import nms as _box_nms


def remove_small_boxes(boxes, min_size):
    ws, hs = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    keep = (
        (ws >= min_size) & (hs >= min_size)
    ).nonzero().squeeze(1)
    return keep

def clip_boxes_to_image(boxes, size):
    boxes = boxes.clone()
    boxes[..., 0].clamp_(min=0, max=size[1])
    boxes[..., 1].clamp_(min=0, max=size[0])
    boxes[..., 2].clamp_(min=0, max=size[1])
    boxes[..., 3].clamp_(min=0, max=size[0])
    return boxes

def box_area(box):
    return (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])

def box_iou(box1, box2):
    N = len(box1)
    M = len(box2)

    area1 = box_area(box1)
    area2 = box_area(box2)

    lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N,M,2]
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou

# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
