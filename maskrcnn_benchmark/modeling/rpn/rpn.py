# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch.nn import functional as F
from torch import nn

from maskrcnn_benchmark.modeling.box_coder import BoxCoder


import math


from maskrcnn_benchmark.structures.bounding_box import BoxList


from torchvision.ops import nms as box_nms
from maskrcnn_benchmark.structures.boxlist_ops import remove_small_boxes
from maskrcnn_benchmark.structures.boxlist_ops import clip_boxes_to_image  # move to BoxList


from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.modeling.rpn.utils import concat_box_prediction_layers

from maskrcnn_benchmark.modeling.balanced_positive_negative_sampler import BalancedPositiveNegativeSampler

from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import box_iou  # move to BoxList


class BufferList(nn.Module):
    """
    Similar to nn.ParameterList, but for buffers
    """

    def __init__(self, buffers=None):
        super(BufferList, self).__init__()
        if buffers is not None:
            self.extend(buffers)

    def extend(self, buffers):
        offset = len(self)
        for i, buffer in enumerate(buffers):
            self.register_buffer(str(offset + i), buffer)
        return self

    def __len__(self):
        return len(self._buffers)

    def __iter__(self):
        return iter(self._buffers.values())


def generate_anchors(base_size, scales, aspect_ratios):
    scales = torch.as_tensor(scales, dtype=torch.float32)
    aspect_ratios = torch.as_tensor(aspect_ratios, dtype=torch.float32)
    h_ratios = torch.sqrt(aspect_ratios)
    w_ratios = 1 / h_ratios

    ws = (w_ratios[:, None] * scales[None, :]).view(-1)
    hs = (h_ratios[:, None] * scales[None, :]).view(-1)

    base_anchors = torch.stack([
        base_size - ws, base_size - hs, base_size + ws, base_size + hs
    ], dim=1) / 2
    return base_anchors.round()


class AnchorGenerator(nn.Module):
    """
    For a set of image sizes and feature maps, computes a set
    of anchors
    """

    def __init__(
        self,
        sizes=(128, 256, 512),
        aspect_ratios=(0.5, 1.0, 2.0),
        anchor_strides=(8, 16, 32),
    ):
        super(AnchorGenerator, self).__init__()

        if len(anchor_strides) == 1:
            anchor_stride = anchor_strides[0]
            cell_anchors = [
                generate_anchors(anchor_stride, sizes, aspect_ratios)
            ]
        else:
            if len(anchor_strides) != len(sizes):
                raise RuntimeError("FPN should have #anchor_strides == #sizes")

            cell_anchors = [
                generate_anchors(
                    anchor_stride,
                    size if isinstance(size, (tuple, list)) else (size,),
                    aspect_ratios
                )
                for anchor_stride, size in zip(anchor_strides, sizes)
            ]
        self.strides = anchor_strides
        self.cell_anchors = BufferList(cell_anchors)

    def num_anchors_per_location(self):
        return [len(cell_anchors) for cell_anchors in self.cell_anchors]

    def grid_anchors(self, grid_sizes):
        anchors = []
        for size, stride, base_anchors in zip(
            grid_sizes, self.strides, self.cell_anchors
        ):
            grid_height, grid_width = size
            device = base_anchors.device
            shifts_x = torch.arange(
                0, grid_width * stride, step=stride, dtype=torch.float32, device=device
            )
            shifts_y = torch.arange(
                0, grid_height * stride, step=stride, dtype=torch.float32, device=device
            )
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

            anchors.append(
                (shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4)
            )

        return anchors

    def forward(self, image_list, feature_maps):
        grid_sizes = [feature_map.shape[-2:] for feature_map in feature_maps]
        anchors_over_all_feature_maps = self.grid_anchors(grid_sizes)
        anchors = []
        for i, (image_height, image_width) in enumerate(image_list.image_sizes):
            anchors_in_image = []
            for anchors_per_feature_map in anchors_over_all_feature_maps:
                anchors_in_image.append(anchors_per_feature_map)
                # boxlist = BoxList(
                #     anchors_per_feature_map, (image_width, image_height), mode="xyxy"
                # )
                # anchors_in_image.append(boxlist)
            anchors.append(anchors_in_image)
        # anchors = [cat_boxlist(anchors_per_image) for anchors_per_image in anchors]
        anchors = [cat(anchors_per_image) for anchors_per_image in anchors]
        return anchors



def box_is_inside_image(boxes, image_size, straddle_thresh=0):
    image_width, image_height = image_size
    if straddle_thresh >= 0:
        inds_inside = (
            (boxes[..., 0] >= -straddle_thresh)
            & (boxes[..., 1] >= -straddle_thresh)
            & (boxes[..., 2] < image_width + straddle_thresh)
            & (boxes[..., 3] < image_height + straddle_thresh)
        )
    else:
        device = boxes.device
        inds_inside = torch.ones(boxes.shape[0], dtype=torch.uint8, device=device)
    return inds_inside



class RPNHead(nn.Module):
    """
    Adds a simple RPN Head with classification and regression heads
    """

    def __init__(self, in_channels, num_anchors):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
            num_anchors (int): number of anchors to be predicted
        """
        super(RPNHead, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(
            in_channels, num_anchors * 4, kernel_size=1, stride=1
        )

        for l in [self.conv, self.cls_logits, self.bbox_pred]:
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):
        logits = []
        bbox_reg = []
        for feature in x:
            t = F.relu(self.conv(feature))
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))
        return logits, bbox_reg


class RPN(torch.nn.Module):

    def __init__(self,
            anchor_generator,
            head,
            #
            fg_iou_thresh, bg_iou_thresh,
            batch_size_per_image, positive_fraction,
            #
            pre_nms_top_n, post_nms_top_n, nms_thresh):
        """
        Arguments:
        """
        super(RPN, self).__init__()
        self.anchor_generator = anchor_generator
        self.head = head
        self.box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

        # used during training
        self.box_similarity = box_iou

        self.proposal_matcher = Matcher(
            fg_iou_thresh,
            bg_iou_thresh,
            allow_low_quality_matches=True,
        )

        self.fg_bg_sampler = BalancedPositiveNegativeSampler(
            batch_size_per_image, positive_fraction
        )
        # used during testing
        self._pre_nms_top_n = pre_nms_top_n
        self._post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.min_size = 0

    def assign_targets_to_anchors(self, anchors, targets):
        labels = []
        matched_gt_boxes = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            gt_boxes = targets_per_image["boxes"]
            match_quality_matrix = self.box_similarity(gt_boxes, anchors_per_image)
            matched_idxs = self.proposal_matcher(match_quality_matrix)
            # get the targets corresponding GT for each proposal
            # NB: need to clamp the indices because we can have a single
            # GT in the image, and matched_idxs can be -2, which goes
            # out of bounds
            matched_gt_boxes_per_image = gt_boxes[matched_idxs.clamp(min=0)]

            labels_per_image = matched_idxs >= 0
            labels_per_image = labels_per_image.to(dtype=torch.float32)

            # Background (negative examples)
            bg_indices = matched_idxs == self.proposal_matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_indices] = 0

            # discard anchors that go out of the boundaries of the image
            # inds_inside = box_is_inside_image(anchors_per_image, image_size)
            # labels_per_image[~inds_inside] = -1

            # discard indices that are between thresholds
            inds_to_discard = matched_idxs == self.proposal_matcher.BETWEEN_THRESHOLDS
            labels_per_image[inds_to_discard] = -1

            labels.append(labels_per_image)
            matched_gt_boxes.append(matched_gt_boxes_per_image)
        return labels, matched_gt_boxes

    @property
    def pre_nms_top_n(self):
        if self.training:
            return self._pre_nms_top_n['training']
        return self._pre_nms_top_n['testing']

    @property
    def post_nms_top_n(self):
        if self.training:
            return self._post_nms_top_n['training']
        return self._post_nms_top_n['testing']

    def apply_deltas_to_anchors(self, anchors, pred_bbox_deltas):
        N = len(anchors)
        with torch.no_grad():
            boxes_per_image = [len(a) for a in anchors]
            concat_anchors = torch.cat(anchors, dim=0)

            proposals = self.box_coder.decode(
                pred_bbox_deltas.view(sum(boxes_per_image), -1), concat_anchors
            )
        proposals = proposals.view(N, -1, 4)

        return proposals

    def select_top_n_pre_nms(self, proposals, objectness):
        N = proposals.shape[0]
        num_anchors = objectness.shape[1]
        device = objectness.device
        pre_nms_top_n = min(self.pre_nms_top_n, num_anchors)

        objectness, topk_idx = objectness.topk(pre_nms_top_n, dim=1, sorted=True)
        batch_idx = torch.arange(N, device=device)[:, None]
        proposals = proposals[batch_idx, topk_idx]
        return proposals, objectness

    def limit_max_proposals(self, boxes, objectness):
        num_boxes = len(boxes)
        post_nms_top_n = min(self.post_nms_top_n, num_boxes)
        if num_boxes <= post_nms_top_n:
            return boxes, objectness
        objectness, inds_sorted = torch.topk(
            objectness, post_nms_top_n, dim=0, sorted=True
        )
        return boxes[inds_sorted], objectness


    def clip_and_nms(self, boxes, objectness, image_size):
        boxlist = clip_boxes_to_image(boxes, image_size)
        keep = remove_small_boxes(boxes, self.min_size)
        boxes = boxes[keep]
        objectness = objectness[keep]

        keep = box_nms(boxes, objectness, self.post_nms_top_n)
        boxes = boxes[keep]
        objectness = objectness[keep]
        return boxes, objectness

    def filter_proposals(self, proposals, objectness, image_shapes, num_anchors):
        with torch.no_grad():
            return self._filter_proposals(proposals, objectness, image_shapes, num_anchors)

    def _filter_proposals(self, proposals, objectness, image_shapes, num_anchors):
        N = proposals.shape[0]
        objectness = objectness.reshape(N, -1)

        all_proposals = proposals.split(num_anchors, 1)
        all_scores = objectness.split(num_anchors, 1)

        final_result = []
        final_boxes = []
        final_scores = []
        for proposals, scores in zip(all_proposals, all_scores):
            proposals, scores = self.select_top_n_pre_nms(proposals, scores)
            result = []
            l_boxes = []
            l_scores = []
            for p, s, im_shape in zip(proposals, scores, image_shapes):
                boxlist = BoxList(p.reshape(-1, 4), im_shape, mode="xyxy")
                p, s = self.clip_and_nms(p.reshape(-1, 4), s, im_shape)
                l_boxes.append(p)
                l_scores.append(s)
            final_boxes.append(l_boxes)
            final_scores.append(l_scores)

        final_boxes = list(zip(*final_boxes))
        final_scores = list(zip(*final_scores))
        final_boxes = [cat(x) for x in final_boxes]
        final_scores = [cat(x) for x in final_scores]
        for i in range(len(final_boxes)):
            boxes, scores = self.limit_max_proposals(final_boxes[i], final_scores[i])
            final_boxes[i] = boxes
            final_scores[i] = scores

        return final_boxes, final_scores

    def map_targets_to_deltas(self, anchors, matched_gt_boxes):
        regression_targets = []
        for anchors_per_image, matched_gt_boxes_per_image in zip(anchors, matched_gt_boxes):
            regression_targets_per_image = self.box_coder.encode(matched_gt_boxes_per_image, anchors_per_image)
            regression_targets.append(regression_targets_per_image)
        return regression_targets

    def forward(self, images, features, targets=None):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        objectness, pred_bbox_deltas = self.head(features)
        anchors = self.anchor_generator(images, features)

        num_anchors = [o[0].numel() for o in objectness]
        objectness, pred_bbox_deltas = \
                concat_box_prediction_layers(objectness, pred_bbox_deltas)
        proposals = self.apply_deltas_to_anchors(anchors, pred_bbox_deltas)
        boxes, scores = self.filter_proposals(proposals, objectness, images.image_sizes, num_anchors)

        losses = {}
        if self.training:
            labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, targets)
            regression_targets = self.map_targets_to_deltas(anchors, matched_gt_boxes)
            loss_objectness, loss_rpn_box_reg = rpn_loss(
                    objectness, pred_bbox_deltas, labels, regression_targets, self.fg_bg_sampler)
            losses = {
                "loss_objectness": loss_objectness,
                "loss_rpn_box_reg": loss_rpn_box_reg,
            }
        return boxes, losses


def rpn_loss(objectness, pred_bbox_deltas, labels, regression_targets, fg_bg_sampler):
    """
    Arguments:
        anchors (list[list[BoxList]])
        objectness (list[Tensor])
        pred_bbox_deltas (list[Tensor])
        targets (list[BoxList])

    Returns:
        objectness_loss (Tensor)
        box_loss (Tensor
    """

    sampled_pos_inds, sampled_neg_inds = fg_bg_sampler(labels)
    sampled_pos_inds = torch.nonzero(torch.cat(sampled_pos_inds, dim=0)).squeeze(1)
    sampled_neg_inds = torch.nonzero(torch.cat(sampled_neg_inds, dim=0)).squeeze(1)

    sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

    objectness = objectness.flatten()

    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    box_loss = smooth_l1_loss(
        pred_bbox_deltas[sampled_pos_inds],
        regression_targets[sampled_pos_inds],
        beta=1.0 / 9,
        size_average=False,
    ) / (sampled_inds.numel())

    objectness_loss = F.binary_cross_entropy_with_logits(
        objectness[sampled_inds], labels[sampled_inds]
    )

    return objectness_loss, box_loss



def build_rpn(cfg, in_channels):
    """
    This gives the gist of it. Not super important because it doesn't change as much
    """

    anchor_sizes = cfg.MODEL.RPN.ANCHOR_SIZES
    aspect_ratios = cfg.MODEL.RPN.ASPECT_RATIOS
    anchor_stride = cfg.MODEL.RPN.ANCHOR_STRIDE
    # straddle_thresh = cfg.MODEL.RPN.STRADDLE_THRESH

    if cfg.MODEL.RPN.USE_FPN:
        assert len(anchor_stride) == len(
            anchor_sizes
        ), "FPN should have len(ANCHOR_STRIDE) == len(ANCHOR_SIZES)"
    else:
        assert len(anchor_stride) == 1, "Non-FPN should have a single ANCHOR_STRIDE"
    anchor_generator = AnchorGenerator(
        anchor_sizes, aspect_ratios, anchor_stride
    )

    head = RPNHead(
        in_channels, anchor_generator.num_anchors_per_location()[0]
    )

    pre_nms_top_n = dict(training=cfg.MODEL.RPN.PRE_NMS_TOP_N_TRAIN, testing=cfg.MODEL.RPN.PRE_NMS_TOP_N_TEST)
    post_nms_top_n = dict(training=cfg.MODEL.RPN.POST_NMS_TOP_N_TRAIN, testing=cfg.MODEL.RPN.POST_NMS_TOP_N_TEST)
    nms_thresh = cfg.MODEL.RPN.NMS_THRESH
    # min_size = cfg.MODEL.RPN.MIN_SIZE

    fg_iou_thresh = cfg.MODEL.RPN.FG_IOU_THRESHOLD
    bg_iou_thresh = cfg.MODEL.RPN.BG_IOU_THRESHOLD

    batch_size_per_image = cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE
    positive_fraction = cfg.MODEL.RPN.POSITIVE_FRACTION

    return RPN(anchor_generator, head,
            fg_iou_thresh, bg_iou_thresh,
            batch_size_per_image, positive_fraction,
            pre_nms_top_n, post_nms_top_n, nms_thresh)
