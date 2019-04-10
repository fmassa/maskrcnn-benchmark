# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch.nn import functional as F
from torch import nn

from maskrcnn_benchmark.modeling.box_coder import BoxCoder


import math


from maskrcnn_benchmark.structures.bounding_box import BoxList


from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms
from maskrcnn_benchmark.structures.boxlist_ops import remove_small_boxes


from .utils import concat_box_prediction_layers

from ..balanced_positive_negative_sampler import BalancedPositiveNegativeSampler

from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou



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
                boxlist = BoxList(
                    anchors_per_feature_map, (image_width, image_height), mode="xyxy"
                )
                anchors_in_image.append(boxlist)
            anchors.append(anchors_in_image)
        anchors = [cat_boxlist(anchors_per_image) for anchors_per_image in anchors]
        return anchors



def boxlist_is_inside_image(boxlist, straddle_thresh=0):
    image_width, image_height = boxlist.size
    anchors = boxlist.bbox
    if straddle_thresh >= 0:
        inds_inside = (
            (anchors[..., 0] >= -straddle_thresh)
            & (anchors[..., 1] >= -straddle_thresh)
            & (anchors[..., 2] < image_width + straddle_thresh)
            & (anchors[..., 3] < image_height + straddle_thresh)
        )
    else:
        device = anchors.device
        inds_inside = torch.ones(anchors.shape[0], dtype=torch.uint8, device=device)
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
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            box_coder (BoxCoder)
        """
        super(RPN, self).__init__()
        self.anchor_generator = anchor_generator
        self.head = head
        self.box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

        # used during training
        self.box_similarity = boxlist_iou

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
            match_quality_matrix = self.box_similarity(targets_per_image, anchors_per_image)
            matched_idxs = self.proposal_matcher(match_quality_matrix)
            gt_boxes = targets_per_image.bbox
            # get the targets corresponding GT for each proposal
            # NB: need to clamp the indices because we can have a single
            # GT in the image, and matched_idxs can be -2, which goes
            # out of bounds
            matched_gt_boxes_per_image = gt_boxes[matched_idxs.clamp(min=0)]

            labels_per_image = matched_idxs >= 0
            labels_per_image = labels_per_image.to(dtype=torch.float32)

            # Background (negative examples)
            bg_indices = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_indices] = 0

            # discard anchors that go out of the boundaries of the image
            inds_inside = boxlist_is_inside_image(anchors_per_image)
            labels_per_image[~inds_inside] = -1

            # discard indices that are between thresholds
            inds_to_discard = matched_idxs == Matcher.BETWEEN_THRESHOLDS
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
        with torch.no_grad():
            N = len(anchors)
            concat_anchors = torch.cat([a.bbox for a in anchors], dim=0)

            proposals = self.box_coder.decode(
                pred_bbox_deltas, concat_anchors
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

    def filter_proposals(self, proposals_, objectness, image_shapes, num_anchors):
        with torch.no_grad():
            N = proposals_.shape[0]
            objectness_ = objectness.reshape(N, -1)

            final_result = []
            for objectness, proposals in zip(objectness_.split(num_anchors, 1), proposals_.split(num_anchors, 1)):
                proposals, objectness = self.select_top_n_pre_nms(proposals, objectness)

                result = []
                for proposal, score, im_shape in zip(proposals, objectness, image_shapes):
                    boxlist = BoxList(proposal, im_shape, mode="xyxy")
                    boxlist.add_field("objectness", score)
                    boxlist = boxlist.clip_to_image(remove_empty=False)
                    boxlist = remove_small_boxes(boxlist, self.min_size)
                    boxlist = boxlist_nms(
                        boxlist,
                        self.nms_thresh,
                        max_proposals=self.post_nms_top_n,
                        score_field="objectness",
                    )
                    result.append(boxlist)
                final_result.append(result)

            boxlists = list(zip(*final_result))
            boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]

            if len(num_anchors) > 1:
                boxlists = self.select_over_all_levels(boxlists)
        return boxlists

    def select_over_all_levels(self, boxlists):
        num_images = len(boxlists)
        for i in range(num_images):
            objectness = boxlists[i].get_field("objectness")
            post_nms_top_n = min(self.post_nms_top_n, len(objectness))
            _, inds_sorted = torch.topk(
                objectness, post_nms_top_n, dim=0, sorted=True
            )
            boxlists[i] = boxlists[i][inds_sorted]
        return boxlists

    def map_targets_to_deltas(self, anchors, matched_gt_boxes):
        regression_targets = []
        for anchors_per_image, matched_gt_boxes_per_image in zip(anchors, matched_gt_boxes):
            regression_targets_per_image = self.box_coder.encode(matched_gt_boxes_per_image, anchors_per_image.bbox)
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
        image_shapes = [box.size for box in anchors]
        proposals = self.apply_deltas_to_anchors(anchors, pred_bbox_deltas)
        boxes = self.filter_proposals(proposals, objectness, image_shapes, num_anchors)

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
