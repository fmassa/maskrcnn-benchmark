# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch.nn import functional as F
from torch import nn

from maskrcnn_benchmark.modeling.box_coder import BoxCoder


import math

import torch
from torch import nn

from maskrcnn_benchmark.structures.bounding_box import BoxList


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
        straddle_thresh=0,
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
        self.straddle_thresh = straddle_thresh

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

    def add_visibility_to(self, boxlist):
        image_width, image_height = boxlist.size
        anchors = boxlist.bbox
        if self.straddle_thresh >= 0:
            inds_inside = (
                (anchors[..., 0] >= -self.straddle_thresh)
                & (anchors[..., 1] >= -self.straddle_thresh)
                & (anchors[..., 2] < image_width + self.straddle_thresh)
                & (anchors[..., 3] < image_height + self.straddle_thresh)
            )
        else:
            device = anchors.device
            inds_inside = torch.ones(anchors.shape[0], dtype=torch.uint8, device=device)
        boxlist.add_field("visibility", inds_inside)

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
                self.add_visibility_to(boxlist)
                anchors_in_image.append(boxlist)
            anchors.append(anchors_in_image)
        return anchors











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



from .utils import concat_box_prediction_layers

from ..balanced_positive_negative_sampler import BalancedPositiveNegativeSampler
from ..utils import cat

from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist


class RPN(torch.nn.Module):

    def __init__(self,
            anchor_generator,
            head,
            #
            fg_iou_thresh, bg_iou_thresh,
            batch_size_per_image, positive_fraction,
            #
            box_selector,
            rpn_only):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            box_coder (BoxCoder)
        """
        super(RPN, self).__init__()
        box_similarity = boxlist_iou

        proposal_matcher = Matcher(
            fg_iou_thresh,
            bg_iou_thresh,
            allow_low_quality_matches=True,
        )

        fg_bg_sampler = BalancedPositiveNegativeSampler(
            batch_size_per_image, positive_fraction
        )

        box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
        fields_to_keep = []
        # self.generate_labels_func = generate_rpn_labels
        discard_cases = ['not_visibility', 'between_thresholds']

        self.anchor_generator = anchor_generator
        self.head = head
        self.rpn_wrapup = RPNWrapUp(fg_bg_sampler, box_coder,
                box_similarity, proposal_matcher, fields_to_keep, discard_cases,
                box_selector, rpn_only)

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
        objectness, rpn_box_regression = self.head(features)
        anchors = self.anchor_generator(images, features)
        boxes, losses = self.rpn_wrapup(objectness, rpn_box_regression, anchors, targets)
        return boxes, losses


def assign_targets_to_anchors(anchors, targets, box_similarity, proposal_matcher, fields_to_keep, discard_cases):
    for anchors_per_image, targets_per_image in zip(anchors, targets):
        match_quality_matrix = box_similarity(targets_per_image, anchors_per_image)
        matched_idxs = proposal_matcher(match_quality_matrix)
        # Fast RCNN only need "labels" field for selecting the targets
        targets_per_image = targets_per_image.copy_with_fields(fields_to_keep)
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = targets_per_image[matched_idxs.clamp(min=0)]

        # labels_per_image = self.generate_labels_func(matched_targets)
        labels_per_image = matched_idxs >= 0
        labels_per_image = labels_per_image.to(dtype=torch.float32)

        # Background (negative examples)
        bg_indices = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
        labels_per_image[bg_indices] = 0

        # discard anchors that go out of the boundaries of the image
        if "not_visibility" in discard_cases:
            labels_per_image[~anchors_per_image.get_field("visibility")] = -1

        # discard indices that are between thresholds
        if "between_thresholds" in discard_cases:
            inds_to_discard = matched_idxs == Matcher.BETWEEN_THRESHOLDS
            labels_per_image[inds_to_discard] = -1

        anchors_per_image.add_field("rpn_matched_gt_boxes", matched_targets.bbox)
        anchors_per_image.add_field("rpn_labels", labels_per_image)


def rpn_loss(objectness, box_regression, anchors, targets,
        fg_bg_sampler, box_coder, box_similarity, proposal_matcher, fields_to_keep, discard_cases):
    """
    Arguments:
        anchors (list[list[BoxList]])
        objectness (list[Tensor])
        box_regression (list[Tensor])
        targets (list[BoxList])

    Returns:
        objectness_loss (Tensor)
        box_loss (Tensor
    """
    anchors = [cat_boxlist(anchors_per_image) for anchors_per_image in anchors]
    assign_targets_to_anchors(anchors, targets, box_similarity, proposal_matcher, fields_to_keep, discard_cases)
    labels = [p.get_field("rpn_labels") for p in anchors]
    regression_targets = []
    for anchors_per_image in anchors:
        matched_gt_boxes = anchors_per_image.get_field("rpn_matched_gt_boxes")
        regression_targets_per_image = box_coder.encode(matched_gt_boxes, anchors_per_image.bbox)
        regression_targets.append(regression_targets_per_image)

    sampled_pos_inds, sampled_neg_inds = fg_bg_sampler(labels)
    sampled_pos_inds = torch.nonzero(torch.cat(sampled_pos_inds, dim=0)).squeeze(1)
    sampled_neg_inds = torch.nonzero(torch.cat(sampled_neg_inds, dim=0)).squeeze(1)

    sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

    objectness, box_regression = \
            concat_box_prediction_layers(objectness, box_regression)

    objectness = objectness.squeeze(1)

    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    box_loss = smooth_l1_loss(
        box_regression[sampled_pos_inds],
        regression_targets[sampled_pos_inds],
        beta=1.0 / 9,
        size_average=False,
    ) / (sampled_inds.numel())

    # TODO replace with Focal Loss and remove fg_bg_sampler?
    objectness_loss = F.binary_cross_entropy_with_logits(
        objectness[sampled_inds], labels[sampled_inds]
    )

    return objectness_loss, box_loss

# This function should be overwritten in RetinaNet
def generate_rpn_labels(matched_targets):
    matched_idxs = matched_targets.get_field("matched_idxs")
    labels_per_image = matched_idxs >= 0
    return labels_per_image

from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms
from maskrcnn_benchmark.structures.boxlist_ops import remove_small_boxes

from ..utils import cat
from .utils import permute_and_flatten

class RPNInference(torch.nn.Module):
    """
    Performs post-processing on the outputs of the RPN boxes, before feeding the
    proposals to the heads
    """

    def __init__(
        self,
        pre_nms_top_n,
        post_nms_top_n,
        nms_thresh,
        min_size,
        box_coder=None,
        fpn_post_nms_top_n=None,
    ):
        """
        Arguments:
            pre_nms_top_n (int)
            post_nms_top_n (int)
            nms_thresh (float)
            min_size (int)
            box_coder (BoxCoder)
            fpn_post_nms_top_n (int)
        """
        super(RPNInference, self).__init__()
        self._pre_nms_top_n = pre_nms_top_n
        self._post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.min_size = min_size

        if box_coder is None:
            box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
        self.box_coder = box_coder

        if fpn_post_nms_top_n is None:
            fpn_post_nms_top_n = post_nms_top_n
        self._fpn_post_nms_top_n = fpn_post_nms_top_n

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

    @property
    def fpn_post_nms_top_n(self):
        if self.training:
            return self._fpn_post_nms_top_n['training']
        return self._fpn_post_nms_top_n['testing']

    def regress_anchors(self, anchors, box_regression):
        # TODO make this return a BoxList?
        N, Ax4, H, W = box_regression.shape
        A = Ax4 // 4

        # put in the same format as anchors
        box_regression = permute_and_flatten(box_regression, N, A, 4, H, W)

        concat_anchors = torch.cat([a.bbox for a in anchors], dim=0)
        concat_anchors = concat_anchors.reshape(N, -1, 4)

        # TODO before we would first select the top_n and then decode
        # doesn't seem to make a difference so I just switch
        # the order to make things clearer
        proposals = self.box_coder.decode(
            box_regression.view(-1, 4), concat_anchors.view(-1, 4)
        )

        proposals = proposals.view(N, -1, 4)
        return proposals

    def select_top_n_pre_nms(self, proposals, objectness):
        selected_boxlists = []
        device = objectness.device
        batch_size, num_anchors = objectness.shape
        pre_nms_top_n = min(self.pre_nms_top_n, num_anchors)
        objectness, topk_idx = objectness.topk(pre_nms_top_n, dim=1, sorted=True)
        batch_idx = torch.arange(batch_size, device=device)[:, None]
        proposals = proposals[batch_idx, topk_idx]
        return proposals, objectness

    def filter_proposals(self, proposals, objectness, image_shapes):
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
        return result

    def process_single_feature_map(self, anchors, objectness, box_regression):
        """
        Arguments:
            anchors: list[BoxList]
            objectness: tensor of size N, A, H, W
            box_regression: tensor of size N, A * 4, H, W
        """
        proposals = self.regress_anchors(anchors, box_regression)
        objectness = objectness.sigmoid()
        N, A, H, W = objectness.shape
        objectness = permute_and_flatten(objectness, N, A, 1, H, W).view(N, -1)
        proposals, objectness = self.select_top_n_pre_nms(proposals, objectness)
        image_shapes = [box.size for box in anchors]
        result = self.filter_proposals(proposals, objectness, image_shapes)
        return result

    def forward(self, anchors, objectness, box_regression):
        with torch.no_grad():
            num_anchors = [len(a) for a in anchors[0]]
            anchors = [cat_boxlist(boxlist) for boxlist in anchors]
            N = len(anchors)

            objectness, box_regression = \
                    concat_box_prediction_layers(objectness, box_regression)

            device = objectness.device

            concat_anchors = torch.cat([a.bbox for a in anchors], dim=0)
            concat_anchors = concat_anchors.reshape(N, -1, 4)

            proposals = self.box_coder.decode(
                box_regression.view(-1, 4), concat_anchors.view(-1, 4)
            )

            objectness_ = objectness.reshape(N, -1)
            proposals_ = proposals.view(N, -1, 4)
            num_total_anchors = objectness_.shape[1]
            image_shapes = [box.size for box in anchors]

            final_result = []
            for objectness, proposals in zip(objectness_.split(num_anchors, 1), proposals_.split(num_anchors, 1)):
                num_anchors_ = objectness.shape[1]
                pre_nms_top_n = min(self.pre_nms_top_n, num_anchors_)

                objectness, topk_idx = objectness.topk(pre_nms_top_n, dim=1, sorted=True)
                batch_idx = torch.arange(N, device=device)[:, None]
                proposals = proposals[batch_idx, topk_idx]

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


    def forward0(self, anchors, objectness, box_regression):
        """
        Arguments:
            anchors: list[list[BoxList]]
            objectness: list[tensor]
            box_regression: list[tensor]

        Returns:
            boxlists (list[BoxList]): the post-processed anchors, after
                applying box decoding and NMS
        """
        with torch.no_grad():
            sampled_boxes = []
            num_levels = len(objectness)
            anchors = list(zip(*anchors))
            for a, o, b in zip(anchors, objectness, box_regression):
                sampled_boxes.append(self.process_single_feature_map(a, o, b))

            boxlists = list(zip(*sampled_boxes))
            boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]

            if num_levels > 1:
                boxlists = self.select_over_all_levels(boxlists)

        return boxlists

    def select_over_all_levels(self, boxlists):
        num_images = len(boxlists)
        for i in range(num_images):
            objectness = boxlists[i].get_field("objectness")
            post_nms_top_n = min(self.fpn_post_nms_top_n, len(objectness))
            _, inds_sorted = torch.topk(
                objectness, post_nms_top_n, dim=0, sorted=True
            )
            boxlists[i] = boxlists[i][inds_sorted]
        return boxlists


class RPNWrapUp(torch.nn.Module):
    def __init__(self, fg_bg_sampler, box_coder,
            box_similarity, proposal_matcher, fields_to_keep, discard_cases,
            box_selector, rpn_only):
        super(RPNWrapUp, self).__init__()
        self.fg_bg_sampler = fg_bg_sampler
        self.box_coder = box_coder
        self.box_similarity = box_similarity
        self.proposal_matcher = proposal_matcher
        self.discard_cases = discard_cases
        self.fields_to_keep = fields_to_keep
        self.rpn_inference = box_selector
        self.rpn_only = rpn_only

    def forward(self, objectness, box_regression, anchors, targets=None):
        if self.training:
            if self.rpn_only:
                # When training an RPN-only model, the loss is determined by the
                # predicted objectness and rpn_box_regression values and there is
                # no need to transform the anchors into predicted boxes; this is an
                # optimization that avoids the unnecessary transformation.
                boxes = anchors
            else:
                # For end-to-end models, anchors must be transformed into boxes and
                # sampled into a training batch.
                boxes = self.rpn_inference(anchors, objectness, box_regression)
            loss_objectness, loss_rpn_box_reg = rpn_loss(
                    objectness, box_regression, anchors, targets, self.fg_bg_sampler, self.box_coder,
                    self.box_similarity, self.proposal_matcher, self.fields_to_keep, self.discard_cases)
            losses = {
                "loss_objectness": loss_objectness,
                "loss_rpn_box_reg": loss_rpn_box_reg,
            }
            return boxes, losses

        boxes = self.rpn_inference(anchors, objectness, box_regression)
        return boxes, {}



def build_rpn(cfg, in_channels):
    """
    This gives the gist of it. Not super important because it doesn't change as much
    """

    anchor_sizes = cfg.MODEL.RPN.ANCHOR_SIZES
    aspect_ratios = cfg.MODEL.RPN.ASPECT_RATIOS
    anchor_stride = cfg.MODEL.RPN.ANCHOR_STRIDE
    straddle_thresh = cfg.MODEL.RPN.STRADDLE_THRESH

    if cfg.MODEL.RPN.USE_FPN:
        assert len(anchor_stride) == len(
            anchor_sizes
        ), "FPN should have len(ANCHOR_STRIDE) == len(ANCHOR_SIZES)"
    else:
        assert len(anchor_stride) == 1, "Non-FPN should have a single ANCHOR_STRIDE"
    anchor_generator = AnchorGenerator(
        anchor_sizes, aspect_ratios, anchor_stride, straddle_thresh
    )

    head = RPNHead(
        in_channels, anchor_generator.num_anchors_per_location()[0]
    )

    rpn_box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))


    fpn_post_nms_top_n = dict(
            training=cfg.MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN, testing=cfg.MODEL.RPN.FPN_POST_NMS_TOP_N_TEST)

    pre_nms_top_n = dict(training=cfg.MODEL.RPN.PRE_NMS_TOP_N_TRAIN, testing=cfg.MODEL.RPN.PRE_NMS_TOP_N_TEST)
    post_nms_top_n = dict(training=cfg.MODEL.RPN.POST_NMS_TOP_N_TRAIN, testing=cfg.MODEL.RPN.POST_NMS_TOP_N_TEST)
    nms_thresh = cfg.MODEL.RPN.NMS_THRESH
    min_size = cfg.MODEL.RPN.MIN_SIZE
    box_selector = RPNInference(
        pre_nms_top_n=pre_nms_top_n,
        post_nms_top_n=post_nms_top_n,
        nms_thresh=nms_thresh,
        min_size=min_size,
        box_coder=rpn_box_coder,
        fpn_post_nms_top_n=fpn_post_nms_top_n,
    )

    fg_iou_thresh = cfg.MODEL.RPN.FG_IOU_THRESHOLD
    bg_iou_thresh = cfg.MODEL.RPN.BG_IOU_THRESHOLD

    batch_size_per_image = cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE
    positive_fraction = cfg.MODEL.RPN.POSITIVE_FRACTION

    rpn_only = cfg.MODEL.RPN_ONLY

    return RPN(anchor_generator, head,
            fg_iou_thresh, bg_iou_thresh,
            batch_size_per_image, positive_fraction,
            box_selector, rpn_only)
