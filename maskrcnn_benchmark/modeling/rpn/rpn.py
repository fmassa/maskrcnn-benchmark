# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch.nn import functional as F
from torch import nn

from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.rpn.retinanet.retinanet import build_retinanet
from .loss import make_rpn_loss_evaluator
from .anchor_generator import make_anchor_generator
from .inference import make_rpn_postprocessor



@registry.RPN_HEADS.register("SingleConvRPNHead")
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
from maskrcnn_benchmark.modeling.assign_target_to_proposal import Target2ProposalAssigner
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
            box_selector_train, box_selector_test,
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
                box_selector_train, box_selector_test, rpn_only)

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
        anchors (list[BoxList])
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

    objectness = objectness.squeeze()

    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    box_loss = smooth_l1_loss(
        box_regression[sampled_pos_inds],
        regression_targets[sampled_pos_inds],
        beta=1.0 / 9,
        size_average=False,
    ) / (sampled_inds.numel())

    objectness_loss = F.binary_cross_entropy_with_logits(
        objectness[sampled_inds], labels[sampled_inds]
    )

    return objectness_loss, box_loss

# This function should be overwritten in RetinaNet
def generate_rpn_labels(matched_targets):
    matched_idxs = matched_targets.get_field("matched_idxs")
    labels_per_image = matched_idxs >= 0
    return labels_per_image


class RPNWrapUp(torch.nn.Module):
    def __init__(self, fg_bg_sampler, box_coder,
            box_similarity, proposal_matcher, fields_to_keep, discard_cases,
            box_selector_train, box_selector_test, rpn_only):
        super(RPNWrapUp, self).__init__()
        self.fg_bg_sampler = fg_bg_sampler
        self.box_coder = box_coder
        self.box_similarity = box_similarity
        self.proposal_matcher = proposal_matcher
        self.discard_cases = discard_cases
        self.fields_to_keep = fields_to_keep
        self.box_selector_train = box_selector_train
        self.box_selector_test = box_selector_test
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
                with torch.no_grad():
                    boxes = self.box_selector_train(
                        anchors, objectness, box_regression, targets
                    )
            loss_objectness, loss_rpn_box_reg = rpn_loss(
                    objectness, box_regression, anchors, targets, self.fg_bg_sampler, self.box_coder,
                    self.box_similarity, self.proposal_matcher, self.fields_to_keep, self.discard_cases)
            losses = {
                "loss_objectness": loss_objectness,
                "loss_rpn_box_reg": loss_rpn_box_reg,
            }
            return boxes, losses

        boxes = self.box_selector_test(anchors, objectness, box_regression)
        return boxes, {}




class RPNWrapUp2(torch.nn.Module):
    def __init__(self, box_selector_train, box_selector_test, loss_evaluator, rpn_only):
        super(RPNWrapUp2, self).__init__()
        self.box_selector_train = box_selector_train
        self.box_selector_test = box_selector_test
        self.loss_evaluator = loss_evaluator
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
                with torch.no_grad():
                    boxes = self.box_selector_train(
                        anchors, objectness, box_regression, targets
                    )
            loss_objectness, loss_rpn_box_reg = self.loss_evaluator(
                anchors, objectness, box_regression, targets
            )
            losses = {
                "loss_objectness": loss_objectness,
                "loss_rpn_box_reg": loss_rpn_box_reg,
            }
            return boxes, losses

        boxes = self.box_selector_test(anchors, objectness, box_regression)
        return boxes, {}



class RPNModule(torch.nn.Module):
    """
    Module for RPN computation. Takes feature maps from the backbone and RPN
    proposals and losses. Works for both FPN and non-FPN.
    """

    def __init__(self, anchor_generator, head, box_selector_train, box_selector_test, loss_evaluator, rpn_only):
        super(RPNModule, self).__init__()

        self.anchor_generator = anchor_generator
        self.head = head
        self.rpn_wrapup = RPNWrapUp2(box_selector_train, box_selector_test, loss_evaluator, rpn_only)
        self.rpn_only = rpn_only

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


def build_rpn0(cfg, in_channels):
    """
    This gives the gist of it. Not super important because it doesn't change as much
    """
    if cfg.MODEL.RETINANET_ON:
        return build_retinanet(cfg, in_channels)


    anchor_generator = make_anchor_generator(cfg)

    rpn_head = registry.RPN_HEADS[cfg.MODEL.RPN.RPN_HEAD]
    head = rpn_head(
        in_channels, anchor_generator.num_anchors_per_location()[0]
    )

    rpn_box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

    box_selector_train = make_rpn_postprocessor(cfg, rpn_box_coder, is_train=True)
    box_selector_test = make_rpn_postprocessor(cfg, rpn_box_coder, is_train=False)

    loss_evaluator = make_rpn_loss_evaluator(cfg, rpn_box_coder)

    return RPNModule(anchor_generator, head, box_selector_train, box_selector_test, loss_evaluator, cfg.MODEL.RPN_ONLY)

def build_rpn(cfg, in_channels):
    """
    This gives the gist of it. Not super important because it doesn't change as much
    """
    if cfg.MODEL.RETINANET_ON:
        return build_retinanet(cfg, in_channels)


    anchor_generator = make_anchor_generator(cfg)

    rpn_head = registry.RPN_HEADS[cfg.MODEL.RPN.RPN_HEAD]
    head = rpn_head(
        in_channels, anchor_generator.num_anchors_per_location()[0]
    )

    rpn_box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

    box_selector_train = make_rpn_postprocessor(cfg, rpn_box_coder, is_train=True)
    box_selector_test = make_rpn_postprocessor(cfg, rpn_box_coder, is_train=False)

    fg_iou_thresh = cfg.MODEL.RPN.FG_IOU_THRESHOLD
    bg_iou_thresh = cfg.MODEL.RPN.BG_IOU_THRESHOLD

    batch_size_per_image = cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE
    positive_fraction = cfg.MODEL.RPN.POSITIVE_FRACTION

    rpn_only = cfg.MODEL.RPN_ONLY

    return RPN(anchor_generator, head,
            fg_iou_thresh, bg_iou_thresh,
            batch_size_per_image, positive_fraction,
            box_selector_train, box_selector_test, rpn_only)
