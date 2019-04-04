# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from .box_head.box_head import build_roi_box_head
from .mask_head.mask_head import build_roi_mask_head, keep_only_positive_boxes
from .keypoint_head.keypoint_head import build_roi_keypoint_head


class CombinedROIHeads(torch.nn.ModuleDict):
    """
    Combines a set of individual heads (for box prediction or masks) into a single
    head.
    """

    def __init__(self, heads, mask_share_box_feature_extractor, keypoint_share_box_feature_extractor):
        super(CombinedROIHeads, self).__init__(heads)

    def forward(self, features, proposals, targets=None):

        if self.training:
            # Faster R-CNN subsamples during training the proposals with a fixed
            # positive / negative ratio
            with torch.no_grad():
                proposals = self.box.loss_evaluator.subsample(proposals, targets)

        # extract features that will be fed to the final classifier. The
        # feature_extractor generally corresponds to the pooler + heads
        x = self.box.feature_extractor(features, proposals)
        # final classifier that converts the features into predictions
        class_logits, box_regression = self.box.predictor(x)

        if self.training:
            loss_classifier, loss_box_reg = self.box.loss_evaluator(
                class_logits, box_regression, proposals
            )
            loss = dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg)

            if 'mask' in self:
                # during training, only focus on positive boxes
                # all_proposals = proposals
                subset_proposals, positive_inds = keep_only_positive_boxes(proposals)
                x = self.mask.feature_extractor(features, subset_proposals)
                mask_logits = self.mask.predictor(x)
                loss_mask = self.mask.loss_evaluator(subset_proposals, mask_logits, targets)
                loss['loss_mask'] = loss_mask
            return x, proposals, loss
        else:
            result = self.box.post_processor((class_logits, box_regression), proposals)
            if 'mask' in self:
                x = self.mask.feature_extractor(features, proposals)
                mask_logits = self.mask.predictor(x)
                result = self.mask.post_processor(mask_logits, result)
            return x, result, {}


def build_roi_heads(cfg, in_channels):
    # individually create the heads, that will be combined together
    # afterwards
    roi_heads = []
    if cfg.MODEL.RETINANET_ON:
        return []

    if not cfg.MODEL.RPN_ONLY:
        roi_heads.append(("box", build_roi_box_head(cfg, in_channels)))
    if cfg.MODEL.MASK_ON:
        roi_heads.append(("mask", build_roi_mask_head(cfg, in_channels)))
    if cfg.MODEL.KEYPOINT_ON:
        roi_heads.append(("keypoint", build_roi_keypoint_head(cfg, in_channels)))

    # combine individual heads in a single module
    if roi_heads:
        roi_heads = CombinedROIHeads(roi_heads, cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR,
            cfg.MODEL.ROI_KEYPOINT_HEAD.SHARE_BOX_FEATURE_EXTRACTOR)

    return roi_heads
