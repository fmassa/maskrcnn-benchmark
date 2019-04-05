# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .generalized_rcnn import GeneralizedRCNN
from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
#from ..roi_heads.roi_heads import build_roi_heads


_DETECTION_META_ARCHITECTURES = {"GeneralizedRCNN": GeneralizedRCNN}


def build_detection_model(cfg):
    meta_arch = _DETECTION_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    backbone = build_backbone(cfg)
    rpn = build_rpn(cfg, backbone.out_channels)
    roi_heads = build_roi_heads(cfg, backbone.out_channels)
    return meta_arch(backbone, rpn, roi_heads)


def build_roi_box_head(cfg, in_channels):
    from ..roi_heads.box_head.box_head import ROIBoxHead
    from ..roi_heads.box_head.roi_box_feature_extractors import FPN2MLPFeatureExtractor
    from ..roi_heads.box_head.roi_box_predictors import FPNPredictor
    from ..roi_heads.box_head.loss import FastRCNNLossComputation, Matcher, BalancedPositiveNegativeSampler, BoxCoder
    from ..roi_heads.box_head.inference import PostProcessor

    resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
    scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
    sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
    representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
    use_gn = cfg.MODEL.ROI_BOX_HEAD.USE_GN

    num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
    cls_agnostic_bbox_reg = cfg.MODEL.CLS_AGNOSTIC_BBOX_REG

    score_thresh = cfg.MODEL.ROI_HEADS.SCORE_THRESH
    nms_thresh = cfg.MODEL.ROI_HEADS.NMS
    detections_per_img = cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMG
    box_coder = BoxCoder(cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS)

    fg_iou_thresh = cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD
    bg_iou_thresh = cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD

    batch_size_per_image = cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE
    positive_fraction = cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION

    feature_extractor = FPN2MLPFeatureExtractor(resolution, scales, sampling_ratio, in_channels, representation_size, use_gn)

    roi_box_head = ROIBoxHead(
        feature_extractor,
        FPNPredictor(
            feature_extractor.out_channels,
            num_classes,
            cls_agnostic_bbox_reg),
        PostProcessor(
            score_thresh,
            nms_thresh,
            detections_per_img,
            box_coder,
            cls_agnostic_bbox_reg),
        FastRCNNLossComputation(
            Matcher(
                fg_iou_thresh,
                bg_iou_thresh,
                allow_low_quality_matches=False),
            BalancedPositiveNegativeSampler(
                batch_size_per_image,
                positive_fraction),
            box_coder,
            cls_agnostic_bbox_reg)
    )

    return roi_box_head

from ..roi_heads.mask_head.mask_head import build_roi_mask_head
from ..roi_heads.keypoint_head.keypoint_head import build_roi_keypoint_head
from ..roi_heads.roi_heads import CombinedROIHeads
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

