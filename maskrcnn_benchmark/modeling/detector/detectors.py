# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .generalized_rcnn import GeneralizedRCNN
from ..backbone import build_backbone


_DETECTION_META_ARCHITECTURES = {"GeneralizedRCNN": GeneralizedRCNN}


def build_detection_model(cfg):
    meta_arch = _DETECTION_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    backbone = build_backbone(cfg)
    rpn = build_rpn(cfg, backbone.out_channels)
    roi_heads = build_roi_heads(cfg, backbone.out_channels)
    return meta_arch(backbone, rpn, roi_heads)


def build_rpn(cfg, in_channels):
    """
    This gives the gist of it. Not super important because it doesn't change as much
    """
    from ..rpn.rpn import AnchorGenerator, RPNHead, RPN

    anchor_sizes = cfg.MODEL.RPN.ANCHOR_SIZES
    aspect_ratios = cfg.MODEL.RPN.ASPECT_RATIOS

    anchor_generator = AnchorGenerator(
        anchor_sizes, aspect_ratios
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

def build_roi_heads(cfg, in_channels):
    from ..roi_heads.roi_heads2 import TwoMLPHead, FastRCNNPredictor, MaskRCNNHeads, MaskRCNNC4Predictor
    from maskrcnn_benchmark.modeling.poolers import Pooler

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

    fg_iou_thresh = cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD
    bg_iou_thresh = cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD

    batch_size_per_image = cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE
    positive_fraction = cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION

    pooler = Pooler(
        output_size=(resolution, resolution),
        scales=scales,
        sampling_ratio=sampling_ratio,
    )
    feature_extractor = TwoMLPHead(
            in_channels * resolution ** 2,
            representation_size,
            use_gn)
    box_predictor = FastRCNNPredictor(
            representation_size,
            num_classes,
            cls_agnostic_bbox_reg)

    mask_pooler = None
    mask_head = None
    mask_predictor = None
    mask_discretization_size = None
    if cfg.MODEL.MASK_ON:
        resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_MASK_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        mask_pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )

        use_gn = cfg.MODEL.ROI_MASK_HEAD.USE_GN
        layers = cfg.MODEL.ROI_MASK_HEAD.CONV_LAYERS
        dilation = cfg.MODEL.ROI_MASK_HEAD.DILATION
        mask_head = MaskRCNNHeads(in_channels, layers, dilation, use_gn)
        dim_reduced = cfg.MODEL.ROI_MASK_HEAD.CONV_LAYERS[-1]
        mask_predictor = MaskRCNNC4Predictor(in_channels, dim_reduced, num_classes)
        mask_discretization_size = cfg.MODEL.ROI_MASK_HEAD.RESOLUTION

    from ..roi_heads.roi_heads2 import RoIHeads

    bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
    return RoIHeads(pooler, feature_extractor, box_predictor,
            fg_iou_thresh, bg_iou_thresh, batch_size_per_image, positive_fraction, bbox_reg_weights,
            score_thresh,
            nms_thresh,
            detections_per_img,
            # Mask
            mask_pooler,
            mask_head,
            mask_predictor,
            mask_discretization_size
            )


