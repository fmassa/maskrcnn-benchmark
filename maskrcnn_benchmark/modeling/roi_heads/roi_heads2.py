import torch

import torch.nn.functional as F
from torch import nn

# inference
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms  # move to BoxList
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist  # move to BoxList
from maskrcnn_benchmark.modeling.box_coder import BoxCoder

# loss
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.layers import smooth_l1_loss

# TwoMLPHead
from maskrcnn_benchmark.modeling.make_layers import make_fc

# StandardRoiHead
from maskrcnn_benchmark.modeling.balanced_positive_negative_sampler import (
    BalancedPositiveNegativeSampler
)
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou  # move to BoxList
from maskrcnn_benchmark.modeling.rpn.rpn import generic_filter_proposals  # TODO remove from there
from maskrcnn_benchmark.modeling.rpn.rpn import apply_deltas_to_boxlists  # TODO remove from there

# Mask
from maskrcnn_benchmark.modeling.make_layers import make_conv3x3
from maskrcnn_benchmark.layers import Conv2d
from maskrcnn_benchmark.layers import ConvTranspose2d


class TwoMLPHead(nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self,
            # head
            in_channels, representation_size, use_gn
            ):
        super(TwoMLPHead, self).__init__()

        self.fc6 = make_fc(in_channels, representation_size, use_gn)
        self.fc7 = make_fc(representation_size, representation_size, use_gn)
        self.out_channels = representation_size

    def forward(self, x):
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x


class FastRCNNPredictor(nn.Module):
    def __init__(self, in_channels, num_classes, cls_agnostic_bbox_reg):
        super(FastRCNNPredictor, self).__init__()
        representation_size = in_channels

        self.cls_score = nn.Linear(representation_size, num_classes)
        num_bbox_reg_classes = 2 if cls_agnostic_bbox_reg else num_classes
        self.bbox_pred = nn.Linear(representation_size, num_bbox_reg_classes * 4)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for l in [self.cls_score, self.bbox_pred]:
            nn.init.constant_(l.bias, 0)

    def forward(self, x):
        if x.ndimension() == 4:
            assert list(x.shape[2:]) == [1, 1]
            x = x.view(x.size(0), -1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas


class FastRCNNInference(nn.Module):
    """
    From a set of classification scores, box regression and proposals,
    computes the post-processed boxes, and applies NMS to obtain the
    final results
    """

    def __init__(
        self,
        score_thresh=0.05,
        nms=0.5,
        detections_per_img=100,
        box_coder=None,
    ):
        """
        Arguments:
            score_thresh (float)
            nms (float)
            detections_per_img (int)
            box_coder (BoxCoder)
        """
        super(FastRCNNInference, self).__init__()
        self.score_thresh = score_thresh
        self.nms = nms
        self.detections_per_img = detections_per_img
        if box_coder is None:
            box_coder = BoxCoder(weights=(10., 10., 5., 5.))
        self.box_coder = box_coder

    def forward(self, class_logits, box_regression, boxes):
        """
        Arguments:
            x (tuple[tensor, tensor]): x contains the class logits
                and the box_regression from the model.
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for ech image

        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra fields labels and scores
        """
        class_prob = F.softmax(class_logits, -1)

        # TODO think about a representation of batch of boxes
        image_shapes = [box.size for box in boxes]
        boxes_per_image = [len(box) for box in boxes]

        num_classes = class_prob.shape[1]
        cls_agnostic_bbox_reg = box_regression.shape[1] // 4 != num_classes

        if cls_agnostic_bbox_reg:
            box_regression = box_regression[:, -4:]

        proposals = apply_deltas_to_boxlists(boxes, box_regression, self.box_coder)

        if cls_agnostic_bbox_reg:
            proposals = proposals.repeat(1, num_classes)

        results = self.filter_proposals(proposals, class_prob, image_shapes, boxes_per_image)
        return results

    def filter_proposals(self, boxes, scores, image_shapes, boxes_per_image):
        """Returns bounding-box detection results by thresholding on scores and
        applying non-maximum suppression (NMS).
        """
        def filter_fn(boxlist, scores, j):
            device = scores.device
            inds = (scores > self.score_thresh).nonzero().squeeze(1)
            boxlist = boxlist[inds]
            scores = scores[inds]
            boxlist = boxlist.clip_to_image(remove_empty=False)
            boxlist.add_field("scores", scores)
            boxlist = boxlist_nms(
                boxlist, self.nms
            )
            num_labels = len(boxlist)
            boxlist.add_field(
                "labels", torch.full((num_labels,), j + 1, dtype=torch.int64, device=device)
            )
            return boxlist

        def post_filter_fn(boxlist):
            number_of_detections = len(boxlist)
            # Limit to max_per_image detections **over all classes**
            if number_of_detections > self.detections_per_img > 0:
                cls_scores = boxlist.get_field("scores")
                image_thresh, _ = torch.kthvalue(
                    cls_scores.cpu(), number_of_detections - self.detections_per_img + 1
                )
                keep = cls_scores >= image_thresh.item()
                keep = torch.nonzero(keep).squeeze(1)
                boxlist = boxlist[keep]
            return boxlist

        boxes = boxes.reshape(sum(boxes_per_image), -1, 4)
        boxes = boxes[:, 1:]
        scores = scores[:, 1:]
        results = generic_filter_proposals(boxes, scores, image_shapes,
                boxes_per_image, 1, filter_fn=filter_fn, post_filter_fn=post_filter_fn)

        return results


def fastrcnn_loss(class_logits, box_regression, proposals, box_coder):
    """
    Computes the loss for Faster R-CNN.

    Arguments:
        class_logits (Tensor)
        box_regression (Tensor)

    Returns:
        classification_loss (Tensor)
        box_loss (Tensor)
    """

    device = class_logits.device

    regression_targets = []
    for proposals_per_image in proposals:
        matched_gt_boxes = proposals_per_image.get_field("matched_gt_boxes")
        regression_targets_per_image = box_coder.encode(matched_gt_boxes, proposals_per_image.bbox)
        regression_targets.append(regression_targets_per_image)

    labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
    regression_targets = cat(regression_targets, dim=0)

    classification_loss = F.cross_entropy(class_logits, labels)

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
    labels_pos = labels[sampled_pos_inds_subset]
    map_inds = 4 * labels_pos[:, None] + torch.tensor(
        [0, 1, 2, 3], device=device)

    box_loss = smooth_l1_loss(
        box_regression[sampled_pos_inds_subset[:, None], map_inds],
        regression_targets[sampled_pos_inds_subset],
        size_average=False,
        beta=1,
    )
    box_loss = box_loss / labels.numel()

    return classification_loss, box_loss




def keep_only_positive_boxes(boxes):
    """
    Given a set of BoxList containing the `labels` field,
    return a set of BoxList for which `labels > 0`.

    Arguments:
        boxes (list of BoxList)
    """
    assert isinstance(boxes, (list, tuple))
    assert isinstance(boxes[0], BoxList)
    assert boxes[0].has_field("labels")
    positive_boxes = []
    positive_inds = []
    num_boxes = 0
    for boxes_per_image in boxes:
        labels = boxes_per_image.get_field("labels")
        inds_mask = labels > 0
        inds = inds_mask.nonzero().squeeze(1)
        positive_boxes.append(boxes_per_image[inds])
        positive_inds.append(inds_mask)
    return positive_boxes, positive_inds


class MaskRCNNHeads(nn.ModuleDict):
    """
    Heads for FPN for classification
    """

    def __init__(self, in_channels, layers, dilation, use_gn):
        """
        Arguments:
            num_classes (int): number of output classes
            input_size (int): number of channels of the input once it's flattened
            representation_size (int): size of the intermediate representation
        """
        super(MaskRCNNHeads, self).__init__()

        input_size = in_channels

        next_feature = input_size
        for layer_idx, layer_features in enumerate(layers, 1):
            layer_name = "mask_fcn{}".format(layer_idx)
            module = make_conv3x3(
                next_feature, layer_features,
                dilation=dilation, stride=1, use_gn=use_gn
            )
            self[layer_name] = module
            next_feature = layer_features
        self.out_channels = layer_features

    def forward(self, x):
        for module in self.values():
            x = F.relu(module(x))

        return x


class MaskRCNNC4Predictor(nn.Module):
    def __init__(self, in_channels, dim_reduced, num_classes):
        super(MaskRCNNC4Predictor, self).__init__()
        num_inputs = in_channels

        self.conv5_mask = ConvTranspose2d(num_inputs, dim_reduced, 2, 2, 0)
        self.mask_fcn_logits = Conv2d(dim_reduced, num_classes, 1, 1, 0)

        for name, param in self.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                # Caffe2 implementation uses MSRAFill, which in fact
                # corresponds to kaiming_normal_ in PyTorch
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        x = F.relu(self.conv5_mask(x))
        return self.mask_fcn_logits(x)


# TODO check if want to return a single BoxList or a composite
# object
def maskrcnn_inference(x, boxes):
    """
    From the results of the CNN, post process the masks
    by taking the mask corresponding to the class with max
    probability (which are of fixed size and directly output
    by the CNN) and return the masks in the mask field of the BoxList.

    Arguments:
        x (Tensor): the mask logits
        boxes (list[BoxList]): bounding boxes that are used as
            reference, one for ech image

    Returns:
        results (list[BoxList]): one BoxList for each image, containing
            the extra field mask
    """
    mask_prob = x.sigmoid()

    # select masks coresponding to the predicted classes
    num_masks = x.shape[0]
    labels = [bbox.get_field("labels") for bbox in boxes]
    labels = torch.cat(labels)
    index = torch.arange(num_masks, device=labels.device)
    mask_prob = mask_prob[index, labels][:, None]

    boxes_per_image = [len(box) for box in boxes]
    mask_prob = mask_prob.split(boxes_per_image, dim=0)

    results = []
    for prob, box in zip(mask_prob, boxes):
        bbox = BoxList(box.bbox, box.size, mode="xyxy")
        for field in box.fields():
            bbox.add_field(field, box.get_field(field))
        bbox.add_field("mask", prob)
        results.append(bbox)

    return results


def project_masks_on_boxes(segmentation_masks, proposals, discretization_size):
    """
    Given segmentation masks and the bounding boxes corresponding
    to the location of the masks in the image, this function
    crops and resizes the masks in the position defined by the
    boxes. This prepares the masks for them to be fed to the
    loss computation as the targets.

    Arguments:
        segmentation_masks: an instance of SegmentationMask
        proposals: an instance of BoxList
    """
    masks = []
    M = discretization_size
    device = proposals.bbox.device
    proposals = proposals.convert("xyxy")
    assert segmentation_masks.size == proposals.size, "{}, {}".format(
        segmentation_masks, proposals
    )
    # TODO put the proposals on the CPU, as the representation for the
    # masks is not efficient GPU-wise (possibly several small tensors for
    # representing a single instance mask)
    proposals = proposals.bbox.to(torch.device("cpu"))
    for segmentation_mask, proposal in zip(segmentation_masks, proposals):
        # crop the masks, resize them to the desired resolution and
        # then convert them to the tensor representation,
        # instead of the list representation that was used
        cropped_mask = segmentation_mask.crop(proposal)
        scaled_mask = cropped_mask.resize((M, M))
        mask = scaled_mask.convert(mode="mask")
        masks.append(mask)
    if len(masks) == 0:
        return torch.empty(0, dtype=torch.float32, device=device)
    return torch.stack(masks, dim=0).to(device, dtype=torch.float32)


def maskrcnn_loss(mask_logits, proposals, discretization_size):
    """
    Arguments:
        proposals (list[BoxList])
        mask_logits (Tensor)
        targets (list[BoxList])

    Return:
        mask_loss (Tensor): scalar tensor containing the loss
    """

    labels = [p.get_field("labels") for p in proposals]
    mask_targets = [project_masks_on_boxes(proposal.get_field("matched_masks"), proposal, discretization_size)
            for proposal in proposals]

    labels = cat(labels, dim=0)
    mask_targets = cat(mask_targets, dim=0)

    # torch.mean (in binary_cross_entropy_with_logits) doesn't
    # accept empty tensors, so handle it separately
    if mask_targets.numel() == 0:
        return mask_logits.sum() * 0

    mask_loss = F.binary_cross_entropy_with_logits(
        mask_logits[torch.arange(labels.shape[0], device=labels.device), labels], mask_targets
    )
    return mask_loss





# the next two functions should be merged inside Masker
# but are kept here for the moment while we need them
# temporarily gor paste_mask_in_image
def expand_boxes(boxes, scale):
    w_half = (boxes[:, 2] - boxes[:, 0]) * .5
    h_half = (boxes[:, 3] - boxes[:, 1]) * .5
    x_c = (boxes[:, 2] + boxes[:, 0]) * .5
    y_c = (boxes[:, 3] + boxes[:, 1]) * .5

    w_half *= scale
    h_half *= scale

    boxes_exp = torch.zeros_like(boxes)
    boxes_exp[:, 0] = x_c - w_half
    boxes_exp[:, 2] = x_c + w_half
    boxes_exp[:, 1] = y_c - h_half
    boxes_exp[:, 3] = y_c + h_half
    return boxes_exp


def expand_masks(mask, padding):
    N = mask.shape[0]
    M = mask.shape[-1]
    pad2 = 2 * padding
    scale = float(M + pad2) / M
    padded_mask = mask.new_zeros((N, 1, M + pad2, M + pad2))
    padded_mask[:, :, padding:-padding, padding:-padding] = mask
    return padded_mask, scale

from maskrcnn_benchmark.layers.misc import interpolate
def paste_mask_in_image(mask, box, im_h, im_w, thresh=0.5, padding=1):
    padded_mask, scale = expand_masks(mask[None], padding=padding)
    mask = padded_mask[0, 0]
    box = expand_boxes(box[None], scale)[0]
    box = box.to(dtype=torch.int32)

    TO_REMOVE = 1
    w = int(box[2] - box[0] + TO_REMOVE)
    h = int(box[3] - box[1] + TO_REMOVE)
    w = max(w, 1)
    h = max(h, 1)

    # Set shape to [batchxCxHxW]
    mask = mask.expand((1, 1, -1, -1))

    # Resize mask
    mask = mask.to(torch.float32)
    mask = interpolate(mask, size=(h, w), mode='bilinear', align_corners=False)
    mask = mask[0][0]

    if thresh >= 0:
        mask = mask > thresh
    else:
        # for visualization and debugging, we also
        # allow it to return an unmodified mask
        mask = (mask * 255).to(torch.uint8)

    im_mask = torch.zeros((im_h, im_w), dtype=torch.uint8)
    x_0 = max(box[0], 0)
    x_1 = min(box[2] + 1, im_w)
    y_0 = max(box[1], 0)
    y_1 = min(box[3] + 1, im_h)

    im_mask[y_0:y_1, x_0:x_1] = mask[
        (y_0 - box[1]) : (y_1 - box[1]), (x_0 - box[0]) : (x_1 - box[0])
    ]
    return im_mask


class Masker(object):
    """
    Projects a set of masks in an image on the locations
    specified by the bounding boxes
    """

    def __init__(self, threshold=0.5, padding=1):
        self.threshold = threshold
        self.padding = padding

    def forward_single_image(self, masks, boxes):
        boxes = boxes.convert("xyxy")
        im_w, im_h = boxes.size
        res = [
            paste_mask_in_image(mask[0], box, im_h, im_w, self.threshold, self.padding)
            for mask, box in zip(masks, boxes.bbox)
        ]
        if len(res) > 0:
            res = torch.stack(res, dim=0)[:, None]
        else:
            res = masks.new_empty((0, 1, masks.shape[-2], masks.shape[-1]))
        return res

    def __call__(self, masks, boxes):
        if isinstance(boxes, BoxList):
            boxes = [boxes]

        # Make some sanity check
        assert len(boxes) == len(masks), "Masks and boxes should have the same length."

        # TODO:  Is this JIT compatible?
        # If not we should make it compatible.
        results = []
        for mask, box in zip(masks, boxes):
            assert mask.shape[0] == len(box), "Number of objects should be the same."
            result = self.forward_single_image(mask, box)
            results.append(result)
        return results




class RoIHeads(torch.nn.Module):
    def __init__(self,
            box_roi_pool,
            box_head,
            box_predictor,
            # Faster R-CNN training
            fg_iou_thresh, bg_iou_thresh,
            batch_size_per_image, positive_fraction,
            bbox_reg_weights,
            # Faster R-CNN inference
            score_thresh,
            nms_thresh,
            detections_per_img,
            # Mask
            mask_roi_pool=None,
            mask_head=None,
            mask_predictor=None,
            mask_discretization_size=None,
            ):
        super(RoIHeads, self).__init__()

        self.box_similarity = boxlist_iou
        # assign ground-truth boxes for each proposal
        self.proposal_matcher = Matcher(
            fg_iou_thresh,
            bg_iou_thresh,
            allow_low_quality_matches=False)

        self.fg_bg_sampler = BalancedPositiveNegativeSampler(
            batch_size_per_image,
            positive_fraction)

        self.box_coder = BoxCoder(bbox_reg_weights)
        self.fields_to_keep = ["labels"]

        self.box_roi_pool = box_roi_pool
        self.box_head = box_head
        self.box_predictor = box_predictor

        self.fastrcnn_inference = FastRCNNInference(score_thresh, nms_thresh,
                detections_per_img, self.box_coder)

        # masks
        if mask_predictor is not None:
            self.fields_to_keep.append("masks")
        self.mask_roi_pool = mask_roi_pool
        self.mask_head = mask_head
        self.mask_predictor = mask_predictor
        self.mask_discretization_size = mask_discretization_size

    def assign_targets_to_proposals(self, proposals, targets):
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            match_quality_matrix = self.box_similarity(targets_per_image, proposals_per_image)
            matched_idxs = self.proposal_matcher(match_quality_matrix)
            # Fast RCNN only need "labels" field for selecting the targets
            targets_per_image = targets_per_image.copy_with_fields(self.fields_to_keep)
            # get the targets corresponding GT for each proposal
            # NB: need to clamp the indices because we can have a single
            # GT in the image, and matched_idxs can be -2, which goes
            # out of bounds
            matched_targets = targets_per_image[matched_idxs.clamp(min=0)]
            # proposals_per_image.add_field("matched_targets", matched_targets)
            # proposals_per_image.add_field("matched_idxs", matched_idxs)
            proposals_per_image.add_field("matched_gt_boxes", matched_targets.bbox)
            for name in matched_targets.fields():
                value = matched_targets.get_field(name)
                proposals_per_image.add_field("matched_" + name, value)

            labels_per_image = proposals_per_image.get_field("matched_labels")
            labels_per_image = labels_per_image.clone().to(dtype=torch.int64)

            # Label background (below the low threshold)
            bg_inds = matched_idxs == self.proposal_matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_inds] = 0

            # Label ignore proposals (between low and high thresholds)
            ignore_inds = matched_idxs == self.proposal_matcher.BETWEEN_THRESHOLDS
            labels_per_image[ignore_inds] = -1  # -1 is ignored by sampler

            proposals_per_image.add_field("labels", labels_per_image)

    def subsample_proposals(self, proposals):
        proposals = list(proposals)

        labels = [p.get_field("labels") for p in proposals]
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)

        # distributed sampled proposals, that were obtained on all feature maps
        # concatenated via the fg_bg_sampler, into individual feature map levels
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(
            zip(sampled_pos_inds, sampled_neg_inds)
        ):
            img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1)
            proposals_per_image = proposals[img_idx][img_sampled_inds]
            proposals[img_idx] = proposals_per_image

        return proposals

    def add_gt_proposals(self, proposals, targets):
        """
        Arguments:
            proposals: list[BoxList]
            targets: list[BoxList]
        """

        gt_boxes = [target.copy_with_fields([]) for target in targets]

        proposals = [
            cat_boxlist((proposal, gt_box))
            for proposal, gt_box in zip(proposals, gt_boxes)
        ]

        return proposals

    def forward(self, features, proposals, targets=None):
        # remove all fields that might have been attached to the proposals
        # they are not used by RoIHeads
        proposals = [p.copy_with_fields([]) for p in proposals]
        if self.training:
            assert targets is not None
            # append ground-truth bboxes to proposals
            proposals = self.add_gt_proposals(proposals, targets)

            self.assign_targets_to_proposals(proposals, targets)
            proposals = self.subsample_proposals(proposals)

        box_features = self.box_roi_pool(features, proposals)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)

        result, losses = [], {}
        if self.training:
            loss_classifier, loss_box_reg = fastrcnn_loss(
                class_logits, box_regression, proposals,
                self.box_coder)
            losses = dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg)
        else:
            result = self.fastrcnn_inference(class_logits, box_regression, proposals)

        if self.mask_predictor:
            mask_proposals = result
            if self.training:
                # during training, only focus on positive boxes
                mask_proposals, positive_inds = keep_only_positive_boxes(proposals)
            mask_features = self.mask_roi_pool(features, mask_proposals)
            mask_features = self.mask_head(mask_features)
            mask_logits = self.mask_predictor(mask_features)

            loss_mask = {}
            if self.training:
                loss_mask = maskrcnn_loss(mask_logits, mask_proposals, self.mask_discretization_size)
                loss_mask = dict(loss_mask=loss_mask)
            else:
                result = maskrcnn_inference(mask_logits, mask_proposals)

            losses.update(loss_mask)

        return result, losses
