import torch

import torch.nn.functional as F
from torch import nn

from torchvision.ops import boxes as box_ops

# inference
from maskrcnn_benchmark.modeling.box_coder import BoxCoder

# StandardRoiHead
from maskrcnn_benchmark.modeling.balanced_positive_negative_sampler import (
    BalancedPositiveNegativeSampler
)
from maskrcnn_benchmark.modeling.matcher import Matcher

# Mask
from maskrcnn_benchmark.layers import Conv2d
from maskrcnn_benchmark.layers import ConvTranspose2d
from maskrcnn_benchmark.layers.misc import interpolate

from torchvision.ops import roi_align


class TwoMLPHead(nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self,
            # head
            in_channels, representation_size, use_gn
            ):
        super(TwoMLPHead, self).__init__()

        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)
        self.out_channels = representation_size
        for fc in self.children():
            nn.init.kaiming_uniform_(fc.weight, a=1)
            nn.init.constant_(fc.bias, 0)

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

def multiclass_nms(boxes, prob, score_thresh, nms_thresh):
    device = boxes.device
    selected_boxes = []
    selected_scores = []
    selected_labels = []
    # remove predictions with the background label
    prob = prob[:, 1:]
    boxes= boxes[:, 1:]
    # split boxes and scores on a per-class list of tensors
    boxes = boxes.unbind(1)
    prob = prob.unbind(1)
    for class_id, (boxes_for_class, prob_for_class) in enumerate(zip(boxes, prob), 1):
        # remove low scoring boxes
        inds = torch.nonzero(prob_for_class > score_thresh).squeeze(1)
        prob_for_class = prob_for_class[inds]
        boxes_for_class = boxes_for_class[inds]
        # non-maximum suppression
        keep = box_ops.nms(boxes_for_class, prob_for_class, nms_thresh)
        boxes_for_class = boxes_for_class[keep]
        prob_for_class = prob_for_class[keep]
        # make label tensor
        num_boxes = len(boxes_for_class)
        label = torch.full((num_boxes,), class_id, dtype=torch.int64, device=device)
        selected_boxes.append(boxes_for_class)
        selected_scores.append(prob_for_class)
        selected_labels.append(label)
    selected_boxes = torch.cat(selected_boxes, dim=0)
    selected_scores = torch.cat(selected_scores, dim=0)
    selected_labels = torch.cat(selected_labels, dim=0)
    return selected_boxes, selected_scores, selected_labels

def keep_max_detections(boxes, scores, labels, detections_per_img):
    number_of_detections = len(boxes)
    # Limit to max_per_image detections **over all classes**
    if number_of_detections > detections_per_img > 0:
        image_thresh, _ = torch.kthvalue(
            scores.cpu(), number_of_detections - detections_per_img + 1
        )
        keep = scores >= image_thresh.item()
        keep = torch.nonzero(keep).squeeze(1)
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]
    return boxes, scores, labels


def fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
    """
    Computes the loss for Faster R-CNN.

    Arguments:
        class_logits (Tensor)
        box_regression (Tensor)

    Returns:
        classification_loss (Tensor)
        box_loss (Tensor)
    """

    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    classification_loss = F.cross_entropy(class_logits, labels)

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
    labels_pos = labels[sampled_pos_inds_subset]
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, -1, 4)

    box_loss = F.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        reduction="sum",
    )
    box_loss = box_loss / labels.numel()

    return classification_loss, box_loss



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
            conv = Conv2d(next_feature, layer_features, kernel_size=3,
                    stride=1, padding=dilation, dilation=dilation)
            nn.init.kaiming_normal_(
                conv.weight, mode="fan_out", nonlinearity="relu"
            )
            nn.init.constant_(conv.bias, 0)
            self[layer_name] = conv
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


def maskrcnn_inference(x, labels):
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
    boxes_per_image = [len(l) for l in labels]
    labels = torch.cat(labels)
    index = torch.arange(num_masks, device=labels.device)
    mask_prob = mask_prob[index, labels][:, None]

    mask_prob = mask_prob.split(boxes_per_image, dim=0)

    return mask_prob


def project_masks_on_boxes(gt_masks, boxes, matched_idxs, M):
    """
    Given segmentation masks and the bounding boxes corresponding
    to the location of the masks in the image, this function
    crops and resizes the masks in the position defined by the
    boxes. This prepares the masks for them to be fed to the
    loss computation as the targets.
    """
    matched_idxs = matched_idxs.float()
    device = boxes.device
    rois = torch.cat([matched_idxs[:, None], boxes], dim=1)
    return roi_align(gt_masks[:, None].float(), rois, (M, M), 1)[:, 0]


def maskrcnn_loss(mask_logits, proposals, gt_masks, gt_labels, mask_matched_idxs, discretization_size):
    """
    Arguments:
        proposals (list[BoxList])
        mask_logits (Tensor)
        targets (list[BoxList])

    Return:
        mask_loss (Tensor): scalar tensor containing the loss
    """

    labels = [l[idxs] for l, idxs in zip(gt_labels, mask_matched_idxs)]
    mask_targets = [project_masks_on_boxes(m, p, i, discretization_size) for m, p, i in zip(gt_masks, proposals, mask_matched_idxs)]

    labels = torch.cat(labels, dim=0)
    mask_targets = torch.cat(mask_targets, dim=0)

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

    def forward_single_image(self, masks, prediction):
        # boxes = boxes.convert("xyxy")
        im_h, im_w = prediction["image_size"].tolist()
        res = [
            paste_mask_in_image(mask[0], box, im_h, im_w, self.threshold, self.padding)
            for mask, box in zip(masks, prediction["boxes"])
        ]
        if len(res) > 0:
            res = torch.stack(res, dim=0)[:, None]
        else:
            res = masks.new_empty((0, 1, masks.shape[-2], masks.shape[-1]))
        return res

    def __call__(self, masks, prediction):
        if isinstance(prediction, dict):
            prediction = [prediction]

        # Make some sanity check
        assert len(prediction) == len(masks), "Masks and boxes should have the same length."

        # TODO:  Is this JIT compatible?
        # If not we should make it compatible.
        results = []
        for mask, p in zip(masks, prediction):
            assert mask.shape[0] == len(p["boxes"]), "Number of objects should be the same."
            result = self.forward_single_image(mask, p)
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

        self.box_similarity = box_ops.box_iou
        # assign ground-truth boxes for each proposal
        self.proposal_matcher = Matcher(
            fg_iou_thresh,
            bg_iou_thresh,
            allow_low_quality_matches=False)

        self.fg_bg_sampler = BalancedPositiveNegativeSampler(
            batch_size_per_image,
            positive_fraction)

        self.box_coder = BoxCoder(bbox_reg_weights)

        self.box_roi_pool = box_roi_pool
        self.box_head = box_head
        self.box_predictor = box_predictor

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img

        self.has_mask = False

        # masks
        if mask_predictor is not None:
            self.has_mask = True
        self.mask_roi_pool = mask_roi_pool
        self.mask_head = mask_head
        self.mask_predictor = mask_predictor
        self.mask_discretization_size = mask_discretization_size

    def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels):
        all_matched_idxs = []
        labels = []
        for proposals_per_image, gt_boxes_per_image, gt_labels_in_image in zip(proposals, gt_boxes, gt_labels):
            match_quality_matrix = self.box_similarity(gt_boxes_per_image, proposals_per_image)
            matched_idxs = self.proposal_matcher(match_quality_matrix)

            # gt_labels_in_image = targets_per_image.get_field("labels")
            labels_per_image = gt_labels_in_image[matched_idxs.clamp(min=0)]
            labels_per_image = labels_per_image.to(dtype=torch.int64)

            # Label background (below the low threshold)
            bg_inds = matched_idxs == self.proposal_matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_inds] = 0

            # Label ignore proposals (between low and high thresholds)
            ignore_inds = matched_idxs == self.proposal_matcher.BETWEEN_THRESHOLDS
            labels_per_image[ignore_inds] = -1  # -1 is ignored by sampler

            all_matched_idxs.append(matched_idxs.clamp(min=0))
            labels.append(labels_per_image)
        return all_matched_idxs, labels

    def subsample(self, labels):
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_inds = []
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(
            zip(sampled_pos_inds, sampled_neg_inds)
        ):
            img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1)
            sampled_inds.append(img_sampled_inds)
        return sampled_inds

    def add_gt_proposals(self, proposals, gt_boxes):
        """
        Arguments:
            proposals: list[BoxList]
            targets: list[BoxList]
        """

        proposals = [
            torch.cat((proposal, gt_box))
            for proposal, gt_box in zip(proposals, gt_boxes)
        ]

        return proposals

    def check_targets(self, targets):
        assert targets is not None
        assert all("boxes" in t for t in targets)
        assert all("labels" in t for t in targets)
        if self.has_mask:
            assert all("masks" in t for t in targets)

    def select_training_samples(self, proposals, targets):
        self.check_targets(targets)
        gt_boxes = [t["boxes"] for t in targets]
        gt_labels = [t["labels"] for t in targets]

        # append ground-truth bboxes to propos
        proposals = self.add_gt_proposals(proposals, gt_boxes)

        # get matching gt indices for each proposal
        matched_idxs, labels = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels)
        # sample a fixed proportion of positive-negative proposals
        sampled_inds = self.subsample(labels)
        num_images = len(proposals)
        for img_id in range(num_images):
            img_sampled_inds = sampled_inds[img_id]
            proposals[img_id] = proposals[img_id][img_sampled_inds]
            labels[img_id] = labels[img_id][img_sampled_inds]
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]

        regression_targets = []
        for proposals_per_image, gt_boxes_in_image, matched_idxs_in_image in zip(
                proposals, gt_boxes, matched_idxs):
            matched_gt_boxes = gt_boxes_in_image[matched_idxs_in_image]
            regression_targets_per_image = self.box_coder.encode(matched_gt_boxes, proposals_per_image)
            regression_targets.append(regression_targets_per_image)

        return proposals, matched_idxs, labels, regression_targets

    def postprocess_detections(self, class_logits, box_regression, proposals, image_shapes):
        boxes_per_image = [len(boxes_in_image) for boxes_in_image in proposals]

        concat_proposals = torch.cat(proposals, dim=0)
        pred_boxes = self.box_coder.decode(
            box_regression.view(sum(boxes_per_image), -1), concat_proposals
        )

        pred_boxes = pred_boxes.reshape(sum(boxes_per_image), -1, 4)

        pred_scores = F.softmax(class_logits, -1)

        # split boxes and scores per image
        pred_boxes = pred_boxes.split(boxes_per_image, 0)
        pred_scores = pred_scores.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        for boxes, scores, image_shape in zip(pred_boxes, pred_scores, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)
            boxes, scores, labels = multiclass_nms(boxes, scores, self.score_thresh, self.nms_thresh)
            boxes, scores, labels = keep_max_detections(boxes, scores, labels, self.detections_per_img)
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        return all_boxes, all_scores, all_labels

    def forward(self, features, proposals, image_shapes, targets=None):
        if self.training:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)

        box_features = self.box_roi_pool(features, proposals)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)

        result, losses = [], {}
        output = []
        if self.training:
            loss_classifier, loss_box_reg = fastrcnn_loss(
                class_logits, box_regression, labels, regression_targets)
            losses = dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg)
        else:
            boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)

            for b, s, l, image_shape in zip(boxes, scores, labels, image_shapes):
                r = {}
                r["boxes"] = b
                r["labels"] = l
                r["scores"] = s
                r["image_size"] = torch.as_tensor(image_shape)
                result.append(r)

        if self.has_mask:
            mask_proposals = [p["boxes"] for p in result]
            if self.training:
                # during training, only focus on positive boxes
                num_images = len(proposals)
                mask_proposals = []
                mask_matched_idxs = []
                for img_id in range(num_images):
                    pos = torch.nonzero(labels[img_id] > 0).squeeze(1)
                    mask_proposals.append(proposals[img_id][pos])
                    mask_matched_idxs.append(matched_idxs[img_id][pos])

            mask_features = self.mask_roi_pool(features, mask_proposals)
            mask_features = self.mask_head(mask_features)
            mask_logits = self.mask_predictor(mask_features)

            loss_mask = {}
            if self.training:
                gt_masks = [t["masks"] for t in targets]
                gt_labels = [t["labels"] for t in targets]
                loss_mask = maskrcnn_loss(mask_logits, mask_proposals,
                        gt_masks, gt_labels, mask_matched_idxs, self.mask_discretization_size)
                loss_mask = dict(loss_mask=loss_mask)
            else:
                labels = [r["labels"] for r in result]
                masks_probs = maskrcnn_inference(mask_logits, labels)
                for mask_prob, r in zip(masks_probs, result):
                    r["mask"] = mask_prob

            losses.update(loss_mask)

        return result, losses
