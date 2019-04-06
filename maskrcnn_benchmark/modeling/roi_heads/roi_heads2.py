import torch

import torch.nn.functional as F
from torch import nn

# inference
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
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
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou


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
        cls_agnostic_bbox_reg=False
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
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg

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
        concat_boxes = torch.cat([a.bbox for a in boxes], dim=0)

        if self.cls_agnostic_bbox_reg:
            box_regression = box_regression[:, -4:]
        proposals = self.box_coder.decode(
            box_regression.view(sum(boxes_per_image), -1), concat_boxes
        )
        if self.cls_agnostic_bbox_reg:
            proposals = proposals.repeat(1, class_prob.shape[1])

        num_classes = class_prob.shape[1]

        proposals = proposals.split(boxes_per_image, dim=0)
        class_prob = class_prob.split(boxes_per_image, dim=0)

        results = []
        for prob, boxes_per_img, image_shape in zip(
            class_prob, proposals, image_shapes
        ):
            boxlist = self.prepare_boxlist(boxes_per_img, prob, image_shape)
            boxlist = boxlist.clip_to_image(remove_empty=False)
            boxlist = self.filter_results(boxlist, num_classes)
            results.append(boxlist)
        return results

    def prepare_boxlist(self, boxes, scores, image_shape):
        """
        Returns BoxList from `boxes` and adds probability scores information
        as an extra field
        `boxes` has shape (#detections, 4 * #classes), where each row represents
        a list of predicted bounding boxes for each of the object classes in the
        dataset (including the background class). The detections in each row
        originate from the same object proposal.
        `scores` has shape (#detection, #classes), where each row represents a list
        of object detection confidence scores for each of the object classes in the
        dataset (including the background class). `scores[i, j]`` corresponds to the
        box at `boxes[i, j * 4:(j + 1) * 4]`.
        """
        boxes = boxes.reshape(-1, 4)
        scores = scores.reshape(-1)
        boxlist = BoxList(boxes, image_shape, mode="xyxy")
        boxlist.add_field("scores", scores)
        return boxlist

    def filter_results(self, boxlist, num_classes):
        """Returns bounding-box detection results by thresholding on scores and
        applying non-maximum suppression (NMS).
        """
        # unwrap the boxlist to avoid additional overhead.
        # if we had multi-class NMS, we could perform this directly on the boxlist
        boxes = boxlist.bbox.reshape(-1, num_classes * 4)
        scores = boxlist.get_field("scores").reshape(-1, num_classes)

        device = scores.device
        result = []
        # Apply threshold on detection probabilities and apply NMS
        # Skip j = 0, because it's the background class
        inds_all = scores > self.score_thresh
        for j in range(1, num_classes):
            inds = inds_all[:, j].nonzero().squeeze(1)
            scores_j = scores[inds, j]
            boxes_j = boxes[inds, j * 4 : (j + 1) * 4]
            boxlist_for_class = BoxList(boxes_j, boxlist.size, mode="xyxy")
            boxlist_for_class.add_field("scores", scores_j)
            boxlist_for_class = boxlist_nms(
                boxlist_for_class, self.nms
            )
            num_labels = len(boxlist_for_class)
            boxlist_for_class.add_field(
                "labels", torch.full((num_labels,), j, dtype=torch.int64, device=device)
            )
            result.append(boxlist_for_class)

        result = cat_boxlist(result)
        number_of_detections = len(result)

        # Limit to max_per_image detections **over all classes**
        if number_of_detections > self.detections_per_img > 0:
            cls_scores = result.get_field("scores")
            image_thresh, _ = torch.kthvalue(
                cls_scores.cpu(), number_of_detections - self.detections_per_img + 1
            )
            keep = cls_scores >= image_thresh.item()
            keep = torch.nonzero(keep).squeeze(1)
            result = result[keep]
        return result



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
    if False:#self.cls_agnostic_bbox_reg:
        map_inds = torch.tensor([4, 5, 6, 7], device=device)
    else:
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


class StandardRoiHeads(torch.nn.Module):
    def __init__(self, box_roi_pool, box_head,
            box_predictor,
            fg_iou_thresh, bg_iou_thresh,
            batch_size_per_image, positive_fraction,
            bbox_reg_weights,
            #
            score_thresh,
            nms_thresh,
            detections_per_img,
            cls_agnostic_bbox_reg
            ):
        super(StandardRoiHeads, self).__init__()

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
                detections_per_img, self.box_coder, cls_agnostic_bbox_reg)

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


    def forward(self, features, proposals, targets=None):
        if self.training:
            self.assign_targets_to_proposals(proposals, targets)
            proposals = self.subsample_proposals(proposals)

        box_features = self.box_roi_pool(features, proposals)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)

        if self.training:
            loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, proposals, self.box_coder)
            return 0, [], dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg)

        result = self.fastrcnn_inference(class_logits, box_regression, proposals)
        return 0, result, {}
