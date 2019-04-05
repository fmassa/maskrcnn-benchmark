from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou


class Target2ProposalAssigner(object):
    def __init__(self, proposal_matcher, fields_to_keep):
        self.box_similarity = boxlist_iou
        self.proposal_matcher = proposal_matcher
        self.fields_to_keep = fields_to_keep

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = self.box_similarity(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # Fast RCNN only need "labels" field for selecting the targets
        target = target.copy_with_fields(self.fields_to_keep)
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def __call__(self, proposals, targets):
        aligned_targets = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            matched_targets = self.match_targets_to_proposals(
                proposals_per_image, targets_per_image
            )
            aligned_targets.append(matched_targets)

        return aligned_targets
