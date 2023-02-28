# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from numpy.lib import pad
import torch
from torch import nn
from torch.nn import functional as F
from random import randint

from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.structures import ImageList, Instances, Boxes
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n

from ..backbone import Backbone, build_backbone
from ..postprocessing import detector_postprocess
from ..proposal_generator import build_proposal_generator
from ..roi_heads import build_roi_heads
from .build import META_ARCH_REGISTRY

from PIL import Image
import copy
from ..backbone.fpn import build_resnet_fpn_backbone
from ..backbone.clip_backbone import build_clip_language_encoder
from detectron2.utils.comm import gather_tensors, MILCrossEntropy
from detectron2.structures.boxes import Boxes

__all__ = ["CLIPFPN"]

@META_ARCH_REGISTRY.register()
class CLIPFPN(nn.Module):
    """
    Fast R-CNN style where the cropping is conducted on feature maps instead of raw images.
    It contains the following two components: 
    1. Localization branch: pretrained backbone+RPN or equivalent modules, and is able to output object proposals
    2. Recognition branch: is able to recognize zero-shot regions
    """
    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        language_encoder: nn.Module, 
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
        clip_crop_region_type: str = 'GT',
        use_clip_attpool: False,
        eval_proposal: False
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super().__init__()
        self.backbone = backbone
        self.lang_encoder = language_encoder
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads

        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        # input format, pixel mean and std for offline modules
        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"
        if np.sum(pixel_mean) < 3.0: # converrt pixel value to range [0.0, 1.0] by dividing 255.0
            assert input_format == 'RGB'
            self.div_pixel = True
        else:
            self.div_pixel = False
        
        self.clip_crop_region_type = clip_crop_region_type
        self.use_clip_attpool = use_clip_attpool # if True (C4+text_emb_as_classifier), use att_pool to replace default mean pool
        self.eval_proposal_only = eval_proposal

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        # build language encoder
        if cfg.MODEL.CLIP.GET_CONCEPT_EMB: # extract concept embeddings
            language_encoder = build_clip_language_encoder(cfg)
        else:
            language_encoder = None
        rpn = build_proposal_generator(cfg, backbone.output_shape()) # tbc: to be check
        roi_heads = build_roi_heads(cfg, backbone.bottom_up.output_shape()) # tbc: hard code 1/2 in roipooler scales


        return {
            "backbone": backbone,
            "proposal_generator": rpn, 
            "backbone": backbone,
            "language_encoder": language_encoder, 
            "roi_heads": roi_heads, 
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "clip_crop_region_type" : cfg.MODEL.CLIP.CROP_REGION_TYPE,
            "use_clip_attpool": cfg.MODEL.ROI_HEADS.NAME in ['CLIPRes5ROIHeads', 'CLIPStandardROIHeads'] and cfg.MODEL.CLIP.USE_TEXT_EMB_CLASSIFIER,
            "eval_proposal": cfg.MODEL.CLIP.EVAL_PROPOSAL
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def _scale_gt_box(self, boxes, scale: int, image_size: Tuple[int, int]):
        """
        Args:
            boxes: tensor
            scale: int
            image_size : Tuple[int, int]
        """
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        x1 = ctr_x - 0.5 * widths * scale
        y1 = ctr_y - 0.5 * heights * scale
        x2 = ctr_x + 0.5 * widths * scale
        y2 = ctr_y + 0.5 * heights + scale
        scaled_boxes = Boxes(torch.stack((x1, y1, x2, y2), dim=-1))
        scaled_boxes.clip(image_size)
        return scaled_boxes
            

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            return self.inference(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
        
        # localization branch: offline modules to get the region proposals
        images = self.preprocess_image(batched_inputs)
        bottom_up_features = self.backbone.bottom_up(images.tensor)
        if self.clip_crop_region_type == "GT":  # from ground-truth
            proposals = []
            for r_i, b_input in enumerate(batched_inputs): 
                this_gt = copy.deepcopy(b_input["instances"])  # Instance
                gt_boxes = this_gt._fields['gt_boxes'].to(self.device)
                this_gt._fields = {'proposal_boxes': gt_boxes, 'objectness_logits': torch.ones(gt_boxes.tensor.size(0)).to(self.device)}
                proposals.append(this_gt)
        elif self.clip_crop_region_type == "RPN": # from trained fpn & rpn
            fpn_features = self.backbone(images.tensor, bottom_up_features)
            proposals, rpn_loss = self.proposal_generator(images, fpn_features, gt_instances)

        # recognition branch: get 2D feature maps using the backbone of recognition branch
        if self.use_clip_attpool:
            _, detector_losses = self.roi_heads(batched_inputs, bottom_up_features, proposals, gt_instances, 
                                    res5=self.backbone.bottom_up.layer4,
                                    attnpool=self.backbone.bottom_up.attnpool, vis_period=self.vis_period)
        else:
           _, detector_losses = self.roi_heads(batched_inputs, bottom_up_features, proposals, gt_instances,
                                    vis_period=self.vis_period)

        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)
        #visualize_proposals(batched_inputs, proposals, self.input_format)

        losses = {}
        losses.update(detector_losses)
        losses.update(rpn_loss)
        return losses

    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 top-scoring predicted
        object proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch

    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training
        
        # localization branch: offline modules to get the region proposals
        images = self.preprocess_image(batched_inputs)
        bottom_up_features = self.backbone.bottom_up(images.tensor)
        if self.clip_crop_region_type == "GT":  # from ground-truth
            proposals = []
            for r_i, b_input in enumerate(batched_inputs): 
                this_gt = copy.deepcopy(b_input["instances"])  # Instance
                gt_boxes = this_gt._fields['gt_boxes'].to(self.device)
                this_gt._fields = {'proposal_boxes': gt_boxes, 'objectness_logits': torch.ones(gt_boxes.tensor.size(0)).to(self.device)}
                proposals.append(this_gt)
        elif self.clip_crop_region_type == "RPN": # from trained fpn & rpn
            fpn_features = self.backbone(images.tensor, bottom_up_features)
            proposals, _ = self.proposal_generator(images, fpn_features, None)        

        if not self.eval_proposal_only:
            # recognition branch: get 2D feature maps using the backbone of recognition branch
            if self.use_clip_attpool:
                results, _ = self.roi_heads(images, bottom_up_features, proposals, None, 
                res5=self.backbone.bottom_up.layer4,
                attnpool=self.backbone.bottom_up.attnpool)
            else:
                results, _ = self.roi_heads(images, bottom_up_features, proposals, None)
        else:
            results = proposals
        
        #visualize_proposals(batched_inputs, proposals, self.input_format)
        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return CLIPFPN._postprocess(results, batched_inputs, self.eval_proposal_only, proposals)
        else:
            return results

    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images. Use CLIP default processing (pixel mean & std).
        Note: Due to FPN size_divisibility, images are padded by right/bottom border. So FPN is consistent with C4 and GT boxes.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        if self.div_pixel:
            images = [((x / 255.0) - self.pixel_mean) / self.pixel_std for x in images]
        else:
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    @staticmethod
    def _postprocess(instances, batched_inputs: List[Dict[str, torch.Tensor]], isproposal, proposals):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, proposals_per_image, input_per_image in zip(
            instances, proposals, batched_inputs):
            height = input_per_image["height"]  # original image size, before resizing
            width = input_per_image["width"]  # original image size, before resizing
            r = detector_postprocess(results_per_image, height, width)
            pr = detector_postprocess(proposals_per_image, height, width)
            if isproposal:
                processed_results.append({"proposals": r})
            else:
                processed_results.append({"instances": r, "proposals": pr})
        return processed_results

