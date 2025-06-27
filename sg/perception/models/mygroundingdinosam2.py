# groundingdino model
from groundingdino.util.inference import Model as GDModel
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sg.utils.utils import grounding_dino_config_path, grounding_dino_checkpoint_path
from sg.utils.utils import sam2_model_cfg_path, sam2_checkpoint_path
import cv2
import torchvision
import torch
import numpy as np
import supervision as sv
from supervision.draw.color import Color, ColorPalette
from typing import Union
import dataclasses

class MyGroundingDINOSAM2():
    def __init__(
        self,
        box_threshold=0.3,
        text_threshold=0.3,
        nms_threshold=0.5,
        device="cuda"
    ): 
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.nms_threshold = nms_threshold
        self.device = device

        ### Initialize the Grounding DINO model ###
        self.grounding_dino_model = GDModel(
            model_config_path=grounding_dino_config_path, 
            model_checkpoint_path=grounding_dino_checkpoint_path, 
            device=self.device
        )

        ### Initialize the SAM2 model ###
        sam2_model = build_sam2(sam2_model_cfg_path, sam2_checkpoint_path, device=self.device)
        self.sam2_predictor = SAM2ImagePredictor(sam2_model)

    def run(self, image, classes: list) -> sv.Detections:
        detections = self.grounding_dino_model.predict_with_classes(
            image=cv2.cvtColor(image, cv2.COLOR_RGB2BGR), # This function expects a BGR image...
            classes=classes,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold
        )

        if len(detections.class_id) > 0:
            ### Non-maximum suppression ###
            # print(f"Before NMS: {len(detections.xyxy)} boxes")
            nms_idx = torchvision.ops.nms(
                torch.from_numpy(detections.xyxy), 
                torch.from_numpy(detections.confidence), 
                self.nms_threshold
            ).numpy().tolist()
            # print(f"After NMS: {len(detections.xyxy)} boxes")
            detections.xyxy = detections.xyxy[nms_idx]
            detections.confidence = detections.confidence[nms_idx]
            detections.class_id = detections.class_id[nms_idx]
            
            # Somehow some detections will have class_id=-1, remove them
            # valid_idx = detections.class_id != -1
            valid_idx = [i for i, val in enumerate(detections.class_id) if (val is not None and val != -1)]
            detections.xyxy = detections.xyxy[valid_idx]
            detections.confidence = detections.confidence[valid_idx]
            detections.class_id = detections.class_id[valid_idx]

        # detections.class_id maybe None
        if len(detections.class_id) > 0:
            ### Segment Anything Model 2###
            detections.mask = self.get_sam2_segmentation_from_xyxy(
                image=image, 
                xyxy=detections.xyxy
            )
            
        return detections

    # Prompting SAM with detected boxes
    def get_sam2_segmentation_from_xyxy(
            self, 
            image: np.ndarray, 
            xyxy: np.ndarray
    ) -> np.ndarray:
        self.sam2_predictor.set_image(image)
        result_masks = []
        for box in xyxy:
            masks, scores, logits = self.sam2_predictor.predict(
                box=box,
                multimask_output=True
            )
            index = np.argmax(scores)
            result_masks.append(masks[index].astype(bool))
        return np.array(result_masks)

    def process_tag_classes(self, text_prompt:str) -> list[str]:
        '''Convert a text prompt from Tag2Text to a list of classes. '''
        classes = text_prompt.split('.')
        classes = [obj_class.strip() for obj_class in classes]
        classes = [obj_class for obj_class in classes if obj_class != '']
        for c in self.add_classes:
            if c not in classes:
                classes.append(c)
        for c in self.remove_classes:
            classes = [obj_class for obj_class in classes if c not in obj_class.lower()]
        return classes
    
    def vis_result(
        self,
        image: np.ndarray, 
        detections: sv.Detections, 
        classes: list[str], 
        color: Union[Color, ColorPalette] = ColorPalette.default(),
        instance_random_color: bool = False,
        draw_bbox: bool = True,
    ) -> np.ndarray:
        '''
        Annotate the image with the detection results. 
        This is fast but of the same resolution of the input image, thus can be blurry. 
        '''
        # annotate image with detections
        box_annotator = sv.BoxAnnotator(
            color = color,
            text_scale=0.3,
            text_thickness=1,
            text_padding=2,
        )
        mask_annotator = sv.MaskAnnotator(
            color = color
        )

        if hasattr(detections, 'confidence') and hasattr(detections, 'class_id'):
            confidences = detections.confidence
            class_ids = detections.class_id
            if confidences is not None:
                labels = [
                    f"{classes[class_id]} {confidence:0.2f}"
                    for confidence, class_id in zip(confidences, class_ids)
                ]
            else:
                labels = [f"{classes[class_id]}" for class_id in class_ids]
        else:
            print("Detections object does not have 'confidence' or 'class_id' attributes or one of them is missing.")

        
        if instance_random_color:
            # generate random colors for each segmentation
            # First create a shallow copy of the input detections
            detections = dataclasses.replace(detections)
            detections.class_id = np.arange(len(detections))
            
        annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
        
        if draw_bbox:
            annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
        return annotated_image, labels
