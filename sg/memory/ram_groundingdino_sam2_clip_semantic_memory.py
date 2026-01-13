import numpy as np
from PIL import Image
import supervision as sv
import sys
from torchvision import transforms
# ram model
from ram.models import ram
from ram import inference_ram
from sg.perception.models.mygroundingdinosam2 import MyGroundingDINOSAM2
from sg.perception.models.myclip import MyClip
from sg.utils.utils import ram_checkpoint_path, bert_base_uncased_path
import warnings
warnings.filterwarnings('ignore')
from typing import Set, List


class RamGroundingDinoSAM2ClipDataset():
    def __init__(
        self,
        classes: List=[],
        box_threshold: float=0.25,
        text_threshold: float=0.25,
        nms_threshold: float=0.5,
        device: str="cuda",
        accumu_classes: bool=False
    ):
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.nms_threshold = nms_threshold
        self.accumu_classes = accumu_classes

        # ### Initialize the GroundingDINO SAM2 model ###
        self.mygroundingdino_sam2 = MyGroundingDINOSAM2(
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            device=self.device
        )

        self.myclip = MyClip(device=device)
        
        ### Initialize the RAM (tagging) model ###
        tagging_model = ram(
            pretrained=ram_checkpoint_path, 
            image_size=384, vit="swin_l", 
            text_encoder_type=bert_base_uncased_path
        )
        self.tagging_model = tagging_model.eval().to(self.device)
        
        # initialize Tag2Text
        self.tagging_transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
        ])

        # Add "other item" to capture objects not in the tag2text captions. 
        # Remove "xxx room", otherwise it will simply include the entire image
        # Also hide "wall" and "floor" for now...
        self.global_classes = set(classes)

        self.add_classes = ["handle", 'Bottled Coke', 'Canned Beer', 'apple', 'potato', 'green toy', 'blue bottle', 'green container', 'blue and grey umbrella', 'blue toy', 'pen', 'orange', 'eggplant', 'yellow bottle', 'corn', 'chili pepper', 'small scissors', 'keys', 'green container', 'cabinet', 'long table', 'big table', 'plate']

        self.remove_classes = [
            "room", "kitchen", "office", "house", "home", "building", "corner",
            "shadow", "carpet", "photo", "shade", "stall", "space", "aquarium", 
            "apartment", "image", "city", "blue", "skylight", "hallway", 
            "bureau", "modern", "salon", "doorway", "wall lamp", "wood floor",
            "floor", "ladder", "sink", "counter top", "door", "screen door", "hardwood",
            "shower curtain", "curtain", "slide", "peak", "closet",
            "man", "woman", "child", "boy", "girl", "person", "human", "drawer",
            "food", "vegetable", "fruit"
        ]
        # self.bg_classes = ["wall", "floor", "ceiling"]
        # self.add_classes += self.bg_classes


    def get_classes_colors(self, classes):
        class_colors = {}
        # Generate a random color for each class
        for class_idx, class_name in enumerate(classes):
            # Generate random RGB values between 0 and 255
            r = np.random.randint(0, 256)/255.0
            g = np.random.randint(0, 256)/255.0
            b = np.random.randint(0, 256)/255.0
            # Assign the RGB values as a tuple to the class in the dictionary
            class_colors[f"{class_name}"] = (r, g, b)
        class_colors[-1] = (0, 0, 0)
        return class_colors

    def semantic_process(self, image: np.ndarray):
        image_pil = Image.fromarray(image)
        raw_image = image_pil.resize((384, 384))
        raw_image = self.tagging_transform(raw_image).unsqueeze(0).to(self.device)
        text_prompt = inference_ram(raw_image , self.tagging_model)[0].replace(' | ', '.')
        classes = self.process_tag_classes(text_prompt=text_prompt)
        self.global_classes.update(classes)
        
        if self.accumu_classes:
            # Use all the classes that have been seen so far
            classes = list(self.global_classes)

        # ### Segment Anything Model 2###
        detections = self.mygroundingdino_sam2.run(
            image=image,
            classes=classes
        )

        # detections.class_id maybe None
        if len(detections.class_id) > 0:
            ### Compute and save the clip features of detections ###
            image_feats, text_feats = self.compute_clip_features(
                # image_pil, 
                image,
                detections, 
                classes, 
                padding=20,
            )
        else:
            image_feats, text_feats = [], []

        ### Visualize results ###
        annotated_image, labels = self.mygroundingdino_sam2.vis_result(image, detections, classes)

        # Convert the detections to a dict. The elements are in np.array
        det_res = {
            "xyxy": detections.xyxy,
            "confidence": detections.confidence,
            "class_id": detections.class_id,
            "mask": detections.mask,
            "classes": classes,
            "image_feats": image_feats,
            "text_feats": text_feats
        }

        return det_res, annotated_image, image_pil

    def get_classes_and_colors(self):
        class_colors = self.get_classes_colors(self.global_classes)
        return {
            "classes": list(self.global_classes),
            "class_colors": class_colors
        }

    def compute_clip_features(
        self,
        # image: Image,
        image:  np.ndarray,
        detections: sv.Detections,
        classes: list,
        padding: int=20,
        masked_weight: float=0.75
    ):
        image_feats = []
        text_feats = []

        for idx in range(len(detections.xyxy)):
            # Get the crop of the mask with padding
            x_min, y_min, x_max, y_max = detections.xyxy[idx]

            # Check and adjust padding to avoid going beyond the image borders
            # image_width, image_height = image.size
            image_height, image_width = image.shape[:2]
            left_padding = min(padding, x_min)
            top_padding = min(padding, y_min)
            right_padding = min(padding, image_width - x_max)
            bottom_padding = min(padding, image_height - y_max)

            # Apply the adjusted padding
            x_min -= left_padding
            y_min -= top_padding
            x_max += right_padding
            y_max += bottom_padding

            x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)

            mask = detections.mask[idx]

            cropped_image_np = image[y_min:y_max, x_min:x_max]
            cropped_mask = np.expand_dims(mask[y_min:y_max, x_min:x_max], axis=-1).astype(np.uint8) 

            cropped_image = Image.fromarray(cropped_image_np.astype(np.uint8))
            cropped_mask_image = Image.fromarray((cropped_image_np * cropped_mask).astype(np.uint8))

            crop_image_feat = self.myclip.get_image_feature(cropped_image)
            crop_mask_image_feat = self.myclip.get_image_feature(cropped_mask_image)

            crop_feat = masked_weight * crop_mask_image_feat + (1 - masked_weight) * crop_image_feat
            
            class_id = detections.class_id[idx]
            text_feat = self.myclip.get_text_feature([classes[class_id]])
            
            crop_feat = crop_feat.cpu().numpy()
            text_feat = text_feat.cpu().numpy()

            image_feats.append(crop_feat)
            text_feats.append(text_feat)
            
        # turn the list of feats into np matrices
        image_feats = np.concatenate(image_feats, axis=0)
        text_feats = np.concatenate(text_feats, axis=0)

        return image_feats, text_feats
    

    def process_tag_classes(self, text_prompt:str) -> list[str]:
        '''Convert a text prompt from Tag2Text to a list of classes. '''
        classes = text_prompt.split('.')
        classes = [obj_class.strip() for obj_class in classes]
        classes = [obj_class for obj_class in classes if obj_class != '']
        for c in self.remove_classes:
            classes = [obj_class for obj_class in classes if c not in obj_class.lower()]
        for c in self.add_classes:
            if c not in classes:
                classes.append(c)
        return classes
        

        

