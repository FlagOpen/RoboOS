import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from sg.memory.view_dataset import ViewDataset
from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import load_image, rbd, numpy_image_to_torch
from pathlib import Path
import torch
from typing import List, Union

class RGBFeatureMatch:
    def __init__(self, max_num_keypoints=1024):
        self.max_num_keypoints = max_num_keypoints
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'
        self.extractor = SuperPoint(max_num_keypoints=self.max_num_keypoints).eval().to(self.device)
        self.matcher = LightGlue(features="superpoint").eval().to(self.device)

    def extract_feature(self, image):
        with torch.no_grad():
            feature = self.extractor.extract(image.to(self.device))
        return feature

    def match_features(self, see_frame_feature, feature_history):
        with torch.no_grad():
            matches = self.matcher({"image0": see_frame_feature, "image1": feature_history})
        see_frame_feature_res, feature_history_res, matches_res = [
            rbd(x) for x in [see_frame_feature, feature_history, matches]
        ]
        # print(matches_res["scores"].sum())
        return see_frame_feature_res, feature_history_res, matches_res

    def extract_memory_features(self, images: List[np.ndarray], features: Union[dict, None]=None):
        images = [numpy_image_to_torch(image) for image in images]
        if features == None:
            features = {}
            start_index = 0
        else:
            start_index = len(features)
        for index in tqdm(range(len(images)), total=len(images), 
                          desc="extract memory lightglue features"):
            features[start_index + index] = self.extract_feature(image=images[index])
        return features
            
    def find_most_similar_image(self, image: np.ndarray, features: dict, visualize=False, view_dataset: ViewDataset=None):
        image_torch = numpy_image_to_torch(image)
        see_frame_feature = self.extract_feature(image=image_torch)
        best_score = None
        best_matches_len = 0
        best_index = -1
        # matches_lens = []
        for index in tqdm(range(len(features)), "Feature extraction and matching: "):
            feature_history = features[index]
            _, _, matches = self.match_features(see_frame_feature, feature_history)
            score = matches["scores"].sum().detach().cpu().numpy().item()
            # matches_lens.append(len(matches["matches"]))
            if best_score is None or score > best_score:
                best_index = index
                best_score = score
                best_matches_len = len(matches["matches"])
        # assert max(matches_lens) == best_matches_len

        # print(best_score, best_matches_len)
        if visualize:
            assert view_dataset is not None, "view dataset must been set when you want to visualize"
            best_image = Image.fromarray(view_dataset.images[best_index])
            see_image = Image.fromarray(image)
            height, width, _ = image.shape
            new_width = width * 2
            new_height = height
            new_img = Image.new('RGB', (new_width, new_height))
            new_img.paste(see_image, (0, 0))
            new_img.paste(best_image, (width, 0))
            new_img.show()

        return best_index, best_matches_len
