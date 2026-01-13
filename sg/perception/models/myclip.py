# import open_clip
# from PIL import Image
# from dovisg.utils.utils import clip_checkpoint_path

# class MyClip:
#     def __init__(self, device="cuda"):
#         self.device = device
#         print("==> Initializing CLIP model...")
#         clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
#             model_name="ViT-H-14", pretrained=clip_checkpoint_path
#         )
#         self.clip_model = clip_model.to(self.device)
#         self.clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")
#         print("==> Done initializing CLIP model.")

#     def get_text_feature(self, text_queries: list):
#         tokenized_text = self.clip_tokenizer(text_queries).to("cuda")
#         text_feat = self.clip_model.encode_text(tokenized_text)
#         text_feat /= text_feat.norm(dim=-1, keepdim=True)

#     def get_image_feature(self, image: Image):
#         preprocessed_image = self.clip_preprocess(image).unsqueeze(0).to("cuda")
#         image_feat = self.clip_model.encode_image(preprocessed_image)
#         image_feat /= image_feat.norm(dim=-1, keepdim=True)


""" my clip just be init once each runing time """
from PIL import Image
import open_clip
from sg.utils.utils import clip_checkpoint_path

class MyClip:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(MyClip, cls).__new__(cls)
        return cls._instance

    def __init__(self, device="cuda"):
        if not hasattr(self, 'initialized'):
            self.device = device
            print("==> Initializing CLIP model...")
            clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
                model_name="ViT-H-14", pretrained=clip_checkpoint_path
            )
            self.clip_model = clip_model.to(self.device)
            self.clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")
            print("==> Done initializing CLIP model.")
            self.initialized = True

    def get_text_feature(self, text_queries: list):
        tokenized_text = self.clip_tokenizer(text_queries).to(self.device)
        text_feat = self.clip_model.encode_text(tokenized_text)
        text_feat /= text_feat.norm(dim=-1, keepdim=True)
        return text_feat

    def get_image_feature(self, image: Image):
        preprocessed_image = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        image_feat = self.clip_model.encode_image(preprocessed_image)
        image_feat /= image_feat.norm(dim=-1, keepdim=True)
        return image_feat
