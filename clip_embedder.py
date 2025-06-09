from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import requests

class CLIPEmbedder:
    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def embed(self, text: str) -> list:
        inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)
        return outputs[0].numpy().tolist()

    def embed_image_from_url(self, url: str) -> list | None:
        try:
            image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
        except Exception as e:
            print(f"[ERROR] Failed to load image: {url} — {e}")
            return None

        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
        return outputs[0].numpy().tolist()
