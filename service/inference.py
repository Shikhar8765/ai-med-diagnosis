# 3_service/inference.py

import torch
from torchvision import models, transforms
from PIL import Image
import os

class InferenceEngine:
    def __init__(self, weight_path, class_names):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.class_names = class_names

        # Load model
        self.model = models.resnet50()
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, len(class_names))
        self.model.load_state_dict(torch.load(weight_path, map_location=self.device))
        self.model.eval().to(self.device)

        # Define preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    @torch.inference_mode()
    def predict(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        logits = self.model(image_tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        predicted_idx = probs.argmax()
        confidence = probs[predicted_idx]
        label = self.class_names[predicted_idx]

        return {
            "label": label,
            "confidence": float(confidence),
            "probabilities": {cls: float(p) for cls, p in zip(self.class_names, probs)}
        }
