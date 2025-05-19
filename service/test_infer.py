# 3_service/test_infer.py

from inference import InferenceEngine

engine = InferenceEngine("../2_models/weights/resnet50_med.pth",
                         ["Normal", "Pneumonia"])

result = engine.predict("../1_data/images/1_Viral Pneumonia-47.png")  # Use your own image here
print(result)
