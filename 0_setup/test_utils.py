# 0_setup/test_utils.py
from utils import load_image, preprocess

# Update the path to your test image
img_path = "../1_data/sample_image.png"   # or sample.dcm

# Load the image
img, meta = load_image(img_path)

print("Image shape:", img.shape)
print("Metadata:", meta)

# Preprocess it
tensor = preprocess(img)
print("Processed shape (CHW):", tensor.shape)
