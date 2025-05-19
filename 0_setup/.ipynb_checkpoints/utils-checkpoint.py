# 0_setup/utils.py

import os, uuid, cv2, pydicom
import numpy as np
from skimage import exposure
from pydicom.uid import generate_uid

# ------------------------------------------------------------------
# Load DICOM, PNG, JPG into RGB image + optional metadata
# ------------------------------------------------------------------
def load_image(path: str):
    """
    Accepts .dcm, .png, .jpg — returns RGB image (H x W x 3) and metadata dict
    """
    ext = os.path.splitext(path)[-1].lower()
    meta = {}

    if ext == ".dcm":
        dcm = pydicom.dcmread(path)
        arr = dcm.pixel_array.astype(np.float32)
        arr = (arr - arr.min()) / (arr.ptp() + 1e-9) * 255
        img = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        meta = {k: getattr(dcm, k, "N/A") for k in ("PatientID", "Modality", "StudyDate")}
    else:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    return img, meta

# ------------------------------------------------------------------
# Preprocessing: histogram equalization, blur, z-score normalization
# ------------------------------------------------------------------
def preprocess(img):
    """
    img: uint8 RGB (HxWx3)
    ➜ returns CHW float32 tensor normalized (3xHxW)
    """
    eq   = exposure.equalize_adapthist(img, clip_limit=0.03)
    eq   = (eq * 255).astype(np.uint8)
    blur = cv2.GaussianBlur(eq, (5,5), 0)

    norm = blur.astype(np.float32) / 255.
    norm = (norm - norm.mean()) / (norm.std() + 1e-8)
    chw  = np.transpose(norm, (2, 0, 1))
    return chw

# ------------------------------------------------------------------
# DICOM Anonymization (overwrite personally identifiable info)
# ------------------------------------------------------------------
def anonymise_dicom(src_path, dst_path):
    dcm = pydicom.dcmread(src_path)
    dcm.PatientName = "ANON"
    dcm.PatientID   = str(uuid.uuid4())
    dcm.StudyInstanceUID = generate_uid()
    dcm.save_as(dst_path)
