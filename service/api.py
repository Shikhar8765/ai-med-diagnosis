# 3_service/api.py

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import tempfile
from service.inference import InferenceEngine
from PIL import Image

app = FastAPI(title="AI Chest X-ray Diagnosis API")

engine = InferenceEngine("2_models/weights/resnet50_med.pth",
                         ["Normal", "Pneumonia"])

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    # Save uploaded image temporarily
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await file.read())
        temp_path = tmp.name

    # Run prediction
    result = engine.predict(temp_path)

    response = {
        "filename": file.filename,
        "prediction": result["label"],
        "confidence": round(result["confidence"], 4),
        "probabilities": result["probabilities"]
    }

    return JSONResponse(content=response)
