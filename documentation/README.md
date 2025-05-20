---
title: AI Medical Diagnosis
emoji: 🩺
colorFrom: indigo
colorTo: blue
sdk: docker
sdk_version: "1.0"
app_file: Dockerfile
pinned: true
---

# 🩺 AI Medical Diagnosis System

This project implements an AI-powered diagnostic service that classifies chest X-ray images into:

- **Normal**
- **Pneumonia**

It uses a **ResNet-50** convolutional neural network trained on the COVID-19 Radiography dataset and is deployed via **FastAPI** inside a **Docker container** on **Hugging Face Spaces**.

---

## 🚀 Live Demo

- 🌐 **App**: [https://Shik12-ai-med-diagnosis.hf.space](https://Shik12-ai-med-diagnosis.hf.space)  
- 📚 **Swagger Docs**: [https://Shik12-ai-med-diagnosis.hf.space/docs](https://Shik12-ai-med-diagnosis.hf.space/docs)

---

## 🧠 Model Architecture

- Base model: `torchvision.models.resnet50(pretrained=True)`
- Final layer replaced with: `nn.Linear(..., 2)` for binary classification
- Loss function: CrossEntropyLoss
- Optimizer: Adam
- Input image size: 224x224 (RGB)

---

## 🗂️ Project Structure
ai-med-diagnosis/
├── 0_setup/ # Setup scripts and requirements
├── 2_models/ # Training code and model weights
├── service/ # FastAPI REST API
├── documentation/ # Submission documentation
├── Dockerfile # Hugging Face Docker config
├── .huggingface.yml # Hugging Face Space metadata

---

## 📦 How to Run Locally

```bash
# Clone the repository
git clone https://github.com/Shikhar8765/ai-med-diagnosis
cd ai-med-diagnosis

# Set up a virtual environment
python -m venv .venv
.\.venv\Scripts\activate   # For Windows

# Install dependencies
pip install -r 0_setup/requirements.txt

# Run the API locally
uvicorn service.api:app --reload --port 8000
curl -X POST "https://Shik12-ai-med-diagnosis.hf.space/predict" ^
     -H "accept: application/json" ^
     -H "Content-Type: multipart/form-data" ^
     -F "file=@sample_xray.jpg"
{
  "label": "Pneumonia",
  "confidence": 0.9812,
  "probabilities": {
    "Normal": 0.0188,
    "Pneumonia": 0.9812
  }
}
🧾 Dataset Source
COVID-19 Radiography Database
https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database
👨‍💻 Maintainer
Name: Shikhar Srivastava

GitHub: github.com/Shikhar8765

Hugging Face: Shik12

Deployed With: FastAPI, Docker, Hugging Face Spaces