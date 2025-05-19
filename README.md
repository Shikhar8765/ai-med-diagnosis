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

# 🧠 AI Medical Diagnosis System

This project uses a FastAPI backend containerized with Docker to classify chest X-ray images (Pneumonia vs. Normal). Built with ResNet-50 and deployed on Hugging Face Spaces.

# AI Medical Diagnosis API

This is a FastAPI-based AI service that classifies chest X-ray images into
- Normal
- Pneumonia

Upload `.png`, `.jpg`, or `.jpeg` images via the `predict` endpoint.
## 🔗 Deployed Demo

> 🌐 [Try the live API]([https://your-space.hf.space](https://huggingface.co/spaces/Shik12/ai-med-diagnosis))
