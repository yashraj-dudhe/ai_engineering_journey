# AI Engineering Journey

A comprehensive collection of practical examples and implementations covering the fundamentals of AI/ML engineering, from basic neural networks to production-ready API deployments.

## 📚 Repository Overview

This repository contains 35+ Python scripts demonstrating various AI/ML concepts in a progressive learning path. Each file is numbered sequentially to guide learners through increasingly complex topics.

## 🗂️ Learning Path Structure

### Fundamentals (01-09)
- **01_basics.py** - Python and AI basics
- **02_numpy.py** - NumPy for numerical computing
- **03_sqlcheck.py** - Data handling with SQL
- **04_neuron.py** - Single neuron implementation
- **05_training.py** - Basic training loops
- **06_pytorch.py** - Introduction to PyTorch
- **07_solvexor.py** - XOR problem solving
- **08_tensor.py** - Tensor operations
- **09_broadcasting.py** - Broadcasting concepts

### Deep Learning (10-18)
- **10_convolutional.py** - Convolutional operations
- **11_cnn.py** - Convolutional Neural Networks
- **12_mnsitcnn.py** - MNIST with CNN
- **13_datapipeline.py** - Data pipeline creation
- **15_inference.py** - Model inference
- **18_confusionmatrix.py** - Model evaluation

### NLP & Transformers (19-28)
- **19_tokenizer.py** - Text tokenization
- **20_embeddings.py** - Word embeddings
- **21_sentimentclassifier.py** - Sentiment analysis model
- **22_rnn.py** - Recurrent Neural Networks
- **23_lstmclassifier.py** - LSTM for classification
- **24_attention.py** - Attention mechanisms
- **25_transformers.py** - Transformer architecture
- **26_gptlite.py** - Lightweight GPT implementation
- **27_hellohuggingface.py** - Hugging Face basics
- **28_finetunebert.py** - BERT fine-tuning

### 🚀 Production Deployment (29-30)

#### **29_fastapi.py** - FastAPI Introduction
A basic FastAPI server demonstrating:
- API endpoint creation
- Health check endpoints
- Server setup basics

#### **30_sentimentapi.py** - Production ML API
A production-ready sentiment analysis API showcasing:
- Real-world ML model serving
- Integration with Hugging Face transformers
- Request/response handling with Pydantic
- Error handling and API best practices
- Dockerized deployment

**What FastAPI is doing here:**
FastAPI serves as the bridge between machine learning models and real-world applications. After learning to build ML models (files 01-28), files 29-30 demonstrate how to:
1. Deploy models as REST APIs
2. Make ML models accessible via HTTP endpoints
3. Create production-ready ML services
4. Enable integration with web/mobile applications

### Computer Vision (31-33)
- **31_yolo.py** - YOLO object detection
- **32_yolowebcam.py** - Real-time object detection
- **33_potholes.py** - Practical CV application

## 🐳 Docker Deployment

The repository includes a Dockerfile for containerizing the sentiment analysis API:

```bash
# Build the Docker image
docker build -t sentiment-api .

# Run the container
docker run -p 7860:7860 sentiment-api
```

## 🛠️ Installation

```bash
pip install -r requirements.txt
```

## 📦 Key Dependencies

- **PyTorch** - Deep learning framework
- **Transformers** - Hugging Face models
- **FastAPI** - Modern API framework
- **Uvicorn** - ASGI server
- **Pydantic** - Data validation

## 🎯 Purpose of FastAPI in This Repository

FastAPI is introduced in the later stages (files 29-30) to teach:

1. **API Development** - How to create REST APIs for ML models
2. **Model Serving** - Deploying trained models in production
3. **Integration** - Connecting ML models with applications
4. **Scalability** - Building production-ready services

The progression from training models to serving them via APIs represents a complete AI engineering workflow, making this repository valuable for understanding both ML theory and practical deployment.

## 🚦 Running the APIs

### Basic FastAPI Server (29_fastapi.py)
```bash
uvicorn 29_fastapi:app --reload
# Access at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### Sentiment Analysis API (30_sentimentapi.py)
```bash
python 30_sentimentapi.py
# Access at http://localhost:8000
# Test at http://localhost:8000/docs
```

Example request:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this movie!"}'
```

## 📝 Learning Outcomes

By following this repository, you will learn:
- ✅ Neural network fundamentals
- ✅ Deep learning with PyTorch
- ✅ NLP and transformer architectures
- ✅ Computer vision applications
- ✅ **Production ML deployment with FastAPI**
- ✅ Containerization with Docker

## 🤝 Contributing

This is a learning repository. Feel free to explore, learn, and experiment!

## 📄 License

Educational purposes - feel free to use and learn from these examples.
