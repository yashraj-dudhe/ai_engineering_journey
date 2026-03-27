"""
Production-Ready Sentiment Analysis API using FastAPI

This file demonstrates how to deploy a machine learning model (sentiment analysis)
as a production-ready REST API using FastAPI.

What This File Does:
-------------------
1. Loads a pre-trained sentiment analysis model from Hugging Face
2. Creates a FastAPI application to serve the model
3. Provides a POST endpoint for sentiment predictions
4. Handles errors gracefully
5. Uses Pydantic for request/response validation

This is the PRODUCTION use case of AI/ML - taking trained models and making them
accessible to applications via HTTP APIs.

Architecture:
------------
Client (Web/Mobile App) → FastAPI Server → ML Model → Response

To run this API:
    python 30_sentimentapi.py

Then test it:
    - Visit http://localhost:8000/docs for interactive API documentation
    - Send POST requests to http://localhost:8000/predict

Example API call:
    curl -X POST "http://localhost:8000/predict" \
         -H "Content-Type: application/json" \
         -d '{"text": "I love this movie!"}'

Docker Deployment:
    docker build -t sentiment-api .
    docker run -p 7860:7860 sentiment-api
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

# Create FastAPI app with a descriptive title
app = FastAPI(title="Sentiment Analyzer API")

# Load the pre-trained sentiment analysis model
# This happens once at startup, not for every request (efficient!)
print("Loading model....")
sentiment_pipeline = pipeline("sentiment-analysis",model='distilbert-base-uncased-finetuned-sst-2-english')
print("Model Loaded")


# Define the request schema using Pydantic
# This provides automatic validation and documentation
class SentimentRequest(BaseModel):
    text: str
    
@app.post("/predict")
def predict_sentiment(request: SentimentRequest):
    """
    Predict sentiment of input text
    
    Args:
        request: SentimentRequest containing the text to analyze
        
    Returns:
        JSON with sentiment prediction and confidence score
        
    Example:
        Input: {"text": "This movie is amazing!"}
        Output: {"input_text": "This movie is amazing!", 
                 "sentiment": "POSITIVE", 
                 "confidence": "0.99"}
    """
    try:
        input_text = request.text
        
        # Run the ML model prediction
        result = sentiment_pipeline(input_text)
        label = result[0]['label']
        score = result[0]['score']
        
        # Return structured response
        return {
            "input_text": input_text,
            "sentiment": label,
            "confidence": f"{score:.2f}"
        }
        
    except Exception as e:
        # Handle errors gracefully with appropriate HTTP status code
        raise HTTPException(status_code=500,detail = str(e))
    

if __name__ == "__main__":
    import uvicorn
    # Start the server on all network interfaces, port 8000
    uvicorn.run(app,host="0.0.0.0",port = 8000)