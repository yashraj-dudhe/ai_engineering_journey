"""
FastAPI Basics - Introduction to FastAPI Framework

This file demonstrates the fundamentals of FastAPI, a modern Python web framework
for building APIs. FastAPI is used in ML/AI applications to serve models as REST APIs.

Why FastAPI in AI Engineering?
------------------------------
- Fast: High performance, on par with NodeJS and Go
- Easy: Simple to use and learn
- Automatic API documentation (Swagger UI)
- Type hints and validation
- Perfect for serving ML models in production

To run this server:
    uvicorn 29_fastapi:app --reload

Then visit:
    - http://localhost:8000 - Home endpoint
    - http://localhost:8000/health - Health check endpoint
    - http://localhost:8000/docs - Interactive API documentation (Swagger UI)
"""

from fastapi import FastAPI

# Create FastAPI application instance
app = FastAPI()

@app.get("/")
def home():
    """Home endpoint - Returns a welcome message"""
    return {"message":"Hello I am your AI Server."}

@app.get("/health")
def health_check():
    """Health check endpoint - Returns server status"""
    return {"status":"Green","gpu":"No(cpu mode)"}

