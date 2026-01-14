from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(title="Sentiment Analyzer API")
print("Loading model....")
sentiment_pipeline = pipeline("sentiment-analysis",model='distilbert-base-uncased-finetuned-sst-2-english')
print("Model Loaded")


class SentimentRequest(BaseModel):
    text: str
    
@app.post("/predict")
def predict_sentiment(request: SentimentRequest):
    try:
        input_text = request.text
        
        result = sentiment_pipeline(input_text)
        label = result[0]['label']
        score = result[0]['score']
        
        return {
            "input_text": input_text,
            "sentiment": label,
            "confidence": f"{score:.2f}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500,detail = str(e))
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app,host="0.0.0.0",port = 8000)