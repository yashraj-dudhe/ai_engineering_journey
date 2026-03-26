from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message":"Hello I am your AI Server."}

@app.get("/health")
def health_check():
    return {"status":"Green","gpu":"No(cpu mode)"}

