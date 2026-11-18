from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class blog(BaseModel):
    tile:str
    author:str
    
    
@app.post('/blog')
def create(request: blog):
    return request