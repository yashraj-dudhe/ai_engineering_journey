from fastapi import FastAPI

#create the application instance

app = FastAPI()

#define the path operations decorator
@app.get("/")
#define the path operation function
async def read_root():
    return {"message": "Hello yashraj", "track":"Iot,Python,AWS"}

@app.get("/sensor/{sensor_id}/logs/")
async def get_sensor_data(sensor_id: int,limit: int = 10, offset:int = 0):
    return {"sensor_id": sensor_id, "limit":limit}