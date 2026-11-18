from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def index() -> None:
    return {"Hey"}

@app.get("/about")
def about_page() -> None:
    return {"About page"}

@app.get("/blogs")
def show_blogs():
    return {f"showing only {limit} blogs"}

@app.get("/string")
def ret_str() -> str:
    return "Hi this is yashraj and this route returns string."