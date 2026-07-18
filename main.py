from fastapi import FastAPI

app = FastAPI(title="Audio Transcript")

@app.get("/")
async def root():
    return {"message": "Welcome to the Audio Transcript API!"}
