from fastapi import FastAPI
from app.api.endpoints import router

# Initialize FastAPI app
app = FastAPI()

# Include routes
app.include_router(router)

# Health check endpoint
@app.get("/")
def read_root():
    return {"message": "Manufacturing Predictive API is running."}