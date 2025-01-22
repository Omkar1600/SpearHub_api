from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from app.core.file_utils import save_dataset, get_dataset_path, fetch_query_example
from app.core.model_utils import train_models, load_model, make_prediction
from fastapi import Form, File, UploadFile
from fastapi import APIRouter, Request
import uuid
import numpy as np
router = APIRouter()

# Upload endpoint
@router.post("/upload")
async def upload_data(
    user_id: str = Form(...), 
    task_type: str = Form(...), 
    output_column: str = Form(...), 
    file: UploadFile = File(...)
):
    """Endpoint to upload a dataset."""
    try:
        task_id = str(uuid.uuid4())
        query_example = save_dataset(user_id, task_id, task_type, output_column, file)
        return {"message": "Dataset uploaded successfully.", "task_id": task_id, "query_example": query_example}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process dataset: {str(e)}")

# Train endpoint
@router.post("/train")
def train_model(user_id: str = Form(...), task_id: str = Form(...)):
    """Endpoint to train a model on a dataset."""
    try:
        results, query_example = train_models(user_id, task_id)
        return {"results": results, "query_example": query_example}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to train model: {str(e)}")

# Predict endpoint
@router.post("/predict")
async def predict(request: Request):
    """Endpoint to make predictions using the trained model."""
    try:
        # Await the asynchronous call to get the JSON body
        response = await request.json()

        # Extract the necessary fields
        user_id = response['user_id']
        task_id = response['task_id']
        input_data = response['input_data']

        # Make predictions using the provided data
        prediction = make_prediction(user_id, task_id, input_data)
        return prediction
    except KeyError as e:
        # Handle missing keys in the JSON body
        raise HTTPException(status_code=400, detail=f"Missing field in request body: {str(e)}")
    except HTTPException as e:
        raise e
    except Exception as e:
        # Handle general exceptions
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
