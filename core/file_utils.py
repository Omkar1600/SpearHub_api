import os
import pandas as pd
from io import BytesIO
from fastapi import HTTPException

# Directory for storing datasets and queries
DATA_DIR = "app/data"
QUERY_FILE = "app/queries.csv"
os.makedirs(DATA_DIR, exist_ok=True)

# Initialize query file if it does not exist
if not os.path.exists(QUERY_FILE):
    pd.DataFrame(columns=["user_id", "task_id", "query"]).to_csv(QUERY_FILE, index=False)


def save_dataset(user_id, task_id, task_type, output_column, file):
    """
    Save uploaded dataset to the filesystem and store query structure.
    """
    if not (file.filename.endswith(".csv") or file.filename.endswith(".xlsx")):
        raise HTTPException(status_code=400, detail="Only CSV or Excel files are allowed.")

    content = file.file.read()
    if file.filename.endswith(".csv"):
        data = pd.read_csv(BytesIO(content))
    else:
        data = pd.read_excel(BytesIO(content))

    if output_column not in data.columns:
        raise HTTPException(status_code=400, detail=f"Output column '{output_column}' not found in dataset.")

    filename = f"{user_id}_{task_id}_{task_type}.csv"
    filepath = os.path.join(DATA_DIR, filename)
    data.to_csv(filepath, index=False)

    input_columns = [col for col in data.columns if col != output_column]
    query_example = {col: "example_value" for col in input_columns}

    # Save query example
    query_data = pd.read_csv(QUERY_FILE)
    query_data = pd.concat([
        query_data,
        pd.DataFrame([{"user_id": user_id, "task_id": task_id, "query": query_example,"output_column":output_column}])
    ])
    query_data.to_csv(QUERY_FILE, index=False)

    return query_example


def get_dataset_path(user_id, task_id):
    """
    Retrieve the path of the dataset for a given user and task ID.
    """
    dataset_files = [f for f in os.listdir(DATA_DIR) if f.startswith(f"{user_id}_{task_id}")]
    if not dataset_files:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return os.path.join(DATA_DIR, dataset_files[0])


def fetch_query_example(user_id, task_id):
    """
    Retrieve the stored query example for a given user and task ID.
    """
    query_data = pd.read_csv(QUERY_FILE)
    query_row = query_data[(query_data["user_id"] == user_id) & (query_data["task_id"] == task_id)]

    if query_row.empty:
        raise HTTPException(status_code=404, detail="Task not found for the user.")
    
    return eval(query_row["query"])
