from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import boto3
import joblib
import os

app = FastAPI()

GCS_BUCKET = os.environ.get("CLOUD_BUCKET")
GCS_MODEL_KEY = "models/latest/model.pkl"
MODEL_PATH = os.path.expanduser("~/models/model.pkl")

def download_model():
    """Tải file model.pkl từ S3 về máy khi server khởi động."""
    if not os.path.exists(os.path.dirname(MODEL_PATH)):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    if GCS_BUCKET:
        print(f"Downloading model from s3://{GCS_BUCKET}/{GCS_MODEL_KEY}")
        s3 = boto3.client('s3')
        s3.download_file(GCS_BUCKET, GCS_MODEL_KEY, MODEL_PATH)
        print("Model downloaded successfully.")
    else:
        print("CLOUD_BUCKET not set. Skipping download.")

download_model()
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None

class PredictRequest(BaseModel):
    features: list[float]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    if len(req.features) != 12:
        raise HTTPException(status_code=400, detail="Expected 12 features (wine quality)")

    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded")

    preds = model.predict([req.features])
    
    label_map = {0: "thap", 1: "trung_binh", 2: "cao"}
    pred_val = int(preds[0])
    return {"prediction": pred_val, "label": label_map.get(pred_val, "unknown")}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
