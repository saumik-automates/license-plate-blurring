from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
import cv2
import numpy as np
import yaml
from pathlib import Path
from src.core.model import ModelHandler
from src.core.processing import process_frame

app = FastAPI(title="License Plate Blurring API", version="1.0.0")

# Load Config
config_path = Path("config/config.yaml")
if config_path.exists():
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
else:
    config = {
        "model": {"default_weights": "derived/yolov8n.pt"},
        "inference": {"conf_threshold": 0.3, "blur_intensity": 51},
    }

model_handler = ModelHandler(trained_path=config["model"].get("trained_weights"))


@app.on_event("startup")
async def startup_event():
    model_handler.load_model()


@app.get("/")
def root():
    return {"message": "License Plate Blurring API is running. Visit /docs for documentation."}


@app.post("/predict/blur")
async def predict_blur(
    file: UploadFile = File(...),
    conf_threshold: float = Form(config["inference"]["conf_threshold"]),
    blur_intensity: int = Form(config["inference"]["blur_intensity"]),
):
    """
    Process an uploaded image to blur license plates.
    Returns the blurred image directly.
    """
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    _, blur_img, _ = process_frame(
        model_handler, img, conf_threshold=conf_threshold, blur_intensity=blur_intensity
    )

    # Encode back to JPEG
    _, encoded_img = cv2.imencode(".jpg", blur_img)
    return Response(content=encoded_img.tobytes(), media_type="image/jpeg")
