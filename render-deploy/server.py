from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
from PIL import Image
import base64
import io

app = FastAPI()

# Load model + labels
model = tf.keras.models.load_model("artifact_model")
labels = [line.strip() for line in open("artifact_labels.txt")]

class ImageData(BaseModel):
    image: str  # base64 string

def preprocess(base64_string):
    img_bytes = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

@app.post("/identify")
def identify(data: ImageData):
    x = preprocess(data.image)
    preds = model.predict(x)[0]
    idx = np.argmax(preds)
    return {
        "label": labels[idx],
        "confidence": float(preds[idx])
    }
