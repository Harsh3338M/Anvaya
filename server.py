import base64
import io
from PIL import Image
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoImageProcessor, AutoModelForImageClassification

app = FastAPI(title="Anvaya ISL Model API")

# Allow your index.html file to call this API without CORS errors
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_ID = "Hemg/Indian-sign-language-classification"
print(f"Loading {MODEL_ID} from Hugging Face...")

# Load processor and ViT model once at startup
processor = AutoImageProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
model.eval()
print("Model loaded successfully! Ready for real-time inference.")

class ImagePayload(BaseModel):
    image_base64: str  # Data URL or raw base64 from index.html canvas

@app.post("/predict")
async def predict_isl(payload: ImagePayload):
    try:
        # 1. Strip the "data:image/jpeg;base64," header if present
        b64_data = payload.image_base64.split(",")[-1]
        image_bytes = base64.b64decode(b64_data)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # 2. Preprocess image to 224x224 for ViT and run inference
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
            top_prob, top_idx = torch.max(probs, dim=-1)

        predicted_label = model.config.id2label[top_idx.item()]
        confidence = round(top_prob.item() * 100, 1)

        return {
            "label": predicted_label,
            "confidence": confidence
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)