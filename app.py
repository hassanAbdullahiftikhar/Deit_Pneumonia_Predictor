import torch
import numpy as np
import cv2
import io
import base64
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from transformers import AutoImageProcessor, AutoModelForImageClassification
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import uvicorn
import os

# ── 1. Load model ─────────────────────────────────────────────────────────────
MODEL_PATH = "./deit-pneumonia-papersplit/deit-pneumonia-papersplit"
MOCK_MODE = False

app = FastAPI(title="PneumoScan AI API")

try:
    processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
    model     = AutoModelForImageClassification.from_pretrained(MODEL_PATH)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # ── 2. Wrapper so Grad-CAM receives plain logits tensor ───────────────────
    class LogitsWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.model = m

        def forward(self, x):
            return self.model(pixel_values=x).logits

    wrapped_model = LogitsWrapper(model)

    # ── 3. Auto-detect backbone for Grad-CAM target layer ────────────────────
    if hasattr(model, 'deit'):
        target_layers = [wrapped_model.model.deit.encoder.layer[-1].layernorm_before]
    elif hasattr(model, 'vit'):
        target_layers = [wrapped_model.model.vit.encoder.layer[-1].layernorm_before]
    else:
        raise AttributeError("Could not resolve backbone.")
except Exception as e:
    print(f"Warning: Could not load model from {MODEL_PATH}. Entering MOCK MODE for UI demo.")
    print(f"Error: {e}")
    MOCK_MODE = True
    model = None

# ── 4. Reshape transform for patch tokens ────────────────────────────────────
def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :]
    result = result.reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result

# ── 5. Preprocessing ──────────────────────────────────────────────────────────
SIZE = 224 # Default for DeiT

def preprocess(image: Image.Image):
    # This will only be called if not in MOCK_MODE
    MEAN = processor.image_mean
    STD  = processor.image_std
    
    eval_transform = Compose([
        Resize((SIZE, SIZE)),
        CenterCrop(SIZE),
        ToTensor(),
        Normalize(mean=MEAN, std=STD),
    ])
    
    img_rgb = image.convert("RGB")
    tensor  = eval_transform(img_rgb).unsqueeze(0).to(device)
    return tensor, img_rgb

# ── 6. Grad-CAM overlay ───────────────────────────────────────────────────────
def apply_gradcam(image: Image.Image, target_class: int):
    tensor, img_rgb = preprocess(image)

    with GradCAM(
        model=wrapped_model,
        target_layers=target_layers,
        reshape_transform=reshape_transform
    ) as cam:
        targets   = [ClassifierOutputTarget(target_class)]
        grayscale = cam(input_tensor=tensor, targets=targets)[0]

    img_np  = np.array(img_rgb.resize((SIZE, SIZE)))
    heatmap = cv2.resize(grayscale, (SIZE, SIZE))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_np, 0.55, heatmap, 0.45, 0)

    return Image.fromarray(overlay)

# ── 7. Helper functions for response ──────────────────────────────────────────
def image_to_base64(image: Image.Image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# ── 8. API Endpoints ──────────────────────────────────────────────────────────

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        if MOCK_MODE:
            # Mock logic based on filename or random
            filename = file.filename.lower()
            is_pneumonia = "pneumonia" in filename or (np.random.random() > 0.5 and "normal" not in filename)
            
            confidence = 0.85 + np.random.random() * 0.14
            other_prob = 1.0 - confidence
            
            probs = {
                "NORMAL": other_prob if is_pneumonia else confidence,
                "PNEUMONIA": confidence if is_pneumonia else other_prob
            }
            
            # Simple "heatmap" mock: just some red circles
            mock_img = np.array(image.convert("RGB").resize((SIZE, SIZE)))
            overlay = mock_img.copy()
            if is_pneumonia:
                cv2.circle(overlay, (SIZE//2, SIZE//2), 40, (255, 0, 0), -1)
                cv2.circle(overlay, (SIZE//3, SIZE//2), 30, (200, 0, 0), -1)
            cv2.addWeighted(overlay, 0.4, mock_img, 0.6, 0, mock_img)
            
            return {
                "prediction": "PNEUMONIA" if is_pneumonia else "NORMAL",
                "confidence": confidence,
                "probabilities": probs,
                "gradcam": image_to_base64(Image.fromarray(mock_img))
            }

        # Core prediction
        tensor, _ = preprocess(image)
        with torch.no_grad():
            outputs = model(pixel_values=tensor)
            probs   = torch.softmax(outputs.logits, dim=1)[0].cpu().numpy()

        pred_idx   = int(np.argmax(probs))
        pred_label = model.config.id2label[pred_idx]
        confidence = float(probs[pred_idx])

        label_probs = {
            model.config.id2label[i]: float(probs[i])
            for i in range(len(probs))
        }

        # Grad-CAM
        gradcam_img = apply_gradcam(image, pred_idx)
        
        return {
            "prediction": pred_label,
            "confidence": confidence,
            "probabilities": label_probs,
            "gradcam": image_to_base64(gradcam_img)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── 9. Static File Serving ───────────────────────────────────────────────────

# Ensure frontend directory exists
if not os.path.exists("frontend"):
    os.makedirs("frontend")

app.mount("/samples", StaticFiles(directory="sample_xrays"), name="samples")
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)