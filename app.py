import gradio as gr
import torch
import numpy as np
import cv2
from PIL import Image

from transformers import AutoImageProcessor, AutoModelForImageClassification
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

# ── 1. Load model ─────────────────────────────────────────────────────────────
MODEL_PATH = "./deit-pneumonia-papersplit"

processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
model     = AutoModelForImageClassification.from_pretrained(MODEL_PATH)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ── 2. Wrapper so Grad-CAM receives plain logits tensor ───────────────────────
class LogitsWrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.model = m

    def forward(self, x):
        return self.model(pixel_values=x).logits

wrapped_model = LogitsWrapper(model)

# ── 3. Auto-detect backbone for Grad-CAM target layer ────────────────────────
print(f"Model type:        {type(model).__name__}")
print(f"Config model_type: {model.config.model_type}")

if hasattr(model, 'deit'):
    target_layers = [wrapped_model.model.deit.encoder.layer[-1].layernorm_before]
    print("Backbone: DeiT")
elif hasattr(model, 'vit'):
    target_layers = [wrapped_model.model.vit.encoder.layer[-1].layernorm_before]
    print("Backbone: ViT")
else:
    print([n for n, _ in model.named_children()])
    raise AttributeError("Could not resolve backbone. Check printed attributes above.")

# ── 4. Reshape transform for patch tokens ────────────────────────────────────
def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :]
    result = result.reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result

# ── 5. Preprocessing ──────────────────────────────────────────────────────────
SIZE = processor.size["height"]
MEAN = processor.image_mean
STD  = processor.image_std

eval_transform = Compose([
    Resize((SIZE, SIZE)),
    CenterCrop(SIZE),
    ToTensor(),
    Normalize(mean=MEAN, std=STD),
])

def preprocess(image: Image.Image):
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

# ── 7. Prediction function ────────────────────────────────────────────────────
def predict(image: Image.Image):
    if image is None:
        return None, None, None

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

    gradcam_img = apply_gradcam(image, pred_idx)

    if pred_label == "PNEUMONIA":
        verdict = f"⚠️ PNEUMONIA DETECTED — {confidence*100:.1f}% confidence"
    else:
        verdict = f"✅ NORMAL — {confidence*100:.1f}% confidence"

    return verdict, label_probs, gradcam_img

# ── 8. Gradio interface ───────────────────────────────────────────────────────
with gr.Blocks(title="Pneumonia Screening Tool") as demo:

    gr.Markdown("""
    # 🫁 Pneumonia Detection from Chest X-Ray
    **Model:** DeiT-base-patch16-224 fine-tuned on Kermany dataset  
    **Reference:** Singh et al., *Scientific Reports*, January 2024  
    Upload a chest X-ray image to receive a Normal / Pneumonia classification
    with confidence scores and a Grad-CAM overlay showing which regions
    influenced the model's decision.
    ---
    """)

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                type="pil",
                label="Upload Chest X-Ray",
                height=320,
            )
            run_btn = gr.Button("Analyse", variant="primary", size="lg")

            gr.Markdown("### ⚠️ Disclaimer")
            gr.Markdown(
                "This tool is for **research and educational purposes only**. "
                "It is not a certified medical device and should not be used "
                "for clinical diagnosis."
            )

        with gr.Column(scale=1):
            verdict_text = gr.Textbox(
                label="Prediction",
                interactive=False,
                text_align="center",
            )
            confidence_label = gr.Label(
                label="Class Probabilities",
                num_top_classes=2,
            )
            gradcam_output = gr.Image(
                label="Grad-CAM — Regions influencing the prediction",
                height=320,
            )

    run_btn.click(
        fn=predict,
        inputs=input_image,
        outputs=[verdict_text, confidence_label, gradcam_output],
    )

demo.launch(share=False)