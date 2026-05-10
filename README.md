# 🫁 Pneumonia Detection from Chest X-Ray

A deep learning application for screening pneumonia in chest X-rays using **Data-efficient Image Transformers (DeiT)**. This tool provides both classification results and visual explainability through **Grad-CAM** overlays, highlighting the regions of the lungs that influenced the model's decision.

## 🚀 Features
- **Accurate Classification:** Identifies "Normal" vs. "Pneumonia" cases.
- **Visual Explainability:** Generates Grad-CAM heatmaps to visualize the model's focus areas.
- **Web Interface:** Easy-to-use UI built with Gradio.
- **Transformer-based:** Uses state-of-the-art Vision Transformers (DeiT).

## 🛠️ Installation

Follow these steps to set up the environment and run the application:

### 1. Clone the repository
```bash
git clone <repository-url>
cd Deit_Pneumonia_Predictor
```

### 2. Create a Virtual Environment
It is recommended to use a virtual environment to manage dependencies:
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Running the Web App
Execute the following command to launch the Gradio interface:
```bash
python app.py
```
Once started, the application will provide a local URL (e.g., `http://127.0.0.1:7860`) where you can upload X-ray images for analysis.

## 🧠 Model Information
- **Architecture:** `DeiT-base-patch16-224` (Data-efficient Image Transformer)
- **Dataset:** Fine-tuned on the Kermany Chest X-Ray dataset.
- **Reference:** Singh et al., *Scientific Reports*, January 2024.

## 📂 Project Structure
- `app.py`: Main application script (Gradio UI + Inference logic).
- `requirements.txt`: Python package dependencies.
- `deit-pneumonia-papersplit/`: Local directory containing the fine-tuned model weights and configuration.
- `sample_xrays/`: Folder containing example images for testing.
- `testing.ipynb`: Jupyter notebook for model evaluation and experimentation.

## ⚠️ Disclaimer
**This tool is for research and educational purposes only.** It is not a certified medical device and should not be used for clinical diagnosis or as a substitute for professional medical advice.

---
*Developed as part of the CV_Project repository.*