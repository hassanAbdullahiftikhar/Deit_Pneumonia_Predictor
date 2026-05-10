# PneumoScan AI — Advanced Pneumonia Detection

A professional, clinical-grade deep learning application for screening pneumonia in chest X-rays using **Data-efficient Image Transformers (DeiT)**. This tool provides both high-accuracy classification and visual explainability through **Grad-CAM** attention maps.

## Features
- **Accurate Classification:** Identifies "Normal" vs. "Pneumonia" cases with high confidence.
- **Visual Explainability:** Generates Grad-CAM heatmaps to visualize the model's focus areas within the lungs.
- **Premium Dashboard:** A custom, modern medical UI designed for clarity and rapid analysis.
- **FastAPI Backend:** High-performance inference engine serving the custom frontend.
- **Transformer-based:** Leverages state-of-the-art Vision Transformers (DeiT).

## Installation

Follow these steps to set up the environment and run the application:

### 1. Clone the repository
```bash
git clone <repository-url>
cd Deit_Pneumonia_Predictor
```

### 2. Create a Virtual Environment
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

## Usage

### Running the Application
Execute the following command to launch the FastAPI server and the premium dashboard:
```bash
python app.py
```
Once started, the application will be available at:
**http://localhost:8000**

## Model Information
- **Architecture:** `DeiT-base-patch16-224` (Data-efficient Image Transformer)
- **Dataset:** Fine-tuned on the Kermany Chest X-Ray dataset.
- **Reference:** Singh et al., *Scientific Reports*, January 2024.

## Project Structure
- `app.py`: FastAPI backend server (Inference logic + API endpoints).
- `frontend/`: Custom web interface (HTML, CSS, JavaScript).
- `requirements.txt`: Python package dependencies.
- `deit-pneumonia-papersplit/`: Local directory containing the fine-tuned model weights.
- `sample_xrays/`: Folder containing example images for testing.
- `walkthrough.md`: Detailed overview of the latest implementation round.

## Disclaimer
**This tool is for research and educational purposes only.** It is not a certified medical device and should not be used for clinical diagnosis or as a substitute for professional medical advice.

---
*Developed as part of the CV_Project repository.*