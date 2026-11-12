
import os
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
import torch.nn as nn

from PIL import Image
from torchvision.models import efficientnet_b0
from torchvision import transforms


# ==========================================
# CONFIGURATION
# ==========================================

MODEL_PATH = Path(__file__).parent / "models" / "best_model-v3.pt"

DEVICE = torch.device("cpu")

# ==========================================
# LOAD MODEL
# ==========================================

def load_model():

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}"
        )

    print("Loading Deepfake Detection Model...")

    model = efficientnet_b0(weights=None)

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        2
    )

    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    # Strict loading detects architecture mismatches
    model.load_state_dict(state_dict, strict=True)

    model.to(DEVICE)
    model.eval()

    print("Model loaded successfully!")

    return model


model = load_model()


# ==========================================
# IMAGE PREPROCESSING
# ==========================================

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==========================================
# PREDICTION
# ==========================================

def predict_image(image):

    image = image.convert("RGB")

    tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():

        output = model(tensor)

        probabilities = torch.softmax(output, dim=1)[0]

        confidence, prediction = torch.max(
            probabilities, dim=0
        )

    # Confirm class order against training configuration
    label = "Real" if prediction.item() == 0 else "Deepfake"

    confidence = float(confidence.item() * 100)

    return label, f"{confidence:.2f}%", image


# ==========================================
# VIDEO PROCESSING
# ==========================================

def predict_video(video_path):

    cap = cv2.VideoCapture(video_path)

    try:
        ret, frame = cap.read()

        if not ret:
            raise ValueError("Unable to read video.")

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        image = Image.fromarray(frame)

        label, confidence, preview = predict_image(image)

        return label + " (First Frame)", confidence, preview

    finally:
        cap.release()


# ==========================================
# MAIN FILE HANDLER
# ==========================================

def predict_file(file_path):

    if not file_path:
        raise gr.Error("Please upload an image or video.")

    path = Path(file_path)

    extension = path.suffix.lower()

    try:

        if extension in [".jpg", ".jpeg", ".png"]:

            image = Image.open(path).convert("RGB")

            return predict_image(image)

        elif extension in [".mp4", ".mov"]:

            return predict_video(str(path))

        else:

            raise gr.Error("Unsupported file format.")

    except gr.Error:
        raise

    except Exception as error:

        print("Prediction error:", repr(error))

        raise gr.Error(
            "Prediction failed. Check the application logs."
        )


# ==========================================
# GRADIO INTERFACE
# ==========================================

with gr.Blocks(title="Deepfake Detector") as demo:

    gr.Markdown(
        """
        # Deepfake Detection System

        Upload an image or video to analyze whether
        the content appears Real or Deepfake.

        **Model:** EfficientNet-B0
        """
    )

    file_input = gr.File(
        label="Upload Image or Video",
        file_types=[
            ".jpg",
            ".jpeg",
            ".png",
            ".mp4",
            ".mov"
        ],
        type="filepath"
    )

    with gr.Row():

        prediction = gr.Textbox(
            label="Prediction",
            interactive=False
        )

        confidence = gr.Textbox(
            label="Confidence (%)",
            interactive=False
        )

    preview = gr.Image(
        label="Analyzed Image / Video Frame",
        interactive=False
    )

    file_input.change(
        fn=predict_file,
        inputs=file_input,
        outputs=[
            prediction,
            confidence,
            preview
        ]
    )

    gr.Markdown(
        """
        ---
        Built using PyTorch, EfficientNet-B0 and Gradio.

        Note: Video prediction currently analyzes only
        the first frame, not the entire video.
        """
    )


# ==========================================
# LAUNCH
# ==========================================

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    )