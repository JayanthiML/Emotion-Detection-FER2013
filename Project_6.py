## Face Detection + TTA + Emotion Detection ---- Final till Now

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
from PIL import Image, ImageDraw
from facenet_pytorch import MTCNN
import mediapipe as mp
import cv2

# Page config
st.set_page_config(page_title="Emotion Detection", layout="centered")

# Custom CSS for styling
st.markdown("""
    <style>
        .main {
            background-color: #f5f5f5;
        }
        .title {
            text-align: center;
            font-size: 3em;
            margin-bottom: 0.5em;
            color: #333333;
        }
        .subtitle {
            font-size: 1.2em;
            font-weight: bold;
            color: #444444;
        }
        .emotion-box {
            font-size: 1.1em;
            color: #007BFF;
            font-weight: 600;
        }
        .confidence-box {
            font-size: 1.1em;
            color: #FF4500;
            font-weight: 600;
        }
        .image-box {
            border: 2px solid #ddd;
            padding: 10px;
            border-radius: 5px;
        }
        .results-box {
            background-color: #F8F9FA;
            padding: 15px;
            border-radius: 5px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
    </style>
""", unsafe_allow_html=True)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Emotion labels
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load model
@st.cache_resource
def load_model():
    with st.spinner("Loading emotion detection model..."):
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(model.classifier[1].in_features, 7)
        )
        checkpoint = torch.load("emotion_detection_model.pth", map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
    return model

model = load_model()

# Load MTCNN and MediaPipe
mtcnn = MTCNN(keep_all=True, device=device)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Transform
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Title
st.markdown("<div class='title'>😃 Emotion Detection</div>", unsafe_allow_html=True)

# Toggle face detection
use_face_detection = st.checkbox("🔍 Use Face Detection", value=False)

# Toggle TTA
use_tta = st.checkbox("🧪 Apply Test-Time Augmentation (TTA)", value=False)

# Upload image
uploaded_file = st.file_uploader("📁 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    detected_emotions = []
    confidence_scores = []
    draw = ImageDraw.Draw(image)

    with st.spinner("Analyzing the image..."):
        if use_face_detection:
            boxes, _ = mtcnn.detect(image)

            if boxes is None or len(boxes) == 0:
                st.warning("No faces detected. Try another image.")
            else:
                for box in boxes:
                    x1, y1, x2, y2 = [int(coord) for coord in box]
                    face = image.crop((x1, y1, x2, y2))
                    face_tensor = transform(face).unsqueeze(0).to(device)

                    with torch.no_grad():
                        output = model(face_tensor)
                        probs = torch.softmax(output, dim=1)
                        pred = torch.argmax(probs, dim=1).item()
                        emotion = class_names[pred]
                        confidence = probs[0, pred].item() * 100

                        detected_emotions.append(emotion)
                        confidence_scores.append(confidence)

                    # Draw bounding box and emotion label with confidence
                    draw.rectangle(((x1, y1), (x2, y2)), outline="green", width=2)
                    draw.text((x1, y1 - 10), f"{emotion} ({confidence:.1f}%)", fill="green")

                    # Landmark detection using MediaPipe
                    face_cv = cv2.cvtColor(np.array(face), cv2.COLOR_RGB2BGR)
                    results = face_mesh.process(face_cv)

                    if results.multi_face_landmarks:
                        for landmarks in results.multi_face_landmarks:
                            annotated_image = np.array(face).copy()
                            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
                            mp_drawing.draw_landmarks(
                                image=annotated_image,
                                landmark_list=landmarks,
                                connections=mp_face_mesh.FACEMESH_TESSELATION,
                                landmark_drawing_spec=drawing_spec,
                                connection_drawing_spec=drawing_spec
                            )
                            image.paste(Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)), (x1, y1))
        else:
            # No face detection - use whole image
            def get_tta_predictions(image):
                augmentations = [
                    lambda x: x,
                    lambda x: x.transpose(Image.FLIP_LEFT_RIGHT),
                    lambda x: x.rotate(15),
                    lambda x: x.rotate(-15),
                    lambda x: x.transpose(Image.FLIP_LEFT_RIGHT).rotate(15)
                ]
                logits = []
                for aug in augmentations:
                    aug_image = aug(image)
                    face_tensor = transform(aug_image).unsqueeze(0).to(device)
                    with torch.no_grad():
                        out = model(face_tensor)
                        logits.append(out)

                avg_logits = torch.mean(torch.stack(logits), dim=0)
                probs = torch.softmax(avg_logits, dim=1)
                pred = torch.argmax(probs, dim=1).item()
                return class_names[pred], probs[0, pred].item() * 100

            if use_tta:
                emotion, confidence = get_tta_predictions(image)
            else:
                face_tensor = transform(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = model(face_tensor)
                    probs = torch.softmax(output, dim=1)
                    pred = torch.argmax(probs, dim=1).item()
                    emotion = class_names[pred]
                    confidence = probs[0, pred].item() * 100

            detected_emotions.append(emotion)
            confidence_scores.append(confidence)

    # Layout: image and results
    col1, col2 = st.columns([1.5, 1])

    with col1:
        st.image(image, caption="🖼️ Processed Image", use_container_width=True)

    with col2:
        st.markdown("<div class='subtitle'>🎭 Detected Emotions:</div>", unsafe_allow_html=True)
        for i, (emotion, confidence) in enumerate(zip(detected_emotions, confidence_scores), 1):
            st.markdown(f"<div class='emotion-box'>Face {i}: {emotion}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='confidence-box'>Confidence: {confidence:.1f}%</div>", unsafe_allow_html=True)
