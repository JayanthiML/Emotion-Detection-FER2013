# Emotion-Detection-FER2013

This repository contains the code for an emotion detection system built using **EfficientNet-B2** as the backbone model, **MixUp**, and **Test-Time Augmentation (TTA)** for enhanced performance. The system also includes a face detection module that uses **MTCNN** and **MediaPipe** to detect and analyze emotions in images.

## Features

- **Emotion Detection:** Detects emotions such as Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.
- **Face Detection:** Identifies faces using MTCNN and draws bounding boxes around them.
- **Test-Time Augmentation (TTA):** Applies multiple augmentations to improve the model's predictions.
- **Confidence Scores:** Displays the prediction confidence for each emotion detected.
- **Real-time Analysis:** Analyze uploaded images for emotion recognition in real time.

## Requirements

To run this project, you'll need to install the dependencies listed in the `requirements.txt` file.

```bash
pip install -r requirements.txt
