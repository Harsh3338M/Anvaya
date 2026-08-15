# Anvaya
# Anvaya 

**Anvaya** is a real-time assistive tool that bridges communication barriers by converting hand gestures and sign language into spoken words. The system captures video frames via a web interface, processes hand landmarks and gestures using a computer vision pipeline, predicts the corresponding signs, converts them to text, and synthesizes speech output in real time.

---

## 📌 Features

- **Real-Time Landmark Detection:** Captures hand gestures frame-by-frame using a web camera feed.
- **Gesture Classification:** Custom machine learning model trained to classify gestures and sign language alphabets/words.
- **Text & Speech Generation:** Instantly converts recognized gestures into readable text and natural audio speech (TTS).
- **Web-Based UI:** Lightweight, responsive interface for video input capture and output display.

---

## 🛠️ System Architecture

1. **Input:** Web camera captures real-time video frames via the web interface.
2. **Preprocessing & Feature Extraction:** OpenCV and MediaPipe extract key 2D/3D hand coordinates and landmark vectors.
3. **Inference:** A trained deep learning model evaluates the coordinate sequence/frame to predict the gesture.
4. **Text-to-Speech (TTS):** The recognized gesture string is buffered into sentences and synthesized into spoken audio.

---

## 🧰 Tech Stack

| Domain | Tools & Frameworks |
|---|---|
| **Computer Vision** | OpenCV, MediaPipe Hands |
| **Model Training** | Python, TensorFlow / Keras (or PyTorch), Scikit-learn, NumPy |
| **Backend & API** | Flask / FastAPI (WebSocket/REST) |
| **Frontend** | HTML5, CSS3, JavaScript (React / Vanilla JS) |
| **Text-to-Speech** | Web Speech API / gTTS / pyttsx3 |

---

## 📂 Project Structure

```text
anvaya/
├── model_training/
│   ├── dataset/              # Collected landmark / image datasets
│   ├── data_collection.py    # Script to record gesture samples
│   ├── train_model.py        # Model architecture & training pipeline
│   └── gesture_model.h5      # Serialized trained model weights
├── backend/
│   ├── app.py                # API server & inference endpoints
│   ├── requirements.txt      # Python dependencies
│   └── utils.py              # Landmark preprocessing & TTS helpers
├── frontend/
│   ├── index.html            # Main web UI
│   ├── style.css             # Styling
│   └── script.js             # Webcam integration & API calls
└── README.md
