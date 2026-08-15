# 🚀 Podium Height Estimation AI

A high-precision, power-optimized computer vision pipeline built with **OpenCV** and **MediaPipe**. This project leverages a modular architecture to provide real-time anatomical height detection for digital podiums while minimizing computational overhead through an intelligent face-gated standby system.

---

## 🔥 Key Features

*   **⚡ Face-Gated Activation**: Uses an ultra-lightweight **Haar Cascade** pass to scan for human presence. The heavy 3D skeletal tracking engine only initializes when a face is detected, significantly reducing idle CPU/GPU usage.
*   **📐 Precision Skeletal Metrics**: Employs MediaPipe's **3D World Landmarks** to calculate anatomical segment lengths (Torso, Thigh, Shin) in real-world centimeters, completely independent of camera perspective.
*   **🏗️ Modular Architecture**: Cleanly separated logic with a dedicated `utils.py` gateway module for face detection, making the codebase scalable and professional.
*   **🏟️ Optimized for Podiums**: Specifically engineered to handle typical podium occlusions by prioritizing upper-body structural vectors.

---

## 🛠️ Project Structure

```text
├── podium.py                      # Main application & pose estimation engine
├── utils.py                       # Modular FaceGateway using Haar Cascades
├── haarcascade_frontalface.xml    # Pre-trained Haar Cascade model
├── requirements.txt               # Project dependencies
└── README.md                      # Documentation
```

---

## 💻 Tech Stack

*   **Language**: Python 3.8+
*   **Computer Vision**: OpenCV, MediaPipe
*   **Math**: NumPy
*   **Models**: Haar Cascade (Face), BlazePose GHUM 3D (Skeletal)

---

## 🚀 Getting Started

### 1. Installation
Ensure you have the required libraries installed:
```bash
pip install opencv-python mediapipe numpy
```

### 2. Usage
Simply run the main script to initialize the system in standby mode:
```bash
python podium.py
```
*   **Standby Mode**: The system will display "STANDBY" and run a low-power face scan.
*   **Active Mode**: Upon detecting a face, the full 3D skeletal tracking will engage, displaying real-time metrics and anatomical vectors.

---

## ⚙️ How It Works

1.  **Gating Phase**: `FaceGateway` in `utils.py` performs a rapid grayscale pass using `detectMultiScale` to establish a facial boundary.
2.  **Tracking Phase**: Once gated, the system processes frames through MediaPipe's 3D Pose engine to extract metrics.
3.  **Vector Calculation**: Uses **Euclidean Distance** to compute structural segments across targeted joints:
    $$Distance = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}$$
4.  **UI Overlay**: Renders real-time telemetry and anti-aliased tracking lines directly onto the video feed.

---

💡 *Developed as a high-performance solution for accurate height analysis in live broadcast and podium environments.*
