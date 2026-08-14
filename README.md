# 🚀 Podium Height Estimation AI

A high-precision computer vision pipeline built with **OpenCV** and **MediaPipe Pose** to dynamically measure anatomical joint structures and track human height metrics in real-time.

Designed specifically to eliminate perspective distortion and posture changes on stages and podium setups.

---

## 🔥 Key Features

* **Anatomical Vector Summing** – Tracks true body segment lengths instead of inaccurate 2D bounding boxes.
* **Torso Stability Vector** – Automatically computes mid-shoulder to mid-hip spatial geometry to maintain tracking stability even if a speaker turns sideways.
* **Jitter-Resistant** – Leverages high-confidence tracking thresholds to filter out background edge noise.
* **Subpixel Precision** – Converts normalized MediaPipe topology into raw pixel coordinate spaces seamlessly.

---

## 🛠️ Project Architecture

[Project Folder]
 ├── podium.py          # Main real-time tracking application script
 ├── requirements.txt   # Core Python library dependencies
 └── README.md          # Project documentation

---

## 💻 Getting Started

### 1. Clone the Repository
git clone https://github.com
cd YOUR_REPO_NAME

### 2. Install Dependencies
pip install -r requirements.txt

### 3. Launch the AI
python podium.py

---

## ⚙️ How It Works Under the Hood

1. **Frame Capture**: Pulls live frames from the stage camera using cv2.VideoCapture().
2. **Pose Processing**: MediaPipe maps 33 high-accuracy 3D physiological landmarks.
3. **Vector Math**: Computes structural Euclidean distance lines across targeted joints using NumPy calculations: Distance = SquareRoot( (x2 - x1)^2 + (y2 - y1)^2 )
4. **Live Overlay**: Projects real-time metrics and coordinate lines on screen using anti-aliased drawing arrays.

---

## 📝 Dependencies

* **Python 3.8+**
* **OpenCV-Python**
* **MediaPipe**
* **NumPy**

---

💡 *Developed to bring pinpoint metric accuracy to real-world live broadcast staging environments.*
