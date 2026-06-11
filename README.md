# ✋ Hand Proximity Detector

## Overview

Hand Proximity Detector is a real-time computer vision application developed using Python, OpenCV, and MediaPipe. The system detects a user's hand through a webcam feed and estimates whether the hand is near or far from the camera by analyzing the depth (Z-axis) information of hand landmarks.

This project demonstrates the application of computer vision, hand tracking, and real-time video processing techniques.



## Features

* Real-time hand detection using webcam input
* Hand landmark tracking using MediaPipe
* Distance estimation based on hand depth information
* Visual display of detected hand landmarks
* Live feedback on hand proximity
* Efficient and lightweight implementation


## Technologies Used

* Python
* OpenCV
* MediaPipe
* NumPy


## Project Structure

Hand-Proximity-Detector/
│
├── face_hand_detection.py
├── object_detection.py
├── MobileNetSSD_deploy.prototxt
├── MobileNetSSD_deploy.caffemodel
├── requirements.txt
└── README.md
```



## How It Works

1. The webcam captures live video frames.
2. MediaPipe detects hand landmarks.
3. The Z-axis coordinate of the index finger tip is extracted.
4. The depth value is analyzed to determine whether the hand is near or far from the camera.
5. The result is displayed in real time along with hand landmark visualization.



## Installation

Clone the repository:

```bash
git clone https://github.com/Akshara-praturi/Hand-Proximity-Detector.git
cd Hand-Proximity-Detector
```

Install dependencies:

```bash
pip install -r requirements.txt
```



## Usage

Run the application:

```bash
python face_hand_detection.py
```

Ensure that your webcam is connected and accessible.



## Applications

* Gesture-based Human Computer Interaction (HCI)
* Touchless User Interfaces
* Smart Surveillance Systems
* Virtual Reality and Augmented Reality Applications
* Interactive Learning Systems


## Learning Outcomes

Through this project, I gained practical experience in:

* Computer Vision
* Real-Time Video Processing
* Hand Landmark Detection
* MediaPipe Framework
* OpenCV Integration
* Python-Based AI Applications



## Future Enhancements

* Hand gesture recognition
* Multi-hand tracking
* Distance measurement calibration
* Integration with IoT devices
* Gesture-controlled system commands




Passionate about Artificial Intelligence, Machine Learning, and Computer Vision.
