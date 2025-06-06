# Object-detection
# Real-Time Object Pickup Detection using YOLOv8 and MediaPipe

This project is a real-time computer vision solution that detects when an object is picked up by a hand using a combination of **YOLOv8** for object detection and **MediaPipe** for hand tracking.

##  Features

- Detect objects using YOLOv8 (`yolov8s.pt`)
- Track hands using MediaPipe
- Recognize when a hand interacts with (picks up) an object
- Real-time webcam feed with overlays
- Displays messages like:
  - `Picked up: [Object Label]`
  - `No object interaction detected`
  - `Object detected, but not recognized` (for unknown objects)

 Architecture

          ┌─────────────┐
          │ MediaPipe   │ → Hand bounding box
          └────┬────────┘
               │
               ▼
          ┌─────────────┐
          │ YOLOv8      │ → Object detection + location
          └────┬────────┘
               │
     ┌─────────▼───────── ------ ─┐
     │ Logic:                     │
     │ - Is object in hand?       │
     │ - Is it moving with hand?  │
     └─────────┬───────── ------ ─┘
               │
           PICKED UP
           
---

##  Requirements

Make sure you have Python 3.7 or later. Then install dependencies:

```bash
pip install ultralytics mediapipe opencv-python

           
