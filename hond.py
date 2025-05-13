import streamlit as st
import cv2
import os
from datetime import datetime
from ultralytics import YOLO
import numpy as np

# Pad voor foto's
SAVE_PATH = r"C:\Users\tomgo\OneDrive\Bureaublad\Data Science\Track SI\Edge Computing"

# Laad YOLOv8-model
model = YOLO("yolov8n.pt")

st.title("Motion Detection")
start_camera = st.button("Start Webcam")
take_photos = st.checkbox("📸 Foto's nemen bij persoon-detectie")

frame_placeholder = st.empty()

def detect_motion_with_yolo(frame):
    results = model(frame)
    annotated_frame = results[0].plot()
    return annotated_frame, results

if start_camera:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("❌ Kan de camera niet openen.")
    else:
        st.info("Druk op 'Stop' om de stream te beëindigen.")
        stop = st.button("🛑 Stop")
        
        while cap.isOpened() and not stop:
            ret, frame = cap.read()
            if not ret:
                st.error("❌ Kan frame niet lezen.")
                break

            frame = cv2.resize(frame, (640, 480))
            annotated_frame, results = detect_motion_with_yolo(frame)

            # Controleer of een 'person' is gedetecteerd
            if take_photos:
                for box in results[0].boxes:
                    cls_id = int(box.cls[0].item())
                    class_name = model.names[cls_id]
                    if class_name == "dog":
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"dog{timestamp}.jpg"
                        save_full_path = os.path.join(SAVE_PATH, filename)
                        cv2.imwrite(save_full_path, frame)
                        st.success(f"📷 Foto opgeslagen: {filename}")
                        break  # Sla maar één keer per frame op

            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(annotated_frame_rgb, channels="RGB")

        cap.release()