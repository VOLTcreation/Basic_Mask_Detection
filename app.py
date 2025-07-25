import cv2
import numpy as np
import json
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import cvlib as cv  # pip install cvlib

# === Load model and class names ===
model = load_model("mask_data.h5")

with open("class_names.json", "r") as f:
    raw_class_names = json.load(f)

# Remap labels to "mask on" / "mask off"
class_names = ["mask on" if c == "face_with_mask" else "mask off" for c in raw_class_names]

# === Start webcam ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Cannot open webcam")
    exit()

print("✅ Webcam started. Press 'q' or 'ESC' to exit.")

# === Set fullscreen window ===
cv2.namedWindow("Mask Detection", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Mask Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# === FPS timer ===
prev_time = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Cannot read frame")
            break

        # Detect faces with cvlib
        faces, confidences = cv.detect_face(frame)

        for i, face_coords in enumerate(faces):
            (x1, y1), (x2, y2) = face_coords[:2], face_coords[2:]

            # Crop and preprocess
            face = frame[y1:y2, x1:x2]
            try:
                face = cv2.resize(face, (224, 224))
                face_array = img_to_array(face)
                face_array = preprocess_input(face_array)
                face_array = np.expand_dims(face_array, axis=0)
            except:
                continue  # Skip badly cropped faces

            # Predict
            pred = model.predict(face_array)[0]
            label_idx = np.argmax(pred) if len(pred) > 1 else int(pred > 0.5)
            label = class_names[label_idx]
            confidence = float(pred[label_idx]) if len(pred) > 1 else float(pred)

            # Draw bounding box
            color = (0, 255, 0) if label == "mask on" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Label background
            label_text = f"{label} ({confidence * 100:.1f}%)"
            (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w + 10, y1), color, -1)
            cv2.putText(frame, label_text, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # === Show FPS ===
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # === Top Bar ===
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(frame, "😷 Real-Time Mask Detection | Press 'q' or ESC to Quit", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        # Show frame
        cv2.imshow("Mask Detection", frame)

        key = cv2.waitKey(1)
        if key == ord('q') or key == 27:  # ESC key
            break

except KeyboardInterrupt:
    print("\n🔌 Stopped by user (Ctrl+C)")

finally:
    print("🛑 Releasing camera and closing windows...")
    cap.release()
    cv2.destroyAllWindows()
