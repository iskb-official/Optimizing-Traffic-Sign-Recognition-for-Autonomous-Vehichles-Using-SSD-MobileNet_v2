import os
import csv
import time
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import load_model  # type: ignore
import cv2
import numpy as np

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.set_printoptions(suppress=True)

# Initialize output directory and files
output_dir = "detection_results"
os.makedirs(output_dir, exist_ok=True)

# Text log file
log_file = os.path.join(output_dir, "detections_log.txt")
with open(log_file, 'a') as f:
    f.write(f"\n\n=== New Session {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")

# CSV file
csv_file = os.path.join(output_dir, "detections.csv")
csv_exists = os.path.exists(csv_file)

# Load model and labels
model = load_model("keras_Model.h5", compile=False)
class_names = [line.strip() for line in open("labels.txt", "r").readlines()]

# Shape mapping for common traffic signs
sign_shapes = {
    "stop": "octagon",
    "speed": "circle",
    "yield": "triangle",
    "warning": "triangle",
    "no entry": "circle",
    "pedestrian": "square",
    # Add more as needed
}

def draw_shape(frame, shape, center, size, color=(0, 255, 0), thickness=2):
    """Draw different shapes based on traffic sign type"""
    if shape == "circle":
        cv2.circle(frame, center, size, color, thickness)
    elif shape == "triangle":
        pts = np.array([
            [center[0], center[1]-size],
            [center[0]-size, center[1]+size],
            [center[0]+size, center[1]+size]
        ], np.int32)
        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=thickness)
    elif shape == "octagon":
        pts = np.array([
            [center[0]+int(size*0.7), center[1]-int(size*0.3)],
            [center[0]+int(size*0.3), center[1]-int(size*0.7)],
            [center[0]-int(size*0.3), center[1]-int(size*0.7)],
            [center[0]-int(size*0.7), center[1]-int(size*0.3)],
            [center[0]-int(size*0.7), center[1]+int(size*0.3)],
            [center[0]-int(size*0.3), center[1]+int(size*0.7)],
            [center[0]+int(size*0.3), center[1]+int(size*0.7)],
            [center[0]+int(size*0.7), center[1]+int(size*0.3)]
        ], np.int32)
        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=thickness)
    else:  # Default rectangle
        cv2.rectangle(frame, 
                     (center[0]-size, center[1]-size),
                     (center[0]+size, center[1]+size), 
                     color, thickness)

# Initialize camera
camera = cv2.VideoCapture(0)
last_print_time = time.time()
frame_count = 0

with open(csv_file, 'a', newline='') as f:
    writer = csv.writer(f)
    if not csv_exists:
        writer.writerow(["Timestamp", "Frame", "Class", "Confidence", "Resolution"])

while True:
    ret, frame = camera.read()
    if not ret:
        print("Failed to capture frame")
        break

    frame_count += 1
    display_frame = frame.copy()
    
    # Preprocess image for model (FIXED THIS LINE)
    img = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
    img_array = (np.asarray(img, dtype=np.float32).reshape(1, 224, 224, 3) / 127.5) - 1

    # Make prediction
    predictions = model.predict(img_array, verbose=0)
    index = np.argmax(predictions)
    confidence = predictions[0][index]
    class_name = class_names[index]

    # Process high-confidence detections
    if confidence > 0.75:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Log to text file
        with open(log_file, 'a') as f:
            f.write(f"{timestamp} - {class_name} ({int(confidence*100)}%)\n")
        
        # Log to CSV
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                frame_count,
                class_name,
                f"{int(confidence*100)}%",
                f"{frame.shape[1]}x{frame.shape[0]}"
            ])

        # Visual detection
        shape = sign_shapes.get(class_name.lower().split()[0], "rectangle")
        center = (frame.shape[1]//2, frame.shape[0]//2)
        size = min(frame.shape[:2])//3
        
        # Draw shape and label
        draw_shape(display_frame, shape, center, size)
        
        label = f"{class_name} {int(confidence*100)}%"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        
        cv2.rectangle(display_frame,
                    (center[0]-w//2, center[1]-size-h-10),
                    (center[0]+w//2, center[1]-size-10),
                    (0, 255, 0), -1)
        
        cv2.putText(display_frame, label,
                   (center[0]-w//2, center[1]-size-15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Display
    cv2.imshow("Traffic Sign Detection", display_frame)
    
    # Exit on ESC
    if cv2.waitKey(1) == 27:
        break

# Cleanup
camera.release()
cv2.destroyAllWindows()
print(f"\nDetection results saved to:\n- {log_file}\n- {csv_file}")