import cv2
import json
import os
import time
from ultralytics import YOLO
from google.colab import files
from google.colab.patches import cv2_imshow

# Step 1: Upload the video file
uploaded = files.upload()  # Upload your video
video_path = list(uploaded.keys())[0]  # Get the uploaded file name
output_json_path = "output_detections.json"
subobject_save_folder = "subobject_images"

# Step 2: Set up YOLO model and classes
model = YOLO('yolov8n.pt')  # Load YOLOv8 lightweight model

# Define main objects and sub-objects for detection
main_objects = ["person", "car"]
sub_objects = ["helmet", "tire", "door"]

# Create folder for saving sub-object images
if not os.path.exists(subobject_save_folder):
    os.makedirs(subobject_save_folder)

# Step 3: Helper functions
def is_within_bbox(sub_bbox, main_bbox):
    sx1, sy1, sx2, sy2 = sub_bbox
    mx1, my1, mx2, my2 = main_bbox
    return sx1 >= mx1 and sy1 >= my1 and sx2 <= mx2 and sy2 <= my2

def save_cropped_image(frame, x1, y1, x2, y2, main_label, sub_label, folder):
    cropped_img = frame[y1:y2, x1:x2]
    filename = f"{folder}/{main_label}_{sub_label}_{x1}_{y1}.jpg"
    cv2.imwrite(filename, cropped_img)

def save_to_json(detection_results, output_path):
    with open(output_path, 'w') as f:
        json.dump(detection_results, f, indent=4)

def visualize_in_colab(frame, detections):
    for det in detections["detections"]:
        x1, y1, x2, y2 = det["bbox"]
        label = det["object"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        for sub in det["subobject"]:
            sub_x1, sub_y1, sub_x2, sub_y2 = sub["bbox"]
            sub_label = sub["object"]
            cv2.rectangle(frame, (sub_x1, sub_y1), (sub_x2, sub_y2), (255, 0, 0), 2)
            cv2.putText(frame, sub_label, (sub_x1, sub_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2_imshow(frame)

# Step 4: Core Processing Function
def process_video(video_path, output_json_path, subobject_save_folder):
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    detection_results = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run YOLO detection
        results = model.predict(frame_rgb, device='cpu')

        # Parse detections
        frame_data = {"frame": frame_idx, "detections": []}
        for result in results.xyxy[0]:  # Loop through detected objects
            x1, y1, x2, y2, conf, cls = map(int, result.tolist()[:6])
            label = model.names[cls]

            if label in main_objects:
                detection = {
                    "object": label,
                    "id": cls,
                    "bbox": [x1, y1, x2, y2],
                    "subobject": []
                }

                # Check for sub-objects
                for sub_result in results.xyxy[0]:
                    sub_x1, sub_y1, sub_x2, sub_y2, sub_conf, sub_cls = map(int, sub_result.tolist()[:6])
                    sub_label = model.names[sub_cls]
                    if sub_label in sub_objects and is_within_bbox([sub_x1, sub_y1, sub_x2, sub_y2], [x1, y1, x2, y2]):
                        subobject_data = {
                            "object": sub_label,
                            "id": sub_cls,
                            "bbox": [sub_x1, sub_y1, sub_x2, sub_y2]
                        }
                        detection["subobject"].append(subobject_data)
                        save_cropped_image(frame, sub_x1, sub_y1, sub_x2, sub_y2, label, sub_label, subobject_save_folder)

                frame_data["detections"].append(detection)

        detection_results.append(frame_data)

        # Optional: Visualize detections
        if frame_idx % 30 == 0:  # Display every 30th frame
            visualize_in_colab(frame, frame_data)

    cap.release()

    end_time = time.time()
    fps = total_frames / (end_time - start_time)
    print(f"Processed {total_frames} frames in {end_time - start_time:.2f} seconds (FPS: {fps:.2f})")

    # Save results
    save_to_json(detection_results, output_json_path)
    print(f"Detection results saved to {output_json_path}")
    print(f"Sub-object images saved in {subobject_save_folder}")

# Step 5: Execute the processing
process_video(video_path, output_json_path, subobject_save_folder)

# Step 6: Download results
files.download(output_json_path)  # Download JSON file

# Zip and download sub-object images
import shutil
shutil.make_archive("subobject_images", 'zip', subobject_save_folder)
files.download("subobject_images.zip")
