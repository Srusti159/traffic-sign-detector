import os
import sys
import time
import argparse
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from ultralytics import YOLO
import json

# Add root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from classification.preprocessing import preprocess_single_image
from classification.traffic_sign_info import GTSRB_CLASSES

def process_video_stream(
    video_source=0,  # 0 for webcam, or path to .mp4
    yolo_weights="models/best_yolo.pt",
    classifier_weights="models/traffic_sign_classifier.keras",
    class_mapping_path="models/class_indices.json",
    conf_thresh=0.25,
    output_video_path=None
):
    """
    Processes video stream / webcam feed in real-time.
    Runs YOLOv8 sign detection + MobileNetV2 classification on each frame.
    """
    print(f"⏳ Loading models for real-time video stream...")
    detector = YOLO(yolo_weights)
    classifier = tf.keras.models.load_model(classifier_weights)
    with open(class_mapping_path, 'r') as f:
        mapping = json.load(f)

    index_to_class = mapping["index_to_class"]
    index_to_name = mapping["index_to_name"]

    # Open Video Source
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video source: {video_source}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps): fps = 25.0

    # Video Writer (if saving output)
    writer = None
    if output_video_path:
        os.makedirs(os.path.dirname(output_video_path) or '.', exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    print("✅ Video stream started! Press 'q' in the display window to exit.\n")

    prev_time = time.time()
    frame_count = 0

    while cap.isOpened():
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_count += 1
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # 1. Run YOLO Detector
        results = detector.predict(frame_bgr, conf=conf_thresh, verbose=False)[0]

        # 2. Process Detections
        for box in results.boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy

            # Crop with 5% padding
            pad_x = int((x2 - x1) * 0.05)
            pad_y = int((y2 - y1) * 0.05)
            cx1, cy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
            cx2, cy2 = min(w, x2 + pad_x), min(h, y2 + pad_y)

            crop_rgb = frame_rgb[cy1:cy2, cx1:cx2]
            if crop_rgb.size == 0 or crop_rgb.shape[0] < 5 or crop_rgb.shape[1] < 5:
                continue

            # Preprocess crop (128x128, [0, 1])
            crop_input = preprocess_single_image(crop_rgb, target_size=(128, 128))

            # MobileNetV2 Classify
            probs = classifier.predict(crop_input, verbose=0)[0]
            pred_idx = int(np.argmax(probs))
            cls_conf = float(probs[pred_idx])

            # Class metadata
            dir_name = index_to_class[str(pred_idx)]
            gtsrb_id = int(dir_name)
            info = GTSRB_CLASSES.get(gtsrb_id, {
                "name": index_to_name.get(str(pred_idx), "Sign"),
                "meaning": "N/A",
                "action": "N/A"
            })

            # Draw visual bounding box & text
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 230, 0), 2)
            label = f"{info['name']} ({cls_conf*100:.0f}%)"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame_bgr, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), (0, 230, 0), -1)
            cv2.putText(frame_bgr, label, (x1 + 2, max(12, y1 - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # Calculate & Display Live FPS
        curr_time = time.time()
        live_fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        cv2.putText(frame_bgr, f"FPS: {live_fps:.1f} | Frame: {frame_count}", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if writer:
            writer.write(frame_bgr)

        # Display window
        cv2.imshow("Traffic Sign Detection & Recognition System (Real-Time)", frame_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n🛑 Stopped by user.")
            break

    cap.release()
    if writer:
        writer.release()
        print(f"✅ Processed video saved to: {output_video_path}")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time Video / Webcam Traffic Sign System.")
    parser.add_argument("--source", type=str, default="0", help="Webcam index (0) or path to video file (.mp4)")
    parser.add_argument("--yolo", type=str, default="models/best_yolo.pt", help="Path to YOLO weights")
    parser.add_argument("--classifier", type=str, default="models/traffic_sign_classifier.keras", help="Path to classifier")
    parser.add_argument("--mapping", type=str, default="models/class_indices.json", help="Path to mapping")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--output", type=str, default=None, help="Optional output video file path")
    args = parser.parse_args()

    # Convert numeric webcam index
    src = int(args.source) if args.source.isdigit() else args.source
    process_video_stream(
        video_source=src,
        yolo_weights=args.yolo,
        classifier_weights=args.classifier,
        class_mapping_path=args.mapping,
        conf_thresh=args.conf,
        output_video_path=args.output
    )
