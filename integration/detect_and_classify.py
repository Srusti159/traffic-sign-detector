import os
import sys
import json
import argparse
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from ultralytics import YOLO

# Add root directory to module search path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from classification.preprocessing import preprocess_single_image
from classification.traffic_sign_info import GTSRB_CLASSES

class TrafficSignDetectionAndRecognitionSystem:
    """
    Two-Stage Deep Learning Pipeline:
      Stage 1: YOLOv8 Object Detection (Localization: Where is the sign?)
      Stage 2: MobileNetV2 Transfer Learning Classifier (Recognition: What sign is it?)
    """
    def __init__(
        self,
        yolo_weights="models/best_yolo.pt",
        classifier_weights="models/traffic_sign_classifier.keras",
        class_mapping_path="models/class_indices.json"
    ):
        print("⏳ Initializing Traffic Sign Detection & Recognition System...")

        if not os.path.exists(yolo_weights):
            raise FileNotFoundError(f"YOLO weights not found: {yolo_weights}")
        if not os.path.exists(classifier_weights):
            raise FileNotFoundError(f"Classifier weights not found: {classifier_weights}")
        if not os.path.exists(class_mapping_path):
            raise FileNotFoundError(f"Class mapping not found: {class_mapping_path}")

        # 1. Load YOLOv8 Detector
        self.detector = YOLO(yolo_weights)

        # 2. Load MobileNetV2 Classifier
        self.classifier = tf.keras.models.load_model(classifier_weights)

        # 3. Load Class Mapping
        with open(class_mapping_path, 'r') as f:
            self.class_mapping = json.load(f)

        self.index_to_class = self.class_mapping["index_to_class"]
        self.index_to_name = self.class_mapping["index_to_name"]
        print("✅ Pipeline loaded and ready for inference!")

    def process_image(self, image_path, yolo_conf=0.25, padding_ratio=0.05):
        """
        Executes end-to-end detection + classification on a road image.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise ValueError(f"Failed to read image at: {image_path}")
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = img_rgb.shape
        annotated_bgr = img_bgr.copy()

        # Step 1: YOLO Detection
        results = self.detector.predict(image_path, conf=yolo_conf, verbose=False)[0]
        detections = []

        for box in results.boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy
            det_conf = float(box.conf[0].cpu().numpy())

            # Step 2: Crop with slight contextual padding
            pad_x = int((x2 - x1) * padding_ratio)
            pad_y = int((y2 - y1) * padding_ratio)
            cx1, cy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
            cx2, cy2 = min(w, x2 + pad_x), min(h, y2 + pad_y)

            crop_rgb = img_rgb[cy1:cy2, cx1:cx2]
            if crop_rgb.size == 0 or crop_rgb.shape[0] < 5 or crop_rgb.shape[1] < 5:
                continue

            # Step 3: Preprocess Crop for MobileNetV2 (128x128, normalized [0, 1])
            crop_input = preprocess_single_image(crop_rgb, target_size=(128, 128))

            # Step 4: MobileNetV2 Classification
            probs = self.classifier.predict(crop_input, verbose=0)[0]
            pred_idx = int(np.argmax(probs))
            cls_conf = float(probs[pred_idx])

            # Step 5: Interpretation & Action Lookup
            dir_name = self.index_to_class[str(pred_idx)]
            gtsrb_id = int(dir_name)
            info = GTSRB_CLASSES.get(gtsrb_id, {
                "name": self.index_to_name.get(str(pred_idx), "Unknown"),
                "meaning": "N/A",
                "action": "N/A"
            })

            detection_record = {
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
                "det_confidence": det_conf,
                "gtsrb_id": gtsrb_id,
                "sign_name": info["name"],
                "cls_confidence": cls_conf,
                "meaning": info["meaning"],
                "action": info["action"]
            }
            detections.append(detection_record)

            # Step 6: Draw Visual Annotations on Image
            cv2.rectangle(annotated_bgr, (x1, y1), (x2, y2), (0, 220, 0), 3)
            label = f"{info['name']} ({cls_conf * 100:.1f}%)"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(annotated_bgr, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), (0, 220, 0), -1)
            cv2.putText(annotated_bgr, label, (x1 + 3, max(14, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

        return annotated_bgr, detections

def main():
    parser = argparse.ArgumentParser(description="End-to-End Traffic Sign Detection and Recognition System.")
    parser.add_argument("--image", type=str, required=True, help="Path to input road scene image")
    parser.add_argument("--yolo_weights", type=str, default="models/best_yolo.pt", help="Path to trained YOLO weights")
    parser.add_argument("--classifier_weights", type=str, default="models/traffic_sign_classifier.keras", help="Path to classifier")
    parser.add_argument("--mapping", type=str, default="models/class_indices.json", help="Path to class_indices.json")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO detection confidence threshold")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save annotated result")
    args = parser.parse_args()

    pipeline = TrafficSignDetectionAndRecognitionSystem(
        yolo_weights=args.yolo_weights,
        classifier_weights=args.classifier_weights,
        class_mapping_path=args.mapping
    )

    annotated_img, detections = pipeline.process_image(args.image, yolo_conf=args.conf)

    print("\n" + "=" * 75)
    print(f"🛣️  END-TO-END PIPELINE RESULT FOR: {args.image}")
    print("=" * 75)
    if not detections:
        print("   No traffic signs detected in the scene.")
    for i, d in enumerate(detections):
        print(f"\n   [Detected Sign #{i+1}]")
        print(f"   • Location (BBox):           {d['bbox']}")
        print(f"   • Detection Confidence:      {d['det_confidence']*100:.1f}%")
        print(f"   • Traffic Sign Category:     {d['sign_name']} (Class ID {d['gtsrb_id']})")
        print(f"   • Classification Confidence: {d['cls_confidence']*100:.2f}%")
        print(f"   • Sign Meaning:              {d['meaning']}")
        print(f"   • Recommended Driver Action: {d['action']}")
    print("=" * 75)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"integrated_{os.path.basename(args.image)}")
    cv2.imwrite(out_path, annotated_img)
    print(f"\n✅ Annotated road scene saved to: {out_path}\n")

if __name__ == "__main__":
    main()
