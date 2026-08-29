import os
import argparse
import cv2
from ultralytics import YOLO

def predict_road_scene_detection(
    image_path,
    model_path="models/best_yolo.pt",
    conf_thresh=0.25,
    save_output=True,
    output_dir="results"
):
    """
    Performs standalone YOLOv8 object detection on a full road scene image.
    Outputs localized bounding boxes for all detected traffic signs.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"YOLO model weights not found at: {model_path}")

    model = YOLO(model_path)
    results = model.predict(image_path, conf=conf_thresh, verbose=False)[0]

    detections = []
    for box in results.boxes:
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        conf = float(box.conf[0].cpu().numpy())
        detections.append({
            "box": (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])),
            "confidence": conf
        })

    print("\n" + "=" * 65)
    print("🚗 YOLO ROAD SCENE DETECTION RESULT")
    print("=" * 65)
    print(f"File:                   {image_path}")
    print(f"Total Signs Detected:   {len(detections)}")
    for i, d in enumerate(detections):
        print(f"  • Sign #{i+1}: Bounding Box {d['box']} (Conf: {d['confidence']*100:.1f}%)")
    print("=" * 65)

    if save_output:
        os.makedirs(output_dir, exist_ok=True)
        annotated_bgr = results.plot()
        out_filename = f"yolo_det_{os.path.basename(image_path)}"
        out_path = os.path.join(output_dir, out_filename)
        cv2.imwrite(out_path, annotated_bgr)
        print(f"✅ Annotated detection saved to: {out_path}\n")

    return detections

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect traffic signs in a road scene using YOLOv8.")
    parser.add_argument("--image", type=str, required=True, help="Path to input road image")
    parser.add_argument("--model", type=str, default="models/best_yolo.pt", help="Path to YOLO weights")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save annotated image")
    args = parser.parse_args()

    predict_road_scene_detection(args.image, args.model, args.conf, save_output=True, output_dir=args.output_dir)
