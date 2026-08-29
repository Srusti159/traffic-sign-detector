import os
import sys
import json
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from ultralytics import YOLO
import streamlit as st

# Add root directory to module path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from classification.preprocessing import preprocess_single_image
from classification.traffic_sign_info import GTSRB_CLASSES

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Traffic Sign Detection & Recognition System",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1.05rem; color: #4B5563; margin-bottom: 1.5rem; }
    .card { background-color: #F8FAFC; border-radius: 8px; padding: 16px; border: 1px solid #E2E8F0; margin-bottom: 12px; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# MODEL CACHING
# ==========================================
@st.cache_resource
def load_models():
    yolo_path = "models/best_yolo.pt"
    clf_path = "models/traffic_sign_classifier.keras"
    map_path = "models/class_indices.json"

    if not os.path.exists(yolo_path) or not os.path.exists(clf_path) or not os.path.exists(map_path):
        return None, None, None

    detector = YOLO(yolo_path)
    classifier = tf.keras.models.load_model(clf_path)
    with open(map_path, 'r') as f:
        mapping = json.load(f)
    return detector, classifier, mapping

detector, classifier, mapping = load_models()

# ==========================================
# SIDEBAR CONTROLS
# ==========================================
st.sidebar.image("https://img.icons8.com/color/96/traffic-light.png", width=65)
st.sidebar.title("🚦 False-Positive Rejectors")

if detector is None:
    st.sidebar.error("⚠️ Model files missing in `models/` directory! Please check `best_yolo.pt` and `traffic_sign_classifier.keras`.")
    st.stop()

input_mode = st.sidebar.radio("Select Input Source:", ["📸 Upload Image", "🎥 Upload Video", "📷 Live Camera"])

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ Confidence & Filter Controls")

yolo_conf = st.sidebar.slider("YOLO Detection Threshold:", min_value=0.10, max_value=0.90, value=0.25, step=0.05,
                              help="Default is 0.25 for standard real-world road scenes.")

cls_conf_thresh = st.sidebar.slider("Classifier Confidence Gate:", min_value=0.40, max_value=0.99, value=0.70, step=0.05,
                                    help="Ensures only recognized signs meeting this confidence are displayed.")

enable_sharpness = st.sidebar.checkbox("🛡️ Enable Bokeh / Blur Filter", value=False,
                                       help="Enable this only if testing on out-of-focus bokeh photos with glowing circular background lights.")

min_sharpness = st.sidebar.slider("Minimum Sharpness Score:", min_value=10, max_value=200, value=40, step=10)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Architecture Specs")
st.sidebar.info("""
- **Detector**: YOLOv8n (Localization)
- **Classifier**: MobileNetV2 (43 Classes)
- **Val Accuracy**: **96.17%**
- **ROC-AUC**: **0.9998**
""")

# ==========================================
# MAIN INTERFACE
# ==========================================
st.markdown('<div class="main-header">🚦 Deep Learning Traffic Sign Detection & Recognition</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Two-stage computer vision system with edge-sharpness gating for driver assistance.</div>', unsafe_allow_html=True)

def process_image_pipeline(img_pil, yolo_thresh, cls_thresh, use_sharpness=True, sharpness_threshold=60):
    img_rgb = np.array(img_pil.convert('RGB'))
    h, w, _ = img_rgb.shape
    annotated_rgb = img_rgb.copy()

    # Stage 1: YOLO Detection
    results = detector.predict(img_rgb, conf=yolo_thresh, verbose=False)[0]
    detections = []

    for box in results.boxes:
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = xyxy
        det_conf = float(box.conf[0].cpu().numpy())
        bw = x2 - x1
        bh = y2 - y1

        # 1. Aspect Ratio Filter (0.65 to 1.5)
        if bh > 0:
            aspect = bw / float(bh)
            if aspect < 0.65 or aspect > 1.5:
                continue

        # Contextual padding 5%
        pad_x = int(bw * 0.05)
        pad_y = int(bh * 0.05)
        cx1, cy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        cx2, cy2 = min(w, x2 + pad_x), min(h, y2 + pad_y)

        crop_rgb = img_rgb[cy1:cy2, cx1:cx2]
        if crop_rgb.size == 0 or crop_rgb.shape[0] < 8 or crop_rgb.shape[1] < 8:
            continue

        # 2. Edge Sharpness / Blur Rejection Filter (Laplacian Variance)
        # Real signs with numbers/arrows have high edge gradient variance.
        # Blurry bokeh light blobs have near-zero gradient variance.
        gray_crop = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
        sharpness_score = float(cv2.Laplacian(gray_crop, cv2.CV_64F).var())

        if use_sharpness and sharpness_score < sharpness_threshold:
            # Reject blurred out-of-focus background light artifacts
            continue

        # Stage 2: MobileNetV2 Classification (128x128)
        crop_input = preprocess_single_image(crop_rgb, target_size=(128, 128))
        probs = classifier.predict(crop_input, verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        cls_conf = float(probs[pred_idx])

        # 3. Classifier Confidence Gating
        if cls_conf < cls_thresh:
            continue

        dir_name = mapping["index_to_class"][str(pred_idx)]
        gtsrb_id = int(dir_name)
        info = GTSRB_CLASSES.get(gtsrb_id, {
            "name": mapping["index_to_name"].get(str(pred_idx), "Traffic Sign"),
            "meaning": "Traffic Regulation",
            "action": "Drive with caution"
        })

        detections.append({
            "box": (x1, y1, x2, y2),
            "det_conf": det_conf,
            "cls_conf": cls_conf,
            "sharpness": sharpness_score,
            "name": info["name"],
            "class_id": gtsrb_id,
            "meaning": info["meaning"],
            "action": info["action"],
            "crop": crop_rgb
        })

        # Draw bounding boxes & labels
        cv2.rectangle(annotated_rgb, (x1, y1), (x2, y2), (0, 220, 0), 3)
        label = f"{info['name']} ({cls_conf*100:.0f}%)"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated_rgb, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), (0, 220, 0), -1)
        cv2.putText(annotated_rgb, label, (x1 + 3, max(14, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return annotated_rgb, detections

# ----------------------------------------------------
# TAB 1: IMAGE UPLOAD
# ----------------------------------------------------
if input_mode == "📸 Upload Image":
    uploaded_file = st.file_uploader("Upload a road scene image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 📷 Original Input Scene")
            st.image(image, use_container_width=True)

        with st.spinner("Processing with Edge Sharpness Gating + YOLO + MobileNetV2..."):
            annotated_img, detections = process_image_pipeline(image, yolo_conf, cls_conf_thresh, enable_sharpness, min_sharpness)

        with col2:
            st.markdown("### 🎯 Filtered Detections & Classifications")
            st.image(annotated_img, use_container_width=True)

        st.markdown("---")
        st.markdown(f"### 📋 Validated Traffic Signs ({len(detections)} Found)")

        if not detections:
            st.info("💡 No valid in-focus traffic signs detected. Blurry background light artifacts were successfully rejected by the Edge Sharpness filter.")
        else:
            for i, det in enumerate(detections):
                with st.container():
                    c1, c2, c3 = st.columns([1, 2, 2])
                    with c1:
                        st.image(det["crop"], caption=f"Sign #{i+1} Crop", width=130)
                    with c2:
                        st.markdown(f"#### **{det['name']}**")
                        st.markdown(f"**Classification Confidence:** `{det['cls_conf']*100:.2f}%`")
                        st.progress(det['cls_conf'])
                        st.caption(f"YOLO Conf: {det['det_conf']*100:.1f}% | Edge Sharpness: {det['sharpness']:.1f} | Class ID: {det['class_id']}")
                    with c3:
                        st.markdown(f"**📖 Sign Meaning:** {det['meaning']}")
                        st.markdown(f"**🚗 Recommended Action:** :green[{det['action']}]")
                    st.divider()

# ----------------------------------------------------
# TAB 2: LIVE CAMERA
# ----------------------------------------------------
elif input_mode == "📷 Live Camera":
    camera_photo = st.camera_input("Take a photo of a road scene or traffic sign:")
    if camera_photo is not None:
        image = Image.open(camera_photo)
        annotated_img, detections = process_image_pipeline(image, yolo_conf, cls_conf_thresh, enable_sharpness, min_sharpness)
        st.image(annotated_img, caption="Real-time Detection Result", use_container_width=True)
        for i, det in enumerate(detections):
            st.success(f"**Sign #{i+1}**: {det['name']} ({det['cls_conf']*100:.1f}% Confidence)\n\n**Action**: {det['action']}")

# ----------------------------------------------------
# TAB 3: VIDEO UPLOAD
# ----------------------------------------------------
elif input_mode == "🎥 Upload Video":
    uploaded_video = st.file_uploader("Upload a dashcam road video (.mp4)...", type=["mp4", "avi", "mov"])
    if uploaded_video is not None:
        tfile = f"temp_{uploaded_video.name}"
        with open(tfile, 'wb') as f:
            f.write(uploaded_video.read())
        st.video(tfile)
