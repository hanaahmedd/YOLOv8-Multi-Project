# main_yolo_hub.py
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile
import os
import time
from pathlib import Path

# -------------------------
# Page config & style
# -------------------------
st.set_page_config(page_title="YOLO Multi-Project Hub", page_icon="🤖", layout="wide")

PAGE_CSS = """
<style>
/* background gradient */
[data-testid="stAppViewContainer"] {
  background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
  color: #fff;
}

/* title */
h1 {
  text-align: center;
  color: #ffe082;
  font-size: 44px;
  text-shadow: 0 0 12px rgba(255,224,130,0.3);
}

/* buttons style */
div.stButton > button {
  background: linear-gradient(90deg,#ff8a65,#ff4081);
  color: white;
  border: none;
  border-radius: 14px;
  padding: 18px 16px;
  font-size: 18px;
  font-weight: 700;
  box-shadow: 0 8px 18px rgba(0,0,0,0.35);
  transition: transform 0.14s ease, box-shadow 0.14s ease;
}
div.stButton > button:hover {
  transform: translateY(-6px);
  box-shadow: 0 18px 30px rgba(0,0,0,0.45);
}

/* card description */
.card-desc {
  color: #e6eef7;
  text-align:center;
  font-size:14px;
  margin-top:6px;
}
.footer {
  text-align:center;
  margin-top:30px;
  color: #dbe9f7;
  opacity: 0.8;
}
</style>
"""
st.markdown(PAGE_CSS, unsafe_allow_html=True)

st.markdown("<h1>🚀 YOLO Multi-Project Control Center</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#dbe9f7'>Upload your image/video for each project. Models are YOLOv8.</p>", unsafe_allow_html=True)

# -------------------------
# Paths to your .pt models
# -------------------------
MODEL_PATHS = {
    "parking": r"E:\hana\instant ai\18\parking counter\parking_counter.pt",
    "money": r"E:\hana\instant ai\18\money recognition\money_recognition.pt",
    "plate": r"E:\hana\instant ai\18\license plat detection\license_plate_detection.pt",
    "football": r"E:\hana\instant ai\18\football match\football_match.pt",
    "people": None,  # we will use yolov8n for people detection (built-in)
    "brain": r"E:\hana\instant ai\18\brain tumor\brain_tumor_segmentation.pt",
}

# -------------------------
# Cache/load models once
# -------------------------
@st.cache_resource(show_spinner=False)
def load_all_models():
    models = {}
    # load user models; if file missing, store None and show warning later
    for key, p in MODEL_PATHS.items():
        try:
            if key == "people":
                models[key] = YOLO("yolov8n.pt")
            else:
                if p is None:
                    models[key] = None
                else:
                    models[key] = YOLO(p)
        except Exception as e:
            models[key] = None
    return models

models = load_all_models()

# -------------------------
# Helper utilities
# -------------------------
def pil_to_cv2(img_pil):
    img = np.array(img_pil)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def run_model_on_image(model, img_cv2):
    # model accepts PIL or np array; return annotated image (BGR)
    res = model(img_cv2)
    annotated = res[0].plot()  # returns numpy array in RGB or BGR depending on version; convert safely
    # Ensure BGR for cv2 operations and for st.image conversion we'll convert to RGB
    if annotated.shape[2] == 3:
        # ultralytics usually returns RGB; convert to BGR for consistency
        annotated = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
    return annotated, res

def process_video_and_write(model, input_path, out_path, max_frames=None, classes=None, progress_bar=None):
    cap = cv2.VideoCapture(str(input_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_path), fourcc, fps if fps>0 else 25, (w,h))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else 0
    processed = 0
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if max_frames is not None and frame_idx > max_frames:
            break
        # run model
        results = model(frame, classes=classes, verbose=False)
        annotated = results[0].plot()
        # ultralytics returns RGB arrays; convert to BGR for writer
        if annotated.shape[2] == 3:
            annotated = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        writer.write(annotated)
        processed += 1
        if progress_bar and total_frames:
            progress_bar.progress(min(processed/total_frames, 1.0))
    cap.release()
    writer.release()
    if progress_bar:
        progress_bar.progress(1.0)
    return out_path

def save_uploadedfile_to_tempfile(uploaded_file):
    t = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix)
    t.write(uploaded_file.read())
    t.flush()
    t.close()
    return t.name

# -------------------------
# Main grid menu (6 buttons)
# -------------------------
cols = st.columns(3, gap="large")
project_keys = [
    ("parking", "🚗 Parking Counter", "Detect free/occupied spaces in a parking video"),
    ("money", "💵 Money Recognition", "Detect and sum money notes in an image"),
    ("plate", "📷 License Plate Detection", "Detect license plates and show plate regions"),
    ("football", "⚽ Football Match", "Label players and referees in match video"),
    ("people", "🚶 People Entry Counter", "Count people entering using a line crossing"),
    ("brain", "🧠 Brain Tumor Segmentation", "Segment tumor regions in MRI images"),
]

# Show buttons
for i, (key, title, desc) in enumerate(project_keys):
    with cols[i % 3]:
        if st.button(f"{title}", use_container_width=True):
            st.session_state.selected = key
        st.markdown(f"<div class='card-desc'>{desc}</div>", unsafe_allow_html=True)

st.write("---")

# Initialize selection state
if "selected" not in st.session_state:
    st.session_state.selected = None

# Back to menu helper
def back_to_menu():
    st.session_state.selected = None

# -------------------------
# Project pages
# -------------------------
sel = st.session_state.selected

# 1) Parking Counter
if sel == "parking":
    st.markdown('<div class="title">🚗 Parking Space Counter</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Upload a video to detect free and occupied parking spaces</div>', unsafe_allow_html=True)

    model = models.get("parking")
    if model is None:
        st.error("⚠️ Parking model failed to load. Check the path in MODEL_PATHS.")
    else:
        uploaded_video = st.file_uploader("🎥 Upload Parking Lot Video", type=["mp4", "avi", "mov"])

        if uploaded_video:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_video.read())
            video_path = tfile.name

            # Open video
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_count = 0

            st.subheader("⚡ Fast Labeled & Counted Video")
            progress = st.progress(0)
            stframe = st.empty()

            frame_skip = 15  # Speed-up factor

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                if frame_count % frame_skip != 0:
                    continue

                # Run YOLO detection
                results = model(frame, verbose=False)
                annotated_frame = results[0].plot()

                # Count detections by class
                free_spaces = 0
                occupied_spaces = 0
                for box in results[0].boxes:
                    cls = int(box.cls[0])
                    if cls == 0:
                        free_spaces += 1
                    else:
                        occupied_spaces += 1

                # Display counts
                cv2.putText(annotated_frame, f"Free: {free_spaces}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
                cv2.putText(annotated_frame, f"Occupied: {occupied_spaces}", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                stframe.image(annotated_frame, channels="BGR", use_column_width=True)
                progress.progress(min(frame_count / total_frames, 1.0))

            cap.release()
            st.success("✅ Processing complete!")

        else:
            st.info("📤 Please upload a parking lot video to start detection.")

    if st.button("⬅️ Back to Menu"):
        back_to_menu()

# 2) Money Recognition
elif sel == "money":
    st.header("💵 Money Recognition")

    if models.get("money") is None:
        st.error("Money model failed to load. Check the path in MODEL_PATHS.")
    else:
        uploaded = st.file_uploader("Upload image of money", type=["jpg", "jpeg", "png"])
        if uploaded:
            # Load image
            img = Image.open(uploaded).convert("RGB")
            img_cv2 = pil_to_cv2(img)  # Convert PIL to OpenCV format

            with st.spinner("Detecting money..."):
                annotated, results = run_model_on_image(models["money"], img_cv2)

            # Keep original colors for display
            display_img = annotated.copy()

            # Resize image to smaller width while keeping aspect ratio
            max_width = 600
            height, width = display_img.shape[:2]
            scale = max_width / width
            new_size = (int(width * scale), int(height * scale))
            display_img_small = cv2.resize(display_img, new_size)

            # Show resized labeled image
            st.image(display_img_small, caption="💵 Detected Money")

        if st.button("⬅️ Back to Menu"):
            back_to_menu()

# 3) License Plate Detection
elif sel == "plate":
    st.header("📷 License Plate Detection")

    if models.get("plate") is None:
        st.error("License plate model failed to load. Check the path in MODEL_PATHS.")
    else:
        uploaded = st.file_uploader("📸 Upload a vehicle image", type=["jpg", "jpeg", "png"])
        if uploaded:
            # Convert image for processing
            img = Image.open(uploaded).convert("RGB")
            img_cv2 = pil_to_cv2(img)

            with st.spinner("🔍 Detecting license plates..."):
                annotated, results = run_model_on_image(models["plate"], img_cv2)

            # Show annotated image
            display_img = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            st.image(display_img, width=500, caption="Detected License Plates")

            # Show detected license plate crops (no OCR)
            boxes = results[0].boxes.xyxy.cpu().numpy() if len(results[0].boxes) > 0 else []
            if len(boxes) > 0:
                st.markdown("### 🔍 Detected Plate Crops")
                crop_cols = st.columns(len(boxes))

                for i, b in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, b[:4])
                    crop = img_cv2[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    with crop_cols[i]:
                        st.image(crop_rgb, use_column_width=True, caption=f"Plate {i+1}")

                st.success("✅ Detection completed successfully!")
            else:
                st.info("No license plate detected.")

    if st.button("⬅️ Back to Menu"):
        back_to_menu()

# 4) Football Match
elif sel == "football":
    st.header("⚽ Football Match Labeling")
    
    if models.get("football") is None:
        st.error("Football model failed to load. Check the path in MODEL_PATHS.")
    else:
        uploaded_video = st.file_uploader("Upload football match video", type=["mp4","avi","mov"])
        if uploaded_video:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_video.read())

            # Load model
            model = models["football"]

            # Read video
            cap = cv2.VideoCapture(tfile.name)
            stframe = st.empty()

            # ✅ Frame skip option for speed
            frame_skip = st.slider("🎞️ Process every Nth frame", 1, 10, 3)
            
            progress = st.progress(0)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            processed = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Skip frames for speed
                if processed % frame_skip != 0:
                    processed += 1
                    continue

                # Resize for faster inference
                frame = cv2.resize(frame, (640, 360))

                # Run YOLO in faster mode
                results = model(frame, stream=True, imgsz=480)

                # Draw thinner boxes & smaller text
                for r in results:
                    frame = r.plot(line_width=1, font_size=0.4)

                # Show the frame instantly
                stframe.image(frame, channels="BGR", use_container_width=True)

                progress.progress(min(processed / frame_count, 1.0))
                processed += 1

            cap.release()
            st.success("✅ Video processing finished!")

    if st.button("⬅️ Back to Menu"):
        back_to_menu()

# 5) People Entering Counter
elif sel == "people":
    st.header("🚶 People Entering Counter")

    uploaded = st.file_uploader("Upload entrance video", type=["mp4", "avi", "mov"])
    if uploaded:
        # Save uploaded file to a temporary path
        tmp_in = save_uploadedfile_to_tempfile(uploaded)
        cap = cv2.VideoCapture(tmp_in)
        stframe = st.empty()

        # --- Tracking setup ---
        tracker = {}
        next_id = 0
        enter_count = 0

        # --- Helper: Get center ---
        def get_center(xyxy):
            x1, y1, x2, y2 = xyxy
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            return cx, cy

        # --- Main loop ---
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            height, width = frame.shape[:2]

            # Vertical line across whole frame
            line_x = int(width * 0.35)
            line_start = (line_x, 0)
            line_end = (line_x, height)

            # Detect people
            results = models["people"](frame, classes=[0], verbose=False)
            detections = results[0].boxes.xyxy.cpu().numpy() if len(results[0].boxes) > 0 else []
            for det in detections:
                x1, y1, x2, y2 = map(int, det[:4])
                cx, cy = get_center((x1, y1, x2, y2))

                # Try to match this detection to existing IDs
                same_person = None
                for pid, prev_center in tracker.items():
                    px, py = prev_center
                    if abs(cx - px) < 40 and abs(cy - py) < 40:
                        same_person = pid
                        tracker[pid] = (cx, cy)
                        break

                # New person
                if same_person is None:
                    tracker[next_id] = (cx, cy)
                    same_person = next_id
                    next_id += 1

                # --- Count every time a red dot comes near the line ---
                if abs(cx - line_x) <= 2:
                    enter_count += 1

                # Draw visuals
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(frame, f"ID {same_person}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            # Draw counting line and total
            cv2.line(frame, line_start, line_end, (0, 0, 255), 3)
            cv2.putText(frame, f"Entered: {enter_count}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

            # Display in Streamlit
            stframe.image(frame, channels="BGR", use_container_width=True)

        cap.release()
        st.success(f"✅ Counting finished — Total Entered: {enter_count}")

    if st.button("⬅️ Back to Menu"):
        back_to_menu()

# 6) Brain Tumor Segmentation
elif sel == "brain":
    st.header("🧠 Brain Tumor Segmentation")
    if models.get("brain") is None:
        st.error("Brain tumor model failed to load. Check the path in MODEL_PATHS.")
    else:
        uploaded = st.file_uploader("Upload MRI image (jpg/png)", type=["jpg","jpeg","png"])
        if uploaded:
            img = Image.open(uploaded).convert("RGB")
            img_cv2 = pil_to_cv2(img)
            with st.spinner("Running segmentation model..."):
                annotated, results = run_model_on_image(models["brain"], img_cv2)
            display_img = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            st.image(display_img, width=500, caption="Segmentation / Detection result")
            st.info("If you want pixel-level masks or percent area, we can extend this to extract masks from results.")
    if st.button("⬅️ Back to Menu"):
        back_to_menu()

# Default: show welcome when no selection
else:
    st.markdown("<p style='text-align:center;font-size:20px;color:#e6eef7'>Select a project above to start 👆</p>", unsafe_allow_html=True)
    st.markdown("<div class='footer'>Created by ❤️HANA❤️ — YOLOv8 projects hub </div>", unsafe_allow_html=True)
