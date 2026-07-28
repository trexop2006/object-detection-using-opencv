import av
import streamlit as st
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

st.set_page_config(
    page_title="Real-Time Object Detection",
    page_icon="🎯",
    layout="wide"
)

# -------------------------
# Sidebar
# -------------------------

st.sidebar.title("⚙ Settings")

confidence = st.sidebar.slider(
    "Confidence Threshold",
    0.10,
    1.00,
    0.65,
    0.05
)

st.sidebar.markdown("---")

st.sidebar.subheader("📌 Detectable Objects")

st.sidebar.write("""
- 👤 Person
- 🚗 Car
- 🚌 Bus
- 🏍 Motorcycle
- 🚲 Bicycle
- 💻 Laptop
- 📱 Cell Phone
- ⌨ Keyboard
- 🖱 Mouse
- 📺 TV
- 🧴 Bottle
- 🪑 Chair
- 🛋 Couch
- 🎒 Backpack
- 🐶 Dog
- 🐱 Cat
- ...and more (80 classes)
""")

st.sidebar.markdown("---")
st.sidebar.success("Model : YOLO11n")

# -------------------------
# Main Page
# -------------------------

st.title("🎯 Real-Time Object Detection")
st.write("Detect objects in real time using the pretrained YOLO11 model.")

model = YOLO("yolo11n.pt")

class VideoProcessor(VideoProcessorBase):

    def recv(self, frame):

        image = frame.to_ndarray(format="bgr24")

        results = model(
            image,
            conf=confidence,
            verbose=False
        )

        annotated = results[0].plot()

        boxes = results[0].boxes

        total = len(boxes)

        y = 40

        import cv2

        cv2.putText(
            annotated,
            f"Total Objects : {total}",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

        y += 40

        names = []

        for box in boxes:

            cls = int(box.cls[0])

            names.append(model.names[cls])

        for obj in sorted(set(names)):

            count = names.count(obj)

            cv2.putText(
                annotated,
                f"{obj} : {count}",
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255,255,255),
                2
            )

            y += 30

        return av.VideoFrame.from_ndarray(
            annotated,
            format="bgr24"
        )


webrtc_streamer(
    key="object-detection",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]}
        ]
    }
)
