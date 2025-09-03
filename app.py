import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
import cv2


@st.cache_resource
def load_model():
    return YOLO("best_14.pt")

model = load_model()

disease_classes = [
    'Aphid', 'Black Rust', 'Blast', 'Brown Rust', 'Common Root Rot',
    'Fusarium Head', 'Leaf Blight', 'Mildew', 'Mite', 'Septoria',
    'Smut', 'Stem_fly', 'Tan spot', 'yellow_rust'
]

st.title("🌾 Wheat Disease Detection")
st.info("ℹ️ **Note:** This model can detect only these diseases: Aphid, Black Rust, Blast, Brown Rust, Common Root Rot, " \
"Fusarium Head, Leaf Blight, Mildew, Mite, Septoria, Smut, Stem_fly, Tan spot, Yellow Rust")

st.write("Upload an image to detect plant diseases using YOLOv8")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image (Original)", use_column_width=True)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    temp_file.write(uploaded_file.read())
    temp_file.close()

    # Run YOLO inference
    st.write("🔍 Detecting diseases...")
    results = model.predict(source=temp_file.name, conf=0.25, save=False)

    # Show detection results
    for r in results:
        # Annotated image (BGR by default)
        im_array = r.plot()
        # Convert back to RGB so colors match original
        im_array = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)

        st.image(im_array, caption="Detected Diseases", use_column_width=True)

        # Extract detected class names
        detected = [disease_classes[int(c)] for c in r.boxes.cls.cpu().numpy()]
        st.subheader("🦠 Detected Diseases:")
        if detected:
            st.write(", ".join(set(detected)))
        else:
            st.write("✅ No disease detected!")

    # Cleanup temp file
    os.remove(temp_file.name)
