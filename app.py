import os
import tempfile
import numpy as np
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from src.evaluate import predict_single, _load_model_and_classes

# Set page config for a premium feel
st.set_page_config(
    page_title="Devanagari Recognizer | AAI Internship",
    page_icon="✈️",
    layout="centered"
)

# Custom CSS for a cleaner, wider look
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        max-width: 800px;
    }
    h1 {
        text-align: center;
        color: #1E3A8A;
    }
    .stButton>button {
        width: 100%;
        background-color: #1E3A8A;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

MODEL_PATH = "model/pretrained_consonant_model.pkl"
DATA_DIR = "data/train"

@st.cache_resource(show_spinner=False)
def load_model():
    """Load model once and cache it."""
    try:
        model, class_names = _load_model_and_classes(MODEL_PATH, DATA_DIR)
        return True
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return False

# Initialize Model
model_ready = load_model()

st.title("Devanagari Character Classifier")
st.markdown("Draw a Devanagari consonant below (like 'ka' or 'kha') and the from-scratch neural network will predict it.")

if not model_ready:
    st.stop()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Draw Here")
    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 1)",
        stroke_width=20,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=256,
        width=256,
        drawing_mode="freedraw",
        key="canvas",
    )

with col2:
    st.subheader("Prediction")
    
    # Upload option as alternative
    uploaded_file = st.file_uploader("Or upload an image", type=["png", "jpg", "jpeg"])
    
    if st.button("Predict 🚀"):
        temp_file_path = None
        
        # 1. Check if user uploaded a file
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tf:
                tf.write(uploaded_file.getbuffer())
                temp_file_path = tf.name
                
        # 2. Or check if user drew something
        elif canvas_result.image_data is not None:
            # image_data is a numpy array (H, W, 4)
            # Check if canvas is essentially empty (all black)
            # The background is black [0, 0, 0, 255]
            if np.sum(canvas_result.image_data[:, :, 0]) > 0:
                img = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
                # Save to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tf:
                    # Convert to RGB (white on black)
                    img.convert("RGB").save(tf.name)
                    temp_file_path = tf.name
            else:
                st.warning("Canvas is empty. Please draw something or upload an image.")
                st.stop()
        else:
            st.warning("Please draw something or upload an image.")
            st.stop()

        # Run inference
        if temp_file_path:
            with st.spinner("Analyzing..."):
                predicted_class, confidence, probs = predict_single(
                    model_path=MODEL_PATH,
                    image_path=temp_file_path,
                    data_dir=DATA_DIR,
                    return_probs=True
                )
                
                _, class_names = _load_model_and_classes(MODEL_PATH, DATA_DIR)

            # Cleanup
            os.unlink(temp_file_path)

            # Display Results
            st.success(f"**Prediction:** {predicted_class}")
            st.info(f"**Confidence:** {confidence:.2%}")
            
            # Show top 5 probabilities as a bar chart
            if probs.size > 0:
                st.write("**Top 5 Probabilities:**")
                top_k = 5
                sorted_idx = np.argsort(probs)[::-1][:top_k]
                
                # Format for streamlit bar chart
                chart_data = {
                    class_names[idx] if idx < len(class_names) else f"class_{idx}": float(probs[idx])
                    for idx in sorted_idx
                }
                st.bar_chart(chart_data)
