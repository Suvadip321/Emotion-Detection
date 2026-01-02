import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf


# Page configuration
st.set_page_config(page_title="Emotion Detection", page_icon="😊")

# Header
st.title("🎭 Emotion Detection")
st.write("Upload a facial image to detect the emotion")

# Emotion labels mapping
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
EMOTION_EMOJIS = ['😠', '🤢', '😨', '😊', '😐', '😢', '😲']


@st.cache_resource
def load_model():
    """Load the trained emotion detection model."""
    model = tf.keras.models.load_model('best_model.keras')
    return model


def preprocess_image(image):
    """Preprocess the uploaded image for model prediction."""
    # Convert to grayscale
    img_gray = image.convert('L')
    # Resize to 48x48
    img_resized = img_gray.resize((48, 48))
    # Convert to numpy array and normalize
    img_array = np.array(img_resized) / 255.0
    # Reshape for model input (batch_size, height, width, channels)
    img_array = img_array.reshape(1, 48, 48, 1)
    return img_array


def predict_emotion(model, image):
    """Predict the emotion from the preprocessed image."""
    predictions = model.predict(image, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    return EMOTION_LABELS[predicted_class], confidence, predictions[0]


# Load model
try:
    model = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Could not load model: {str(e)}")
    st.info("Please make sure 'best_model.keras' is in the same directory as this app.")

if model_loaded:
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png', 'bmp', 'webp']
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        if st.button("Detect Emotion", type="primary"):
            with st.spinner("Analyzing..."):
                processed_image = preprocess_image(image)
                emotion, confidence, all_probs = predict_emotion(model, processed_image)
                
                emoji_idx = EMOTION_LABELS.index(emotion)
                emoji = EMOTION_EMOJIS[emoji_idx]
                
                st.success(f"**Detected Emotion:** {emoji} {emotion}")
                
                st.subheader("All Probabilities")
                for i, (label, prob) in enumerate(zip(EMOTION_LABELS, all_probs)):
                    st.write(f"{EMOTION_EMOJIS[i]} {label}: {prob*100:.1f}%")
                    st.progress(float(prob))
    else:
        st.info("Please upload an image to get started!")

