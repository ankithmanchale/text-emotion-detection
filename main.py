# emotion_detection/streamlit_app.py
import streamlit as st
import tensorflow as tf
from model import predict_emotion
from tensorflow.keras.models import model_from_json
from preprocessing import clean_text
import pickle
import base64

# Load the tokenizer
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Load the model architecture and weights
json_file2 = open('model2.json', 'r')
loaded_model_json2 = json_file2.read()
json_file2.close()
loaded_model2 = model_from_json(loaded_model_json2)
loaded_model2.load_weights("model2.weights.h5")

# Label mapping
label_map = {'fear': 1, 'sad': 4, 'love': 3, 'joy': 2, 'surprise': 5, 'anger': 0}

# Function to add background image
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image:
        encoded_string = base64.b64encode(image.read())
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
            background-size: cover;
            background-position: top;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


# Streamlit App UI
st.title("🌟 Emotion Detection in Text")
st.markdown("### Enter text below to detect the emotion 👇")

# Text input field for the user
example_text = st.text_input("🔍 Enter a sentence to analyze:")

# Button to trigger prediction
if st.button("✨ Predict Emotion"):
    if example_text:
        # Add a spinner to show prediction in progress
        with st.spinner("Analyzing emotion..."):
            # Predict the emotion using the model
            predicted_emotion2 = predict_emotion(loaded_model2, example_text, label_map)
            # Display the predicted emotion with an emoji based on the emotion
            emotions_emoji = {
                'fear': '😨',
                'sad': '😢',
                'love': '❤️',
                'joy': '😊',
                'surprise': '😲',
                'anger': '😡'
            }
            st.success(f'Predicted Emotion for "{example_text}": {emotions_emoji[predicted_emotion2]} **{predicted_emotion2}**')
    else:
        st.warning("⚠️ Please enter some text to analyze.")

# Add footer
st.markdown("""
    <style>
    footer {visibility: hidden;}
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #f1f1f1;
        color: black;
        text-align: center;
        padding: 10px;
    }
    </style>
    <div class="footer">
        <p>Made with ❤️ </p>
    </div>
""", unsafe_allow_html=True)
