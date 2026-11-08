
import streamlit as st
import joblib

# Page configuration
st.set_page_config(
    page_title="Email Spam Classifier",
    page_icon="📧",
    layout="centered"
)

# Load model once and cache it
@st.cache_resource
def load_model():
    return joblib.load("spam_classifier.pkl")

model = load_model()

# Title
st.title("📧 Email Spam Classifier")

st.write(
    "Enter a message below and the machine learning model "
    "will predict whether it is spam or not."
)

# Message input
message = st.text_area(
    "Enter your message:",
    placeholder="Example: Congratulations! You won a free prize!"
)

# Prediction button
if st.button("🔍 Check Message"):

    if message.strip() == "":
        st.warning("Please enter a message.")

    else:
        # Prediction
        prediction = model.predict([message])[0]

        # Probabilities
        probabilities = model.predict_proba([message])[0]

        # Match probabilities to their actual classes
        class_probabilities = dict(zip(model.classes_, probabilities))

        ham_probability = class_probabilities["ham"] * 100
        spam_probability = class_probabilities["spam"] * 100

        # Results
        st.subheader("Prediction")

        if prediction == "spam":
            st.error("🚨 This message is SPAM!")
        else:
            st.success("✅ This message is NOT SPAM!")

        # Confidence
        st.subheader("Model Confidence")

        st.metric(
            "Spam Probability",
            f"{spam_probability:.2f}%"
        )

        st.progress(float(spam_probability / 100))

        # Detailed probabilities
        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Ham Probability",
                f"{ham_probability:.2f}%"
            )

        with col2:
            st.metric(
                "Spam Probability",
                f"{spam_probability:.2f}%"
            )
