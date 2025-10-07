# app.py
import streamlit as st
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# -------------------------------
# 1️⃣ App title
# -------------------------------
st.set_page_config(page_title="IMDB Sentiment Classifier", page_icon="🎬", layout="centered")
st.title("🎬 IMDB Movie Review Sentiment Analysis")
st.markdown(
    "Enter a movie review below and get the predicted sentiment (Positive/Negative) with confidence score."
)

# -------------------------------
# 2️⃣ Load trained model
# -------------------------------
@st.cache_resource(show_spinner=True)
def load_model():
    device = 0 if torch.cuda.is_available() else -1

    tokenizer = AutoTokenizer.from_pretrained("ary08/best_sentiment_model")
    model = AutoModelForSequenceClassification.from_pretrained(
        "ary08/best_sentiment_model",
        load_in_8bit=True,
        device_map="auto"
    )

    clf = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=device
    )
    return clf
    
clf = load_model()

# -------------------------------
# 3️⃣ Example reviews
# -------------------------------
example_reviews = [
    "I absolutely loved this movie! The acting was fantastic and the story was gripping.",
    "Terrible film. Waste of time. The plot made no sense and the acting was awful.",
    "It was okay, some parts were good, but it dragged on too long.",
]

# Store review text in session_state for smooth example usage
if 'review_text' not in st.session_state:
    st.session_state['review_text'] = ""

# Example buttons
st.subheader("Try Example Reviews")
for i, ex in enumerate(example_reviews):
    if st.button(f"Use Example: {ex[:50]}...", key=i):
        st.session_state['review_text'] = ex

# -------------------------------
# 4️⃣ User input
# -------------------------------
review_text = st.text_area(
    "Enter your movie review here:",
    value=st.session_state.get('review_text', ''),
    height=150,
    placeholder="Type or paste your review..."
)

# -------------------------------
# 5️⃣ Prediction button
# -------------------------------
if st.button("Predict Sentiment"):
    if review_text.strip() == "":
        st.warning("Please enter a review to predict.")
    else:
        with st.spinner("Analyzing review..."):
            try:
                results = clf(review_text, truncation=True, max_length=256)
                label_raw = results[0]['label']
                score = results[0]['score']

                # Normalize label
                label_map = {"LABEL_0": "NEGATIVE", "LABEL_1": "POSITIVE"}
                label = label_map.get(label_raw.upper(), label_raw.upper())

                # Display results with color
                if label == "POSITIVE":
                    st.success(f"**Predicted Sentiment:** {label} ({score:.2f})")
                else:
                    st.error(f"**Predicted Sentiment:** {label} ({score:.2f})")

                # Optional: Show confidence progress bar
                st.progress(int(score * 100))

            except Exception as e:
                st.error(f"An error occurred during prediction:\n{e}")
