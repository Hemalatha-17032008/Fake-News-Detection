import streamlit as st
import joblib


# -----------------------------------
# Load trained model and vectorizer
# -----------------------------------

model = joblib.load("models/fake_news_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")


# -----------------------------------
# Page configuration
# -----------------------------------

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)


# -----------------------------------
# Title
# -----------------------------------

st.title("📰 Fake News Detection System")

st.write(
    "Enter a news article below and the machine-learning "
    "model will analyze its text and classify it as "
    "possibly fake or possibly real."
)

st.info(
    "⚠️ This system provides a machine-learning prediction. "
    "It does not independently fact-check the news."
)


# -----------------------------------
# News article input
# -----------------------------------

news_text = st.text_area(
    "📝 Enter News Article",
    height=300,
    placeholder="Paste the complete news article here..."
)


# -----------------------------------
# Analyze button
# -----------------------------------

if st.button("🔍 Analyze News"):

    if not news_text.strip():

        st.warning("Please enter a news article first.")

    else:

        # Convert article into TF-IDF features
        text_vector = vectorizer.transform([news_text])


        # Make prediction
        prediction = model.predict(text_vector)[0]


        # Get prediction probabilities
        probabilities = model.predict_proba(text_vector)[0]

        confidence = max(probabilities) * 100


        # -----------------------------------
        # Display prediction
        # -----------------------------------

        if str(prediction).upper() == "FAKE":

            st.error("⚠️ Prediction: POSSIBLY FAKE")

        else:

            st.success("✅ Prediction: POSSIBLY REAL")


        # -----------------------------------
        # Display confidence
        # -----------------------------------

        st.metric(
            "Model Confidence",
            f"{confidence:.2f}%"
        )


        # -----------------------------------
        # Additional information
        # -----------------------------------

        st.write("### Prediction Details")

        col1, col2 = st.columns(2)

        with col1:

            fake_probability = probabilities[
                list(model.classes_).index("FAKE")
            ] * 100

            st.metric(
                "FAKE Probability",
                f"{fake_probability:.2f}%"
            )


        with col2:

            real_probability = probabilities[
                list(model.classes_).index("REAL")
            ] * 100

            st.metric(
                "REAL Probability",
                f"{real_probability:.2f}%"
            )


        st.caption(
            "The prediction is based on patterns learned from "
            "the training dataset and should not be treated as "
            "independent verification of factual accuracy."
        )