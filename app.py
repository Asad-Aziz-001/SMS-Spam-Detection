import streamlit as st
import joblib

# =============================
# 1. Load Model & Vectorizer
# =============================
try:
    model = joblib.load("spam_detector.joblib")
    vectorizer = joblib.load("tfidf_vectorizer.joblib")
except:
    model, vectorizer = None, None

# =============================
# 2. Streamlit UI Config
# =============================
st.set_page_config(
    page_title="Spam Detector",
    page_icon="📧",
    layout="wide"
)

# Sidebar Navigation
st.sidebar.title("📌 Navigation")
menu = st.sidebar.radio("Go to", ["Home", "About"])

# =============================
# 3. Home Page (Prediction)
# =============================
if menu == "Home":
    st.title("📧 Email / SMS Spam Detection")
    st.markdown("### 🔍 Detect spam instantly with Machine Learning")

    with st.container():
        st.markdown("Enter your email or SMS message below:")
        user_input = st.text_area("", placeholder="Type or paste your message here...", height=150)

        if st.button("🚀 Analyze Message", use_container_width=True):
            if not user_input.strip():
                st.warning("⚠️ Please enter a message before predicting.")
            elif model is None or vectorizer is None:
                st.error("❌ Model files not found. Please train and save the model first.")
            else:
                # Transform input
                input_tfidf = vectorizer.transform([user_input])
                prediction = model.predict(input_tfidf)[0]
                proba = (
                    model.predict_proba(input_tfidf)[0][prediction]
                    if hasattr(model, "predict_proba") else None
                )

                # Display result card
                if prediction == 1:
                    st.error("🚨 This looks like **SPAM**!")
                else:
                    st.success("✅ This looks like **NOT Spam (Ham)**.")

                # Confidence with progress bar
                if proba is not None:
                    st.write("### 🔒 Confidence Level")
                    st.progress(float(proba))

# =============================
# 4. About Page
# =============================
elif menu == "About":
    st.title("ℹ️ About This App")
    st.markdown("""
    This **Spam Detector App** was built with:
    - 🧠 Machine Learning (TF-IDF + Classifier)
    - 📊 Scikit-learn
    - 🎨 Streamlit for modern UI

    **How it works:**
    1. Input message → Text converted into numerical features (TF-IDF).  
    2. ML model predicts **Spam (1)** or **Ham (0)**.  
    3. Result is displayed with a confidence score.  

    🔮 Future Upgrades:
    - File upload (CSV of multiple messages)  
    - Word cloud visualization  
    - Advanced deep learning models (BERT, LSTMs)
    """)

    # Divider
    st.markdown("---")

    # Profile Card
    st.markdown("""
    <div style="background-color:#f8f9fa; padding:20px; border-radius:15px; text-align:center; box-shadow: 0px 4px 8px rgba(0,0,0,0.1);">
        <h2 style="color:#333;">👨‍💻 Developer</h2>
        <h3 style="margin-top:10px; color:#0078D7;">ASAD AZIZ</h3>
        <p style="color:gray;">Student of BS-Artificial Intelligence | AI Enthusiast | Developer</p>
        <a href="https://github.com/Asad-Aziz-001" target="_blank" style="margin:10px; text-decoration:none;">🐙 GitHub</a> |
        <a href="https://www.linkedin.com/in/asad-aziz-140p" target="_blank" style="margin:10px; text-decoration:none;">💼 LinkedIn</a>
    </div>
    """, unsafe_allow_html=True)
