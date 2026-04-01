import streamlit as st
import joblib
import warnings

# Suppress all warnings (including scikit-learn version mismatch)
warnings.filterwarnings('ignore')

# Specifically suppress scikit-learn version warnings
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings('ignore', category=InconsistentVersionWarning)
except:
    pass

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

# =============================
# 3. Global CSS
# =============================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

/* ---- Reset & Base ---- */
html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

/* ---- Background ---- */
.stApp {
    background: #080c14;
    background-image:
        radial-gradient(ellipse 80% 50% at 50% -10%, rgba(0,200,255,0.08) 0%, transparent 70%),
        repeating-linear-gradient(
            0deg,
            transparent,
            transparent 39px,
            rgba(0,200,255,0.025) 39px,
            rgba(0,200,255,0.025) 40px
        ),
        repeating-linear-gradient(
            90deg,
            transparent,
            transparent 39px,
            rgba(0,200,255,0.025) 39px,
            rgba(0,200,255,0.025) 40px
        );
    color: #e8f4f8;
}

/* ---- Sidebar ---- */
[data-testid="stSidebar"] {
    background: #0b1120 !important;
    border-right: 1px solid rgba(0,200,255,0.12);
}
[data-testid="stSidebar"] * {
    color: #b0ccd8 !important;
}

/* ---- Hero Badge ---- */
.hero-badge {
    display: inline-block;
    background: rgba(0,200,255,0.08);
    border: 1px solid rgba(0,200,255,0.3);
    color: #00c8ff;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    padding: 5px 14px;
    border-radius: 2px;
    text-transform: uppercase;
    margin-bottom: 16px;
}

/* ---- Hero Title ---- */
.hero-title {
    font-size: clamp(2rem, 5vw, 3.2rem);
    font-weight: 800;
    line-height: 1.1;
    letter-spacing: -0.02em;
    color: #f0f8ff;
    margin-bottom: 10px;
}
.hero-title span {
    color: #00c8ff;
}

/* ---- Hero Subtitle ---- */
.hero-subtitle {
    font-size: 1rem;
    color: #6a8fa0;
    font-weight: 400;
    margin-bottom: 36px;
    font-family: 'Space Mono', monospace;
    letter-spacing: 0.03em;
}

/* ---- Textarea ---- */
[data-testid="stTextArea"] textarea {
    background: #0d1826 !important;
    border: 1px solid rgba(0,200,255,0.2) !important;
    border-radius: 8px !important;
    color: #c8e8f4 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.88rem !important;
    padding: 16px !important;
    transition: border-color 0.25s ease;
    box-shadow: 0 0 0px rgba(0,200,255,0) !important;
}
[data-testid="stTextArea"] textarea:focus {
    border-color: rgba(0,200,255,0.6) !important;
    box-shadow: 0 0 16px rgba(0,200,255,0.07) !important;
}
[data-testid="stTextArea"] textarea::placeholder {
    color: #2e4a5a !important;
}

/* ---- Button ---- */
.stButton > button {
    background: linear-gradient(135deg, #00c8ff 0%, #0072ff 100%) !important;
    color: #000d1a !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.06em !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 14px 28px !important;
    cursor: pointer !important;
    transition: opacity 0.2s, transform 0.15s !important;
    box-shadow: 0 4px 20px rgba(0,200,255,0.25) !important;
    text-transform: uppercase !important;
}
.stButton > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 28px rgba(0,200,255,0.35) !important;
}
.stButton > button:active {
    transform: translateY(0px) !important;
}

/* ---- Result Cards ---- */
.result-card {
    border-radius: 10px;
    padding: 24px 28px;
    margin-top: 24px;
    display: flex;
    align-items: center;
    gap: 16px;
}
.result-card.spam {
    background: rgba(255, 48, 79, 0.08);
    border: 1px solid rgba(255, 48, 79, 0.35);
}
.result-card.ham {
    background: rgba(0, 230, 150, 0.07);
    border: 1px solid rgba(0, 230, 150, 0.3);
}
.result-icon {
    font-size: 2.2rem;
    line-height: 1;
}
.result-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.result-label.spam { color: rgba(255,80,100,0.7); }
.result-label.ham  { color: rgba(0,220,140,0.7); }
.result-title {
    font-size: 1.5rem;
    font-weight: 800;
    line-height: 1.1;
}
.result-title.spam { color: #ff4060; }
.result-title.ham  { color: #00e696; }

/* ---- Confidence Block ---- */
.confidence-block {
    margin-top: 28px;
    padding: 20px 24px;
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px;
}
.confidence-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.16em;
    color: #4a7080;
    text-transform: uppercase;
    margin-bottom: 10px;
}
.confidence-bar-track {
    height: 6px;
    background: rgba(255,255,255,0.06);
    border-radius: 99px;
    overflow: hidden;
}
.confidence-bar-fill {
    height: 100%;
    border-radius: 99px;
    transition: width 0.6s cubic-bezier(.16,1,.3,1);
}
.confidence-bar-fill.spam {
    background: linear-gradient(90deg, #ff4060, #ff1040);
    box-shadow: 0 0 10px rgba(255,48,80,0.4);
}
.confidence-bar-fill.ham {
    background: linear-gradient(90deg, #00c896, #00e6b4);
    box-shadow: 0 0 10px rgba(0,200,150,0.35);
}
.confidence-pct {
    margin-top: 8px;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    color: #c0d8e4;
    text-align: right;
}

/* ---- Stats Row ---- */
.stats-row {
    display: flex;
    gap: 14px;
    margin-top: 28px;
}
.stat-pill {
    flex: 1;
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 8px;
    padding: 14px 16px;
    text-align: center;
}
.stat-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.25rem;
    font-weight: 700;
    color: #00c8ff;
}
.stat-desc {
    font-size: 0.72rem;
    color: #3a5a6a;
    margin-top: 3px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* ---- Divider ---- */
.fancy-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,200,255,0.2), transparent);
    margin: 32px 0;
}

/* ---- Warning/Error override ---- */
[data-testid="stAlert"] {
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.82rem !important;
}

/* ---- Hide Streamlit branding ---- */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# =============================
# 4. Sidebar Navigation
# =============================
st.sidebar.title("📌 Navigation")
menu = st.sidebar.radio("Go to", ["Home", "About"])

# =============================
# 5. HOME PAGE
# =============================
if menu == "Home":

    # Hero Section
    st.markdown("""
    <div class="hero-badge">⚡ ML-Powered · Real-time Detection</div>
    <div class="hero-title">Detect <span>Spam</span><br>Before It Reaches You.</div>
    <div class="hero-subtitle">// TF-IDF + Classifier · Instant Analysis · Confidence Scoring</div>
    """, unsafe_allow_html=True)

    # Stats Row
    st.markdown("""
    <div class="stats-row">
        <div class="stat-pill">
            <div class="stat-value">TF-IDF</div>
            <div class="stat-desc">Vectorizer</div>
        </div>
        <div class="stat-pill">
            <div class="stat-value">ML</div>
            <div class="stat-desc">Classifier</div>
        </div>
        <div class="stat-pill">
            <div class="stat-value">&lt;1s</div>
            <div class="stat-desc">Latency</div>
        </div>
        <div class="stat-pill">
            <div class="stat-value">2-Class</div>
            <div class="stat-desc">Spam / Ham</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    # Input Area
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("""
        <div style="font-family:'Space Mono',monospace; font-size:0.72rem;
                    letter-spacing:0.15em; color:#4a7080; text-transform:uppercase;
                    margin-bottom:8px;">
            ▸ Paste your message
        </div>
        """, unsafe_allow_html=True)
        user_input = st.text_area(
            label="Message to analyze",
            placeholder="Type or paste your email / SMS message here...",
            height=150,
            label_visibility="collapsed"
        )

    with col2:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        analyze_clicked = st.button("🚀 Analyze", use_container_width=True)

    # Analysis Logic
    if analyze_clicked:
        if not user_input.strip():
            st.warning("⚠️ Please enter a message before analyzing.")
        elif model is None or vectorizer is None:
            st.error("❌ Model files not found. Train and save `spam_detector.joblib` and `tfidf_vectorizer.joblib` first.")
        else:
            input_tfidf = vectorizer.transform([user_input])
            prediction = model.predict(input_tfidf)[0]
            proba = (
                model.predict_proba(input_tfidf)[0][prediction]
                if hasattr(model, "predict_proba") else None
            )

            # Result Card
            if prediction == 1:
                st.markdown("""
                <div class="result-card spam">
                    <div class="result-icon">🚨</div>
                    <div>
                        <div class="result-label spam">Verdict</div>
                        <div class="result-title spam">SPAM Detected</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                bar_class = "spam"
            else:
                st.markdown("""
                <div class="result-card ham">
                    <div class="result-icon">✅</div>
                    <div>
                        <div class="result-label ham">Verdict</div>
                        <div class="result-title ham">Looks Legitimate</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                bar_class = "ham"

            # Confidence Bar
            if proba is not None:
                pct = round(float(proba) * 100, 1)
                st.markdown(f"""
                <div class="confidence-block">
                    <div class="confidence-label">▸ Confidence Score</div>
                    <div class="confidence-bar-track">
                        <div class="confidence-bar-fill {bar_class}" style="width:{pct}%"></div>
                    </div>
                    <div class="confidence-pct">{pct}%</div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:'Space Mono',monospace; font-size:0.68rem;
                color:#263a45; text-align:center; letter-spacing:0.1em;">
        SPAM_DETECTOR · ML ENGINE · BUILT WITH SCIKIT-LEARN + STREAMLIT
    </div>
    """, unsafe_allow_html=True)


# =============================
# 6. ABOUT PAGE (unchanged)
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
    st.markdown("---")
    st.markdown("""
    <div style="background-color:#f8f9fa; padding:20px; border-radius:15px; text-align:center; box-shadow: 0px 4px 8px rgba(0,0,0,0.1);">
        <h2 style="color:#333;">👨‍💻 Developer</h2>
        <h3 style="margin-top:10px; color:#0078D7;">ASAD AZIZ</h3>
        <p style="color:gray;">Student of BS-Artificial Intelligence | AI Enthusiast | Developer</p>
        <a href="https://asad-aziz-001.github.io/Portfolio/" target="_blank" style="margin:10px; text-decoration:none;">🐙 Portfolio</a> |
        <a href="https://www.linkedin.com/in/asad-aziz-ai" target="_blank" style="margin:10px; text-decoration:none;">💼 LinkedIn</a>
    </div>
    """, unsafe_allow_html=True)
