import streamlit as st
import joblib
import pandas as pd
import re
import string

# Page configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    .main-header p {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
    }
    
    .info-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        color: #2c3e50;
    }
    
    .info-box strong {
        color: #1a1a1a;
    }
    
    .result-container {
        padding: 2rem;
        border-radius: 10px;
        margin: 1.5rem 0;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 600;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        animation: fadeIn 0.5s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .result-real {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
    }
    
    .result-fake {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        color: white;
    }
    
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        padding: 0.75rem 2rem;
        border: none;
        border-radius: 8px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    
    .stTextArea>div>div>textarea {
        border-radius: 8px;
        border: 2px solid #e0e0e0;
        padding: 1rem;
        font-size: 1rem;
    }
    
    .stTextArea>div>div>textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .sidebar-info {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: #2c3e50;
    }
    
    .sidebar-info h2 {
        color: #667eea !important;
    }
    
    .sidebar-info p {
        color: #2c3e50 !important;
    }
    
    .feature-item {
        padding: 0.5rem 0;
        border-bottom: 1px solid rgba(0, 0, 0, 0.1);
    }
    
    .feature-item:last-child {
        border-bottom: none;
    }
    </style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    vectorizer = joblib.load("vectorizer.jb")
    model = joblib.load("lr_model.jb")
    return vectorizer, model

# Load sample articles
@st.cache_data
def load_sample_articles():
    try:
        true_df = pd.read_csv("True.csv")
        fake_df = pd.read_csv("Fake.csv")
        
        # Clean text function (same as training)
        def clean_text(text):
            if pd.isna(text):
                return ""
            text = str(text).lower()
            text = re.sub(r"https?://\S+|www\.\S+", "", text)
            text = re.sub(r"<.*?>+", "", text)
            text = re.sub(r"\[.*?\]", "", text)
            text = re.sub(r"[^a-zA-Z\s]", "", text)
            text = re.sub(r"\s+", " ", text)
            return text.strip()
        
        # Combine title and text for better context
        true_df['full_text'] = true_df['title'].astype(str) + " " + true_df['text'].astype(str)
        fake_df['full_text'] = fake_df['title'].astype(str) + " " + fake_df['text'].astype(str)
        
        # Clean the text
        true_df['full_text'] = true_df['full_text'].apply(clean_text)
        fake_df['full_text'] = fake_df['full_text'].apply(clean_text)
        
        # Remove empty texts
        true_df = true_df[true_df['full_text'].str.len() > 0]
        fake_df = fake_df[fake_df['full_text'].str.len() > 0]
        
        # Limit to first 100 for performance
        true_samples = true_df.head(100)['full_text'].tolist()
        fake_samples = fake_df.head(100)['full_text'].tolist()
        
        return true_samples, fake_samples
    except Exception as e:
        st.error(f"Error loading sample articles: {str(e)}")
        return [], []

vectorizer, model = load_models()
true_samples, fake_samples = load_sample_articles()

# Sidebar
with st.sidebar:
    st.markdown("""
        <div class="sidebar-info">
            <h2 style="color: #667eea !important; margin-top: 0;">ℹ️ About</h2>
            <p style="text-align: justify; line-height: 1.6; color: #2c3e50 !important;">
                This Fake News Detector uses advanced machine learning algorithms 
                to analyze news articles and determine their authenticity. 
                Simply paste or type a news article to get instant results.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### ✨ Features")
    st.markdown("""
        <div class="feature-item">
            <strong>🔬 AI-Powered Analysis</strong><br>
            <small>Advanced ML models trained on thousands of articles</small>
        </div>
        <div class="feature-item">
            <strong>⚡ Instant Results</strong><br>
            <small>Get predictions in real-time</small>
        </div>
        <div class="feature-item">
            <strong>📊 High Accuracy</strong><br>
            <small>Reliable detection using logistic regression</small>
        </div>
        <div class="feature-item">
            <strong>🔒 Privacy First</strong><br>
            <small>Your data is processed securely</small>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📝 How to Use")
    st.markdown("""
        1. Copy a news article text
        2. Paste it in the text area
        3. Click "Analyze News" button
        4. View the results
    """)

# Main content
st.markdown("""
    <div class="main-header">
        <h1>🔍 Fake News Detector</h1>
        <p>AI-Powered News Authenticity Analysis</p>
    </div>
""", unsafe_allow_html=True)

# Info box
st.markdown("""
    <div class="info-box">
        <strong style="color: #1a1a1a;">💡 Instructions:</strong> <span style="color: #2c3e50;">Enter or paste a news article in the text area below 
        and click the "Analyze News" button to check whether it's real or fake. You can also select a sample article from the test dataset.</span>
    </div>
""", unsafe_allow_html=True)

# Sample article selection section
st.markdown("### 📋 Test with Sample Articles")
tab1, tab2 = st.tabs(["✅ Real News Samples", "❌ Fake News Samples"])

selected_sample = None

with tab1:
    if true_samples:
        st.markdown("**Select a real news article to test:**")
        # Create numbered options
        options = [f"Article #{i+1}" for i in range(len(true_samples))]
        selected_idx = st.selectbox(
            "Choose a real news article:",
            range(len(options)),
            format_func=lambda x: options[x],
            key="true_news_select"
        )
        if st.button("Load Selected Article", key="load_true"):
            selected_sample = true_samples[selected_idx]
            st.session_state['selected_text'] = selected_sample
            st.session_state['article_input'] = selected_sample
            st.session_state['expected_label'] = "Real"
            st.rerun()
        st.info(f"📊 Total real news samples available: {len(true_samples)}")
    else:
        st.warning("Real news samples could not be loaded.")

with tab2:
    if fake_samples:
        st.markdown("**Select a fake news article to test:**")
        # Create numbered options
        options = [f"Article #{i+1}" for i in range(len(fake_samples))]
        selected_idx = st.selectbox(
            "Choose a fake news article:",
            range(len(options)),
            format_func=lambda x: options[x],
            key="fake_news_select"
        )
        if st.button("Load Selected Article", key="load_fake"):
            selected_sample = fake_samples[selected_idx]
            st.session_state['selected_text'] = selected_sample
            st.session_state['article_input'] = selected_sample
            st.session_state['expected_label'] = "Fake"
            st.rerun()
        st.info(f"📊 Total fake news samples available: {len(fake_samples)}")
    else:
        st.warning("Fake news samples could not be loaded.")

st.markdown("---")

# Main input area
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("### 📰 News Article")
    
    # Initialize session state if not exists
    if 'selected_text' not in st.session_state:
        st.session_state['selected_text'] = ""
    if 'expected_label' not in st.session_state:
        st.session_state['expected_label'] = None
    if 'article_input' not in st.session_state:
        st.session_state['article_input'] = ""
    
    # Sync session state - if selected_text was updated, update article_input
    if st.session_state.get('selected_text') and st.session_state.get('selected_text') != st.session_state.get('article_input', ''):
        st.session_state['article_input'] = st.session_state['selected_text']
    
    # Use session state for text area
    inputn = st.text_area(
        "Paste or type your news article here:",
        value=st.session_state.get('article_input', ''),
        height=250,
        label_visibility="collapsed",
        placeholder="Enter the news article text here or select a sample above...",
        key="article_input"
    )
    
    # Update selected_text when user manually types
    if inputn != st.session_state.get('selected_text', ''):
        st.session_state['selected_text'] = inputn
    
    # Clear button
    col_clear1, col_clear2, col_clear3 = st.columns([2, 1, 2])
    with col_clear2:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state['selected_text'] = ""
            st.session_state['article_input'] = ""
            st.session_state['expected_label'] = None
            st.rerun()
    
    # Show expected label if sample was selected and text matches
    if st.session_state.get('expected_label') and inputn and inputn == st.session_state.get('selected_text', ''):
        expected = st.session_state['expected_label']
        color = "🟢" if expected == "Real" else "🔴"
        st.caption(f"{color} Expected label: **{expected} News** (from test dataset)")

with col2:
    st.markdown("### 📊 Statistics")
    if inputn:
        word_count = len(inputn.split())
        char_count = len(inputn)
        st.metric("Words", word_count)
        st.metric("Characters", char_count)
    else:
        st.info("Enter text to see statistics")

# Analyze button
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])

with col_btn2:
    analyze_button = st.button("🔍 Analyze News", use_container_width=True)

# Results section
if analyze_button:
    if inputn.strip():
        with st.spinner("🔬 Analyzing article... Please wait."):
            transform_input = vectorizer.transform([inputn])
            prediction = model.predict(transform_input)
            
            # Get prediction probability if available
            try:
                proba = model.predict_proba(transform_input)[0]
                confidence = max(proba) * 100
            except:
                confidence = None
        
        # Check if prediction matches expected label
        predicted_label = "Real" if prediction[0] == 1 else "Fake"
        expected_label = st.session_state.get('expected_label')
        is_correct = None
        if expected_label:
            is_correct = (predicted_label == expected_label)
        
        # Display results
        if prediction[0] == 1:
            result_html = f"""
                <div class="result-container result-real">
                    <h2 style="margin: 0; font-size: 2rem;">✅ REAL NEWS</h2>
                    <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem; opacity: 0.95;">
                        This article appears to be authentic and trustworthy.
                    </p>
                    {f'<p style="margin: 0.5rem 0 0 0; font-size: 0.9rem; opacity: 0.85;">Confidence: {confidence:.1f}%</p>' if confidence else ''}
            """
            if is_correct is not None:
                result_html += f'<p style="margin: 0.5rem 0 0 0; font-size: 1rem; font-weight: bold; color: {"#00ff00" if is_correct else "#ff0000"};">{"✓ Correct Prediction!" if is_correct else "✗ Incorrect Prediction"}</p>'
            result_html += "</div>"
            st.markdown(result_html, unsafe_allow_html=True)
            
            # Additional metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Status", "✅ Authentic", delta="Verified")
            with col2:
                st.metric("Analysis", "Complete", delta="Success")
            with col3:
                if confidence:
                    st.metric("Confidence", f"{confidence:.1f}%", delta=f"{confidence-50:.1f}%")
        else:
            result_html = f"""
                <div class="result-container result-fake">
                    <h2 style="margin: 0; font-size: 2rem;">⚠️ FAKE NEWS</h2>
                    <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem; opacity: 0.95;">
                        This article appears to be fake or misleading. Please verify from reliable sources.
                    </p>
                    {f'<p style="margin: 0.5rem 0 0 0; font-size: 0.9rem; opacity: 0.85;">Confidence: {confidence:.1f}%</p>' if confidence else ''}
            """
            if is_correct is not None:
                result_html += f'<p style="margin: 0.5rem 0 0 0; font-size: 1rem; font-weight: bold; color: {"#00ff00" if is_correct else "#ff0000"};">{"✓ Correct Prediction!" if is_correct else "✗ Incorrect Prediction"}</p>'
            result_html += "</div>"
            st.markdown(result_html, unsafe_allow_html=True)
            
            # Additional metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Status", "⚠️ Suspicious", delta="Warning", delta_color="inverse")
            with col2:
                st.metric("Analysis", "Complete", delta="Detected")
            with col3:
                if confidence:
                    st.metric("Confidence", f"{confidence:.1f}%", delta=f"{confidence-50:.1f}%", delta_color="inverse")
    else:
        st.warning("⚠️ Please enter some text to analyze.")

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>Made with ❤️ using Streamlit | Powered by Machine Learning</p>
    </div>
""", unsafe_allow_html=True) 
