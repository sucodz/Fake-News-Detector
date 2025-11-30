"""
Fake News Detector - Streamlit Application

A production-quality application for detecting fake news using machine learning.
Uses logistic regression model and TF-IDF vectorizer for predictions.

Installation:
    pip install streamlit pandas numpy joblib scikit-learn matplotlib

Usage:
    streamlit run app.py

Requirements:
    - vectorizer.jb (TF-IDF vectorizer saved with joblib)
    - lr_model.jb (Logistic regression model saved with joblib)
    - True.csv (Dataset with true news articles)
    - Fake.csv (Dataset with fake news articles)

All files should be in the same directory as app.py.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from typing import List, Tuple, Dict, Optional
import io
import re
import string
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# Page configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS for modern, minimal design
st.markdown("""
<style>
    /* Main styling */
    .main {
        background-color: #fafbfc;
    }
    
    /* Header styling */
    .header-container {
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 1.5rem 0;
        margin-bottom: 2rem;
        border-bottom: 1px solid #e0e0e0;
    }
    
    .header-logo {
        margin-right: 1rem;
    }
    
    .header-title {
        font-size: 2rem;
        font-weight: 600;
        color: #1a1a1a;
        margin: 0;
    }
    
    .header-subtitle {
        font-size: 0.95rem;
        color: #666;
        margin-top: 0.25rem;
    }
    
    /* Card styling */
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    /* Metric cards */
    .metric-container {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
    }
    
    /* Text area styling */
    .stTextArea > div > div > textarea {
        border-radius: 6px;
        border: 1px solid #d0d0d0;
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 6px;
        font-weight: 500;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    
    /* Chart container */
    .chart-container {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# SVG Logo (inline, subtle and professional)
LOGO_SVG = """
<svg width="32" height="32" viewBox="0 0 32 32" xmlns="http://www.w3.org/2000/svg">
    <rect width="32" height="32" rx="4" fill="#4a90e2" opacity="0.1"/>
    <path d="M8 12h16M8 16h16M8 20h12" stroke="#4a90e2" stroke-width="2" stroke-linecap="round"/>
    <circle cx="24" cy="12" r="2" fill="#4a90e2"/>
</svg>
"""

# Decorative SVG for charts
CHART_DECORATION_SVG = """
<svg width="20" height="20" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg">
    <rect x="2" y="14" width="3" height="4" fill="#4a90e2" opacity="0.6"/>
    <rect x="6" y="10" width="3" height="8" fill="#4a90e2" opacity="0.6"/>
    <rect x="10" y="6" width="3" height="12" fill="#4a90e2" opacity="0.6"/>
    <rect x="14" y="8" width="3" height="10" fill="#4a90e2" opacity="0.6"/>
</svg>
"""


@st.cache_resource
def load_models() -> Tuple[TfidfVectorizer, LogisticRegression]:
    """
    Load the vectorizer and model from disk using joblib.
    
    Returns:
        Tuple containing (vectorizer, model)
        
    Raises:
        FileNotFoundError: If model files are not found
    """
    try:
        vectorizer = joblib.load("vectorizer.jb")
        model = joblib.load("lr_model.jb")
        return vectorizer, model
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()


@st.cache_data
def load_dataset(filename: str) -> pd.DataFrame:
    """
    Load and clean a dataset CSV file.
    
    Args:
        filename: Name of the CSV file to load
        
    Returns:
        Cleaned DataFrame with normalized column names
    """
    try:
        df = pd.read_csv(filename)
        
        # Normalize column names to lowercase
        df.columns = df.columns.str.lower().str.strip()
        
        # Try to find text column (could be 'text', 'article', 'content', etc.)
        text_col = None
        for col in ['text', 'article', 'content', 'title']:
            if col in df.columns:
                text_col = col
                break
        
        # If no standard text column found, use first column
        if text_col is None and len(df.columns) > 0:
            text_col = df.columns[0]
        
        # Ensure we have a text column
        if text_col is None:
            st.error(f"No text column found in {filename}")
            return pd.DataFrame()
        
        # Clean the text column
        df[text_col] = df[text_col].astype(str)
        df = df[df[text_col].str.len() > 0]  # Remove empty texts
        
        return df
    except FileNotFoundError:
        st.error(f"Dataset file not found: {filename}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading dataset {filename}: {e}")
        return pd.DataFrame()


def clean_text(text: str, max_length: int = 10000) -> str:
    """
    Clean and normalize text input.
    
    Args:
        text: Raw text input
        max_length: Maximum length to keep (truncate if longer)
        
    Returns:
        Cleaned text string
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Basic cleaning
    text = text.strip()
    
    # Truncate if too long
    if len(text) > max_length:
        text = text[:max_length]
    
    return text


def predict_texts(
    texts: List[str], 
    vectorizer: TfidfVectorizer, 
    model: LogisticRegression
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict labels and probabilities for a list of texts.
    
    Args:
        texts: List of text strings to predict
        vectorizer: Fitted TF-IDF vectorizer
        model: Trained logistic regression model
        
    Returns:
        Tuple of (predictions, probabilities)
        - predictions: Array of predicted labels (0=Fake, 1=True)
        - probabilities: Array of probability arrays [prob_fake, prob_true]
    """
    # Clean texts
    cleaned_texts = [clean_text(text) for text in texts]
    
    # Transform texts
    X = vectorizer.transform(cleaned_texts)
    
    # Handle sparse matrices
    if hasattr(X, 'toarray'):
        # For single prediction, keep sparse for efficiency
        if len(cleaned_texts) == 1:
            predictions = model.predict(X)
            probabilities = model.predict_proba(X)
        else:
            # For batch, convert to dense if needed
            predictions = model.predict(X)
            probabilities = model.predict_proba(X)
    else:
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
    
    return predictions, probabilities


def top_contributing_words(
    text: str,
    vectorizer: TfidfVectorizer,
    model: LogisticRegression,
    top_n: int = 8
) -> List[Tuple[str, float, str]]:
    """
    Find the top N words that most contribute to the prediction.
    
    Args:
        text: Input text to analyze
        vectorizer: Fitted TF-IDF vectorizer
        model: Trained logistic regression model
        top_n: Number of top words to return
        
    Returns:
        List of tuples (word, contribution_score, direction)
        where direction is "Fake" or "True"
    """
    # Clean and transform text
    cleaned_text = clean_text(text)
    X = vectorizer.transform([cleaned_text])
    
    # Get feature names (vocabulary)
    feature_names = vectorizer.get_feature_names_out()
    
    # Get coefficients (assuming binary classification)
    # For binary LR, we typically have one coefficient array
    if hasattr(model, 'coef_'):
        coef = model.coef_[0]  # Get coefficients for positive class
    else:
        return []
    
    # Get non-zero features for this text
    if hasattr(X, 'toarray'):
        X_dense = X.toarray()[0]
    else:
        X_dense = X[0]
    
    # Calculate contribution: TF-IDF value * coefficient
    contributions = X_dense * coef
    
    # Get indices of non-zero features
    nonzero_indices = np.nonzero(X_dense)[0]
    
    # Get contributions for non-zero features
    word_contributions = [
        (feature_names[i], contributions[i], "True" if contributions[i] > 0 else "Fake")
        for i in nonzero_indices
    ]
    
    # Sort by absolute contribution value
    word_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    
    # Return top N
    return word_contributions[:top_n]


def generate_sample_csv(df: pd.DataFrame) -> str:
    """
    Generate CSV string from DataFrame for download.
    
    Args:
        df: DataFrame to convert
        
    Returns:
        CSV string
    """
    output = io.StringIO()
    df.to_csv(output, index=False)
    return output.getvalue()


def compute_word_frequencies(df: pd.DataFrame, text_col: str, top_n: int = 15) -> Dict[str, int]:
    """
    Compute word frequencies from dataset.
    
    Args:
        df: DataFrame with text data
        text_col: Name of text column
        top_n: Number of top words to return
        
    Returns:
        Dictionary of word -> frequency
    """
    # Combine all texts
    all_text = ' '.join(df[text_col].astype(str).tolist())
    
    # Basic tokenization (split on whitespace, remove punctuation)
    words = re.findall(r'\b[a-zA-Z]{3,}\b', all_text.lower())
    
    # Count frequencies
    word_counts = Counter(words)
    
    # Get top N
    top_words = dict(word_counts.most_common(top_n))
    
    return top_words


# Load models (cached)
try:
    vectorizer, model = load_models()
except Exception as e:
    st.error(f"Failed to load models: {e}")
    st.stop()


# Header with logo
st.markdown(f"""
<div class="header-container">
    <div class="header-logo">{LOGO_SVG}</div>
    <div>
        <h1 class="header-title">Fake News Detector</h1>
        <p class="header-subtitle">AI-powered news authenticity analysis</p>
    </div>
</div>
""", unsafe_allow_html=True)


# Sidebar controls
with st.sidebar:
    st.markdown("### Dataset Controls")
    
    # Dataset selector
    dataset_choice = st.selectbox(
        "Select Dataset",
        ["True.csv", "Fake.csv"],
        help="Choose which dataset to explore"
    )
    
    # Sample rows slider
    sample_rows = st.slider(
        "Preview Rows",
        min_value=5,
        max_value=100,
        value=20,
        step=5,
        help="Number of rows to preview in the Explore tab"
    )
    
    st.markdown("---")
    
    # Download sample option
    st.markdown("### Download Sample")
    if st.button("Download Current Preview"):
        df_preview = load_dataset(dataset_choice)
        if not df_preview.empty:
            # Get text column name
            text_col = None
            for col in ['text', 'article', 'content', 'title']:
                if col in df_preview.columns:
                    text_col = col
                    break
            if text_col is None and len(df_preview.columns) > 0:
                text_col = df_preview.columns[0]
            
            # Create preview
            preview_df = df_preview.head(sample_rows)
            csv_data = generate_sample_csv(preview_df)
            
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name=f"{dataset_choice.replace('.csv', '')}_preview.csv",
                mime="text/csv"
            )
        else:
            st.warning("No data available to download")
    
    st.markdown("---")
    
    # Portfolio section
    st.markdown("### Portfolio")
    st.markdown(
        "<div style='text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 6px; margin: 1rem 0;'>"
        "<p style='margin: 0 0 0.5rem 0; color: #666; font-size: 0.9rem;'>Developed by</p>"
        "<a href='https://soulef-bentorki.vercel.app' target='_blank' "
        "style='color: #4a90e2; text-decoration: none; font-weight: 600; font-size: 1rem;'>"
        "Soulef Bentorki</a><br>"
        "<a href='https://soulef-bentorki.vercel.app' target='_blank' "
        "style='color: #4a90e2; text-decoration: none; font-size: 0.85rem; margin-top: 0.5rem; display: inline-block;'>"
        "View Portfolio →</a>"
        "</div>",
        unsafe_allow_html=True
    )


# Main content tabs
tab1, tab2 = st.tabs(["Predict", "Explore"])


# Predict Tab
with tab1:
    st.markdown("### Text Input")
    
    # Text area for single article
    article_text = st.text_area(
        "Paste article text here:",
        height=200,
        placeholder="Enter or paste the news article text you want to analyze...",
        help="Paste a single news article for prediction"
    )
    
    st.markdown("---")
    st.markdown("### Batch Prediction (Optional)")
    
    # File upload for batch prediction
    uploaded_file = st.file_uploader(
        "Upload CSV file for batch prediction",
        type=['csv'],
        help="CSV file should have a 'text' column or a single text column"
    )
    
    # Predict button
    predict_button = st.button("Predict", type="primary", use_container_width=True)
    
    # Prediction logic
    if predict_button:
        if article_text.strip() or uploaded_file is not None:
            # Single text prediction
            if article_text.strip() and uploaded_file is None:
                with st.spinner("Analyzing article..."):
                    progress_bar = st.progress(0)
                    
                    # Predict
                    predictions, probabilities = predict_texts([article_text], vectorizer, model)
                    progress_bar.progress(100)
                    
                    # Get result
                    pred_label = "True" if predictions[0] == 1 else "Fake"
                    prob = probabilities[0]
                    confidence = prob[1] if predictions[0] == 1 else prob[0]
                    confidence_pct = confidence * 100
                    
                    # Display result
                    st.markdown("---")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if pred_label == "True":
                            st.success(f"**Predicted: {pred_label}**")
                        else:
                            st.error(f"**Predicted: {pred_label}**")
                    
                    with col2:
                        st.metric("Confidence", f"{confidence_pct:.1f}%")
                    
                    # Top contributing words
                    st.markdown("### Top Contributing Words")
                    contributing_words = top_contributing_words(article_text, vectorizer, model, top_n=8)
                    
                    if contributing_words:
                        word_data = []
                        for word, score, direction in contributing_words:
                            word_data.append({
                                "Word": word,
                                "Contribution": f"{abs(score):.4f}",
                                "Direction": direction
                            })
                        
                        word_df = pd.DataFrame(word_data)
                        st.dataframe(word_df, use_container_width=True, hide_index=True)
                    else:
                        st.info("Could not extract contributing words")
            
            # Batch prediction
            elif uploaded_file is not None:
                try:
                    # Read uploaded CSV
                    batch_df = pd.read_csv(uploaded_file)
                    
                    # Find text column
                    text_col = None
                    for col in ['text', 'article', 'content', 'title']:
                        if col.lower() in batch_df.columns.str.lower():
                            text_col = col
                            break
                    
                    if text_col is None and len(batch_df.columns) == 1:
                        text_col = batch_df.columns[0]
                    
                    if text_col is None:
                        st.error("Could not find text column in uploaded CSV. Please ensure there's a 'text' column or a single text column.")
                    else:
                        # Check row count
                        num_rows = len(batch_df)
                        if num_rows > 500:
                            st.warning(f"Large file detected ({num_rows} rows). Processing may take time.")
                            if not st.session_state.get('confirmed_large_batch', False):
                                if st.button("Continue with full batch"):
                                    st.session_state.confirmed_large_batch = True
                                    st.rerun()
                                if st.button("Sample first 500 rows"):
                                    batch_df = batch_df.head(500)
                                    num_rows = 500
                                    st.session_state.confirmed_large_batch = False
                        else:
                            st.session_state.confirmed_large_batch = False
                        
                        # Get texts
                        texts = batch_df[text_col].astype(str).tolist()
                        
                        # Predict
                        with st.spinner(f"Processing {len(texts)} articles..."):
                            progress_bar = st.progress(0)
                            
                            predictions, probabilities = predict_texts(texts, vectorizer, model)
                            
                            progress_bar.progress(100)
                        
                        # Create results DataFrame
                        results_df = batch_df.copy()
                        results_df['predicted_label'] = ['True' if p == 1 else 'Fake' for p in predictions]
                        results_df['probability'] = [prob[1] if p == 1 else prob[0] for p, prob in zip(predictions, probabilities)]
                        results_df['probability_pct'] = results_df['probability'] * 100
                        
                        # Display summary
                        st.markdown("### Batch Prediction Results")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Total Articles", len(results_df))
                        with col2:
                            true_count = (results_df['predicted_label'] == 'True').sum()
                            st.metric("Predicted True", true_count)
                        with col3:
                            fake_count = (results_df['predicted_label'] == 'Fake').sum()
                            st.metric("Predicted Fake", fake_count)
                        
                        # Display results table
                        st.markdown("### Results Preview")
                        display_cols = [text_col, 'predicted_label', 'probability_pct']
                        st.dataframe(
                            results_df[display_cols].head(20),
                            use_container_width=True
                        )
                        
                        # Download button
                        csv_output = generate_sample_csv(results_df)
                        st.download_button(
                            label="Download Full Results (CSV)",
                            data=csv_output,
                            file_name="batch_predictions.csv",
                            mime="text/csv"
                        )
                
                except Exception as e:
                    st.error(f"Error processing batch file: {e}")
            
            # Both provided - prioritize batch
            elif article_text.strip() and uploaded_file is not None:
                st.info("Both text input and file uploaded. Processing batch file. Clear text area if you want single prediction.")
        
        else:
            st.warning("Please enter article text or upload a CSV file for prediction.")


# Explore Tab
with tab2:
    st.markdown("### Dataset Preview")
    
    # Load selected dataset
    df = load_dataset(dataset_choice)
    
    if not df.empty:
        # Find text column
        text_col = None
        for col in ['text', 'article', 'content', 'title']:
            if col in df.columns:
                text_col = col
                break
        if text_col is None and len(df.columns) > 0:
            text_col = df.columns[0]
        
        if text_col:
            # Display preview
            st.markdown(f"**Dataset: {dataset_choice}** ({len(df)} rows)")
            preview_df = df.head(sample_rows)
            st.dataframe(preview_df, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            
            # Class distribution (for datasets that might have labels)
            st.markdown("### Dataset Statistics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"{CHART_DECORATION_SVG} **Dataset Info**", unsafe_allow_html=True)
                st.metric("Total Rows", len(df))
                st.metric("Columns", len(df.columns))
                
                # Check if there's a label column
                label_col = None
                for col in ['label', 'class', 'category', 'type']:
                    if col in df.columns:
                        label_col = col
                        break
                
                if label_col:
                    st.metric("Classes", df[label_col].nunique())
            
            with col2:
                # Word count statistics
                word_counts = df[text_col].astype(str).str.split().str.len()
                st.markdown(f"{CHART_DECORATION_SVG} **Text Statistics**", unsafe_allow_html=True)
                st.metric("Avg Words/Article", f"{word_counts.mean():.0f}")
                st.metric("Max Words", word_counts.max())
                st.metric("Min Words", word_counts.min())
            
            st.markdown("---")
            
            # Class distribution chart (if label column exists)
            if label_col:
                st.markdown("### Class Distribution")
                class_counts = df[label_col].value_counts()
                
                fig, ax = plt.subplots(figsize=(8, 5))
                bars = ax.bar(class_counts.index.astype(str), class_counts.values, color=['#4a90e2', '#e24a4a'])
                ax.set_xlabel('Class')
                ax.set_ylabel('Count')
                ax.set_title('Distribution of Classes in Dataset')
                plt.xticks(rotation=45)
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height)}',
                           ha='center', va='bottom')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            
            st.markdown("---")
            
            # Word frequency chart
            st.markdown("### Top Word Frequencies")
            
            with st.spinner("Computing word frequencies..."):
                word_freqs = compute_word_frequencies(df, text_col, top_n=15)
            
            if word_freqs:
                words = list(word_freqs.keys())
                freqs = list(word_freqs.values())
                
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.barh(words, freqs, color='#4a90e2', alpha=0.7)
                ax.set_xlabel('Frequency')
                ax.set_ylabel('Words')
                ax.set_title('Top 15 Most Frequent Words')
                ax.invert_yaxis()  # Highest frequency at top
                
                # Add value labels
                for i, (word, freq) in enumerate(zip(words, freqs)):
                    ax.text(freq + max(freqs) * 0.01, i, str(freq),
                           va='center', fontsize=9)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            else:
                st.info("Could not compute word frequencies")
    
    else:
        st.warning(f"Could not load dataset: {dataset_choice}")


# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #999; padding: 1rem; font-size: 0.85rem;'>"
    "Fake News Detector | Powered by Machine Learning & Python<br>"
    "Developed by <a href='https://soulef-bentorki.vercel.app' target='_blank' style='color: #4a90e2; text-decoration: none; font-weight: 500;'>Soulef Bentorki</a> | "
    "<a href='https://soulef-bentorki.vercel.app' target='_blank' style='color: #4a90e2; text-decoration: none;'>View Portfolio</a>"
    "</div>",
    unsafe_allow_html=True
)
