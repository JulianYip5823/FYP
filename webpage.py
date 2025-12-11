import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import hstack
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# SETUP
st.set_page_config(page_title="Airline Sentiment AI", layout="centered")

# Download VADER lexicon (Running this check prevents errors on first run)
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

# Initialize VADER
sid = SentimentIntensityAnalyzer()

# LOAD SAVED MODELS
@st.cache_resource
def load_models():
    # Load the Vectorizer
    with open('tfidf.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    
    # Load the Model 
    with open('sentiment_model.pkl', 'rb') as f:
        model = pickle.load(f)
        
    return tfidf, model

try:
    tfidf, model = load_models()
except FileNotFoundError:
    st.error("⚠️ Error: Pickle files not found. Please make sure 'tfidf.pkl' and 'sentiment_model.pkl' are in the same folder as this script.")
    st.stop()

# 3. WEB INTERFACE 
st.title("Airline Review Analyzer")
st.markdown("""
Type a customer review below to see if the AI predicts it as **Recommended** or **Not Recommended**.
The AI uses a hybrid of **Text Analysis (TF-IDF)** and **Sentiment Scoring (VADER)**.
""")

st.divider()

# Input Box
user_text = st.text_area("Enter Customer Review:", height=120, placeholder="Example: The flight was on time but the food was absolutely terrible.")

# PREDICTION LOGIC 
if st.button("Analyze Sentiment", type="primary"):
    if not user_text.strip():
        st.warning("Please enter some text to analyze.")
    else:
        # Transform Text to Numbers (TF-IDF)
        text_vectorized = tfidf.transform([user_text])
        
        # Calculate Sentiment Scores Live (VADER)
        scores = sid.polarity_scores(user_text)
        # We need 4 scores: neg, neu, pos, compound
        vader_vector = np.array([[scores['neg'], scores['neu'], scores['pos'], scores['compound']]])
        
        # Combine (Stacking)
        final_input = hstack([text_vectorized, vader_vector])
        
        # Predict
        prediction = model.predict(final_input)[0]
        
        #  5. DISPLAY RESULTS 
        st.subheader("Results")
        
        # Check prediction (Logic handles 'yes'/'1'/'recommended')
        if str(prediction).lower() in ['yes', 'recommended', '1', 'true']:
            st.success("## ✅ Prediction: RECOMMENDED")
        else:
            st.error("## ❌ Prediction: NOT RECOMMENDED")
            
        # 6. XAI (EXPLAINABLE AI) - FIXED GRAPH 
        st.divider()
        st.subheader("🔍 Why did the AI decide this?")
        st.write("These represent the **Top 10** most influential factors for this decision:")
        
        # A. Get Feature Names & Coefs
        feature_names = tfidf.get_feature_names_out()
        coefs = model.coef_[0]
        
        # B. Separate the coefficients
        text_coefs = coefs[:-4]
        vader_coefs = coefs[-4:]
        vader_names = ['Sentiment (Neg)', 'Sentiment (Neu)', 'Sentiment (Pos)', 'Sentiment (Compound)']
        
        # C. Find Impactful Words in User Input
        # Use a simpler split to catch basic words, removing punctuation
        input_words = user_text.lower().replace('.', ' ').replace(',', ' ').split()
        impact_data = []
        
        # Check Words
        unique_words = set(input_words)
        for word in unique_words:
            if word in tfidf.vocabulary_:
                idx = tfidf.vocabulary_[word]
                score = text_coefs[idx]
                impact_data.append({'Feature': f'"{word}"', 'Impact': score})
        
        # Check VADER Scores
        # We always add these to see if they mattered
        for name, score in zip(vader_names, vader_coefs):
             # Only show if they have non-zero weight
             if abs(score) > 0.01: 
                impact_data.append({'Feature': name, 'Impact': score})
        
        # D. Plot the Chart (FIXED LOGIC)
        if impact_data:
            df_impact = pd.DataFrame(impact_data)
            
            # --- THE FIX: SORT BY ABSOLUTE VALUE ---
            # 1. Calculate magnitude (absolute value)
            df_impact['abs_impact'] = df_impact['Impact'].abs()
            
            # 2. Sort by magnitude and take Top 10
            df_impact = df_impact.sort_values(by='abs_impact', ascending=False).head(10)
            
            # 3. Sort back by real value so the chart looks nice (Negative -> Positive)
            df_impact = df_impact.sort_values(by='Impact', ascending=True)
            
            # Plot
            fig, ax = plt.subplots(figsize=(8, 5))
            colors = ['#ff4b4b' if x < 0 else '#2bd666' for x in df_impact['Impact']]
            
            ax.barh(df_impact['Feature'], df_impact['Impact'], color=colors)
            ax.set_xlabel("Negative Influence <----------------> Positive Influence")
            ax.set_title("Top 10 Influential Features")
            ax.axvline(0, color='black', linestyle='--', linewidth=0.8)
            
            st.pyplot(fig)
        else:
            st.info("The AI made a decision based on small accumulated weights, but no single word stood out strongly.")