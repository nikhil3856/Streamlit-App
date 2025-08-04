import streamlit as st
import pandas as pd
import nltk
import plotly.express as px
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
import torch
import io
import time
from collections import Counter

@st.cache_resource
def download_nltk_resources():
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    return set(stopwords.words('english'))

STOP_WORDS = download_nltk_resources()

st.set_page_config(page_title="Data Processing & Insights", layout="wide")
st.title("Aspect-Based Feedback Analysis")

if "processed_data" not in st.session_state:
    st.session_state.processed_data = None
if "summary_data" not in st.session_state:
    st.session_state.summary_data = None

@st.cache_resource
def load_model():
    device = 0 if torch.cuda.is_available() else -1
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=device)
    return classifier

sentiment_model = load_model()

def map_sentiment(label):
    label = label.upper()
    if "LABEL_0" in label:
        return "Negative"
    elif "LABEL_1" in label:
        return "Neutral"
    elif "LABEL_2" in label:
        return "Positive"
    return "Neutral"

def sentiment_to_numerical(sentiment):
    return {"Positive": 1, "Neutral": 0, "Negative": -1}.get(sentiment, 0)

def numerical_to_sentiment(score):
    if score > 0.1:
        return "Positive"
    elif score < -0.1:
        return "Negative"
    else:
        return "Neutral"

def extract_aspects_from_sentence(text, stop_words):
    if not isinstance(text, str): return []
    tokens = word_tokenize(text.lower())
    tagged_words = pos_tag(tokens)
    potential_aspects = []
    current_aspect_phrase = []

    for word, tag in tagged_words:
        if tag.startswith('N') or tag.startswith('J'):
            if word.isalpha() and word not in stop_words:
                current_aspect_phrase.append(word)
            else:
                if current_aspect_phrase:
                    potential_aspects.append(" ".join(current_aspect_phrase).title())
                    current_aspect_phrase = []
        else:
            if current_aspect_phrase:
                potential_aspects.append(" ".join(current_aspect_phrase).title())
                current_aspect_phrase = []
    if current_aspect_phrase:
        potential_aspects.append(" ".join(current_aspect_phrase).title())

    filtered = [a for a in potential_aspects if len(a.split()) > 1 or len(a) > 2]
    if filtered:
        return sorted(Counter(filtered).keys(), key=lambda x: Counter(filtered)[x], reverse=True)[:2]
    return []

# --- Process Reviews with a Unified Progress Bar ---
def process_reviews(df, review_col, nps_col=None):
    all_sentences = []
    sentence_map = []
    total_rows = len(df)
    start_time = time.time()

    status_text = st.empty()
    progress_bar = st.progress(0)

    # Phase 1: Sentence Tokenization
    status_text.info(f"🔄 **Phase 1/2**: Tokenizing reviews into sentences... (0/{total_rows})")
    
    for idx, review in enumerate(df[review_col]):
        try:
            sentences = sent_tokenize(str(review))
        except:
            sentences = [str(review)]
        for sentence in sentences:
            all_sentences.append(sentence[:512])
            sentence_map.append((idx, review, sentence.strip()))
        
        progress = (idx + 1) / total_rows * 0.1 # This phase is a small part of the total work
        progress_bar.progress(progress)
        status_text.info(f"🔄 **Phase 1/2**: Tokenizing reviews into sentences... ({idx + 1}/{total_rows})")

    total_sentences = len(all_sentences)

    # Phase 2: Sentiment analysis
    status_text.info(f"🔍 **Phase 2/2**: Analyzing sentiment for {total_sentences} sentences...")

    sentiment_results = []
    BATCH_SIZE = 256
    
    for i in range(0, total_sentences, BATCH_SIZE):
        batch = all_sentences[i:i + BATCH_SIZE]
        sentiment_results.extend(sentiment_model(batch))
        
        # Unified progress update
        sentiment_progress = (i + len(batch)) / total_sentences
        total_progress = 0.1 + sentiment_progress * 0.9 # Sentiment analysis is 90% of the work
        progress_bar.progress(total_progress)
        
        elapsed_time = time.time() - start_time
        time_per_sentence = elapsed_time / (i + len(batch)) if i + len(batch) > 0 else 0
        est_remaining = (total_sentences - (i + len(batch))) * time_per_sentence
        
        def format_time(seconds):
            if seconds < 60:
                return f"{seconds:.0f}s"
            elif seconds < 3600:
                return f"{seconds / 60:.1f}m"
            else:
                return f"{seconds / 3600:.1f}h"
        
        status_text.markdown(f"""
        🔄 **Processing Data**  
        📥 Analyzed Sentences: **{i + len(batch)}/{total_sentences}**  
        ⏳ Estimated Remaining: **{format_time(est_remaining)}**  
        🕒 Total Estimated Time: **{format_time(elapsed_time + est_remaining)}**  
        """)

    progress_bar.empty()
    status_text.success(f"✅ Completed in **{time.time() - start_time:.1f} seconds**! Processed **{total_sentences}** sentences from **{total_rows}** reviews.")

    # --- Rest of the logic remains unchanged ---
    full_data = []
    summary_sentiment_map = {}

    for i, (review_idx, review_text, original_sentence) in enumerate(sentence_map):
        if i >= len(sentiment_results):
            continue
        pred = sentiment_results[i]
        label = map_sentiment(pred['label'])
        score = pred['score']
        aspects = extract_aspects_from_sentence(original_sentence, STOP_WORDS)

        if review_idx not in summary_sentiment_map:
            summary_sentiment_map[review_idx] = []

        if aspects:
            for asp in aspects:
                full_data.append({
                    "Review No.": review_idx + 1,
                    "Review Text": review_text,
                    "Aspect": asp,
                    "Aspect_Sentiment": label,
                    "Aspect_Sentiment_Score": score,
                    "Aspect_Context": original_sentence
                })
                summary_sentiment_map[review_idx].append({"sentiment": label, "score": score})
        else:
            summary_sentiment_map[review_idx].append({"sentiment": label, "score": score})

    summary_data = []
    for review_idx in range(total_rows):
        sentiments = summary_sentiment_map.get(review_idx, [])
        model_score = sum(sentiment_to_numerical(s['sentiment']) * s['score'] for s in sentiments) / len(sentiments) if sentiments else 0
        nps_score = None
        nps_val = 0
        if nps_col and nps_col in df.columns:
            try:
                nps_score = float(df.loc[review_idx, nps_col])
                if nps_score >= 9:
                    nps_val = 1
                elif nps_score <= 6:
                    nps_val = -1
            except:
                pass
        blended_score = (0.6 * model_score) + (0.4 * nps_val)
        final_sentiment = numerical_to_sentiment(blended_score)
        summary_data.append({
            "Review No.": review_idx + 1,
            "Review Text": df.loc[review_idx, review_col],
            "NPS Score": nps_score,
            "Final_Sentiment": final_sentiment
        })

    return pd.DataFrame(full_data), pd.DataFrame(summary_data)

# --- Streamlit UI Code (remains unchanged) ---
uploaded_file = st.file_uploader("Upload a CSV/Excel with reviews & NPS", type=["csv", "xlsx"])
if uploaded_file:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
    review_cols = [col for col in df.columns if df[col].dtype == object]
    selected_col = st.selectbox("Select review column", review_cols)
    nps_col = st.selectbox("Select NPS score column (optional)", [None] + list(df.columns))

    if st.button("Process Data"):
        start_time = time.time()
        processed_df, summary_df = process_reviews(df, selected_col, nps_col)
        st.session_state.processed_data = processed_df
        st.session_state.summary_data = summary_df

if st.session_state.summary_data is not None:
    st.subheader("Review Sentiment Summary")
    st.dataframe(st.session_state.summary_data)

    st.subheader("Sentiment Distribution")
    dist_df = st.session_state.summary_data["Final_Sentiment"].value_counts().reset_index()
    dist_df.columns = ["Sentiment", "Count"]
    st.plotly_chart(px.bar(dist_df, x="Sentiment", y="Count", color="Sentiment", text="Count"), use_container_width=True)

if st.session_state.processed_data is not None:
    st.subheader("Aspect-Level Breakdown")
    st.dataframe(st.session_state.processed_data)