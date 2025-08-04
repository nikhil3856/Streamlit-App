import streamlit as st
import requests
from streamlit_lottie import st_lottie

# Page configuration
st.set_page_config(page_title="Sentiment Framework", layout="wide")

# Load animation from Lottie URL
def load_lottie_url(url):
    try:
        r = requests.get(url)
        if r.status_code == 200:
            return r.json()
    except:
        return None

animation = load_lottie_url("https://assets6.lottiefiles.com/packages/lf20_qp1q7mct.json")

# Sidebar navigation
st.sidebar.title("🧭 Navigation")
st.sidebar.page_link("Pages/2_Analysis.py", label="⚙️ Analyze Data")
st.sidebar.page_link("Pages/3_Report.py", label="📊 Report")

# Main title
st.markdown("<h1 style='color:#1f77b4;'>🧠 Sentiment Analysis Framework</h1>", unsafe_allow_html=True)

# Animation and description layout
col1, col2 = st.columns([2, 3])
with col1:
    if animation:
        st_lottie(animation, speed=1, height=250, key="welcome")
    else:
        st.warning("⚠️ Could not load animation.")

with col2:
    st.markdown("""
    <div style='font-size:16px;'>
        Welcome to <strong>Sentiment.app</strong>, your NLP-powered solution for breaking down and analyzing textual reviews.  
        <br><br>
        <strong>🔍 Features:</strong><br>
        • Aspect-level sentiment classification (Positive / Negative / Neutral)<br>
        • Word frequency and topic analysis<br>
        • Final review sentiment + insights & reports<br><br>

        <span style='color:#888;'>Use the sidebar to move between pages</span>
    </div>
    """, unsafe_allow_html=True)

# Horizontal divider
st.markdown("---")

# Technical how-to section
with st.expander("🔧 How This App Works"):
    st.markdown("""
    **Workflow Steps:**
    1. **Load** your dataset (CSV/Excel) from inside the Analyze Data page
    2. **Select** relevant text columns
    3. **Process** reviews using a BERT-based NLP model
    4. **Extract** aspects and their sentiments
    5. **Visualize** using plots & download results

    All processing is local and secure.
    """)

# Proceed tab at bottom
st.markdown("---")
st.markdown("### 👉 Ready to Begin?")
if st.button("🚀 Proceed to Analyze", use_container_width=True):
    st.switch_page("Pages/2_Analysis.py")
