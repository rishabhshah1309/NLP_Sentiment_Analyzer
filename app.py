import streamlit as st
import pandas as pd
import os

st.set_page_config(layout="wide", page_title="Sponsorship Sentiment Dashboard")

st.title("📊 Sponsorship & Partnership Sentiment Dashboard")
st.markdown("Explore platform trends, sentiment, and keyword analysis for sponsorship posts.")

RESULTS_DIR = "results"  # or "." if everything is in current dir

# ---- Helper to show images (PNG dashboards) ----
def show_image(path, caption):
    if os.path.exists(path):
        st.image(path, caption=caption, use_column_width=True)
        with open(path, "rb") as img_file:
            st.download_button(f"Download {caption}", img_file, file_name=os.path.basename(path))
    else:
        st.warning(f"{caption} not found.")

# ---- Helper to show HTML reports inline ----
def show_html_report(path, caption):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            html = f.read()
            st.markdown(f"### {caption}", unsafe_allow_html=True)
            st.components.v1.html(html, height=800, scrolling=True)
        with open(path, "rb") as file:
            st.download_button(f"Download {caption}", file, file_name=os.path.basename(path))
    else:
        st.warning(f"{caption} not found.")

# ---- Show Dashboard Images ----
st.header("Executive Dashboards")
show_image("comprehensive_analysis_dashboard.png", "Comprehensive Analysis Dashboard")
show_image("platform_comparison_analysis.png", "Platform Comparison Analysis")
show_image("detailed_sentiment_analysis.png", "Detailed Sentiment Analysis")
show_image("keyword_sentiment_static.png", "Keyword Sentiment Static Visualization")
show_image("safe_analysis_dashboard.png", "Safe/Fallback Analysis Dashboard")

# ---- Show HTML Reports ----
st.header("Keyword & Phrase Sentiment Reports")
show_html_report("keyword_sentiment_report.html", "Keyword Sentiment Analysis Report")
show_html_report("keyword_sentiment_dashboard.html", "Interactive Keyword Dashboard")

# ---- Data Table Exploration ----
st.header("Explore Raw Sentiment Data")
csv_files = [
    "detailed_keyword_sentiments.csv",
    "phrase_sentiments.csv",
    "keyword_analysis_summary.csv"
]
selected_csv = st.selectbox("Select a data file to explore:", csv_files)
if os.path.exists(selected_csv):
    df = pd.read_csv(selected_csv)
    st.dataframe(df, use_container_width=True)
    st.download_button(
        label=f"Download {selected_csv}",
        data=df.to_csv(index=False),
        file_name=selected_csv,
        mime="text/csv"
    )
else:
    st.warning(f"{selected_csv} not found.")

# ---- Download All Results Section ----
st.header("Download All Results")
for file in [
    "comprehensive_analysis_dashboard.png",
    "platform_comparison_analysis.png",
    "detailed_sentiment_analysis.png",
    "keyword_sentiment_static.png",
    "safe_analysis_dashboard.png",
    "keyword_sentiment_report.html",
    "keyword_sentiment_dashboard.html",
    "detailed_keyword_sentiments.csv",
    "phrase_sentiments.csv",
    "keyword_analysis_summary.csv"
]:
    if os.path.exists(file):
        with open(file, "rb") as f:
            st.download_button(f"Download {file}", f, file_name=file)

