import os
import streamlit as st
import torch
from utils import (
    load_sentiment_model,
    get_esg_data,
    get_stock_data,
    get_company_name,
    get_news_sentiment,
    display_metrics,
    plot_stock_performance,
    display_news_sentiment
)

# Render-specific configuration
IS_RENDER = os.environ.get('RENDER', False)
PORT = int(os.environ.get("PORT", 8501))

# Constants
DEFAULT_TICKER = "AAPL"
DEFAULT_DAYS = 365
MAX_NEWS_ARTICLES = 10 if IS_RENDER else 20

# Configure PyTorch for Render's limited resources
torch.set_num_threads(1)

def main():
    st.set_page_config(
        page_title="ESG Dashboard",
        page_icon="🌱",
        layout="centered" if IS_RENDER else "wide"
    )

    # Sidebar controls
    with st.sidebar:
        st.header("Controls")
        ticker = st.text_input("Stock Ticker", DEFAULT_TICKER).upper()
        days = st.slider("Analysis Period (Days)", 30, 730, DEFAULT_DAYS)
        st.markdown("---")
        st.markdown("**Optimized for Render**" if IS_RENDER else "**Local Development**")

    # Load model
    with st.spinner("Loading AI model (first time may take ~60s)..."):
        model_info = load_sentiment_model()

    # Data loading
    esg_scores = get_esg_data(ticker)
    stock_data = get_stock_data(ticker, days)
    
    # News analysis
    news_df, avg_sentiment = None, 0
    if st.sidebar.checkbox("Show News Analysis", True):
        news_df, avg_sentiment = get_news_sentiment(
            ticker, 
            get_company_name(ticker), 
            model_info,
            max_articles=MAX_NEWS_ARTICLES
        )

    # Main dashboard
    st.title(f"🌱 ESG Analytics - {get_company_name(ticker)} ({ticker})")
    
    if esg_scores:
        display_metrics(esg_scores, avg_sentiment)
    else:
        st.warning("No ESG data available for this company")

    tab1, tab2 = st.tabs(["Financial Performance", "News Analysis"])
    with tab1:
        plot_stock_performance(stock_data)
    with tab2:
        if news_df is not None:
            display_news_sentiment(news_df)
        else:
            st.info("Enable news analysis in sidebar")

if __name__ == "__main__":
    main()
