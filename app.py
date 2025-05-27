import os
import torch
import streamlit as st
from utils import (
    load_sentiment_model,
    get_esg_data,
    get_stock_data,
    analyze_sentiment,
    display_metrics,
    plot_stock_performance,
    display_news_sentiment
)

# Render-specific configuration
IS_RENDER = 'RENDER' in os.environ
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
        layout="centered" if IS_RENDER else "wide"
    )

    # Sidebar controls
    ticker = st.sidebar.text_input("Stock Ticker", DEFAULT_TICKER).upper()
    days = st.sidebar.slider("Analysis Period (Days)", 30, 730, DEFAULT_DAYS)

    # Load model
    with st.spinner("Loading AI model..."):
        model_info = load_sentiment_model()

    # Data loading
    esg_scores = get_esg_data(ticker)
    stock_data = get_stock_data(ticker, days)
    
    # News analysis
    news_df, avg_sentiment = None, 0
    if st.sidebar.checkbox("Show News Analysis"):
        news_df, avg_sentiment = get_news_sentiment(
            ticker, 
            get_company_name(ticker), 
            model_info,
            max_articles=MAX_NEWS_ARTICLES
        )

    # Display dashboard
    st.title(f"🌱 ESG Dashboard - {ticker}")
    if esg_scores:
        display_metrics(esg_scores, avg_sentiment)
    
    tab1, tab2 = st.tabs(["Financial Data", "News Analysis"])
    with tab1:
        plot_stock_performance(stock_data)
    with tab2:
        if news_df is not None:
            display_news_sentiment(news_df)

if __name__ == "__main__":
    main() 
