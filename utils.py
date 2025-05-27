import streamlit as st  
import yfinance as yf
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
from newsapi import NewsApiClient
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

# Rest of your utils.py code remains the same...
# Sentiment model cache
@st.cache_resource
def load_sentiment_model():
    try:
        model_name = "finiteautomata/bertweet-base-sentiment-analysis"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        return (model, tokenizer)
    except Exception as e:
        st.error(f"Model error: {str(e)}")
        return None

def get_esg_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        esg_data = stock.sustainability
        return {
            'environment': esg_data.loc.get('environmentScore', {}).get('Value', 0) / 100,
            'social': esg_data.loc.get('socialScore', {}).get('Value', 0) / 100,
            'governance': esg_data.loc.get('governanceScore', {}).get('Value', 0) / 100
        } if esg_data is not None else None
    except:
        return None

def get_stock_data(ticker, days):
    try:
        data = yf.download(
            ticker,
            start=(datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d'),
            end=datetime.now().strftime('%Y-%m-%d')
        )
        return data[['Close']].reset_index()
    except:
        return None

def get_news_sentiment(ticker, company_name, model_info, max_articles=10):
    try:
        newsapi = NewsApiClient(api_key=os.environ["NEWS_API_KEY"])
        news = newsapi.get_everything(
            q=f"{company_name} OR {ticker}",
            page_size=max_articles,
            language='en'
        )
        
        headlines = [a['title'] for a in news.get('articles', [])]
        if not headlines:
            return pd.DataFrame(), 0
            
        results = analyze_sentiment(headlines, model_info)
        
        news_df = pd.DataFrame({
            'Date': [a['publishedAt'] for a in news['articles']],
            'Source': [a['source']['name'] for a in news['articles']],
            'Headline': headlines,
            'Sentiment': [r['label'].lower() for r in results],
            'Score': [r['score'] for r in results]
        })
        
        avg_sentiment = news_df['Score'].mean()
        return news_df, avg_sentiment
    except Exception as e:
        st.error(f"News error: {str(e)}")
        return pd.DataFrame(), 0 
