import streamlit as st
import yfinance as yf
import pandas as pd
from finvader import finvader
import nltk

def analyze_sentiment_with_finvader(text):
    """Analyzes sentiment using the FinVADER library."""
    scores = finvader(
                    text,
                    use_sentibignomics=True, 
                    use_henry=True, 
                    indicator="compound"
                )
    return scores

def show_news_with_sentiment(ticker):
    """
    Fetches, analyzes, and displays a sentiment summary table at the top,
    followed by the detailed news articles.
    """
    st.header("4. Recent News & Sentiment Analysis")
    try:
        news = yf.Ticker(ticker).news
        if news:
            sentiment_data = []

            for item in news:
                content = item.get('content', {})
                title = content.get('title', 'No Title Available')
                summary = content.get('summary', 'No summary available.')
                provider = content.get('provider', {}).get('displayName', 'Unknown Source')
                link = content.get('canonicalUrl', {}).get('url', '#')

                scores = analyze_sentiment_with_finvader(title + " " + summary)

                sentiment_data.append({
                    'Title': title,
                    'Source': provider,
                    'Score': scores,
                    'Link': link
                })

            st.subheader("Sentiment Summary")
            df_sentiment = pd.DataFrame(sentiment_data)
            st.dataframe(df_sentiment)
            
            st.markdown("---")

            for item_data in sentiment_data:
                st.subheader(f"[{item_data['Title']}]({item_data['Link']})")
                st.write(f"Source: {item_data['Source']}")
                st.write(f"**Score: {item_data['Score']}**")
                
                original_summary = [item.get('content', {}).get('summary', '') for item in news if item.get('content', {}).get('title', '') == item_data['Title']]
                if original_summary:
                    st.write(f"Summary: {original_summary[0]}")
                st.markdown("---")
            
        else:
            st.info("No recent news found for this ticker.")
            
    except Exception as e:
        st.error(f"An error occurred while fetching news: {e}")

