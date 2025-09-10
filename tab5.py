import streamlit as st
import yfinance as yf
import pandas as pd
from finvader import finvader
import tweepy

# ---------------- Sentiment Analysis ----------------
def analyze_sentiment_with_finvader(text):
    """Analyzes sentiment using the FinVADER library."""
    scores = finvader(
        text,
        use_sentibignomics=True, 
        use_henry=True, 
        indicator="compound"
    )
    return scores

# ---------------- Yahoo Finance ----------------
@st.cache_data(show_spinner=False)
def get_yahoo_news(ticker):
    news_data = []
    try:
        yf_news = yf.Ticker(ticker).news
        for item in yf_news:
            content = item.get('content', {})
            news_data.append({
                'title': content.get('title', 'No Title'),
                'summary': content.get('summary', ''),
                'link': content.get('canonicalUrl', {}).get('url', '#'),
                'source': content.get('provider', {}).get('displayName', 'Yahoo Finance')
            })
    except Exception as e:
        st.error(f"Error fetching Yahoo Finance news: {e}")
    return news_data

# ---------------- Twitter v2 ----------------
@st.cache_data(show_spinner=False)
def get_twitter_news(ticker, bearer_token):
    """Fetch recent tweets mentioning the stock ticker using free Twitter v2 API."""
    client = tweepy.Client(bearer_token=bearer_token)
    # Plain ticker, no $ or operators
    # query = f"{ticker.strip().lstrip('$')} -is:retweet"
    # print('query +++++++++++ '+query)
    articles = []

    try:
        tweets = client.search_recent_tweets(query="APPL")
        if tweets.data:
            for t in tweets.data:
                articles.append({
                    'title': t.text[:50] + "...",
                    'summary': t.text,
                    'link': f"https://twitter.com/i/web/status/{t.id}",
                    'source': 'Twitter'
                })
    except Exception as e:
        st.error(f"Error fetching Twitter data: {e}")

    return articles

# ---------------- Aggregate All News ----------------
@st.cache_data(show_spinner=False)
def get_all_news(ticker, twitter_bearer_token=None):
    news_data = []

    # Yahoo Finance
    news_data.extend(get_yahoo_news(ticker))

    # Twitter free account has 0 functionality, not able to retrieve any tweets.
    # Twitter
    # if twitter_bearer_token:
    #     news_data.extend(get_twitter_news(ticker, twitter_bearer_token))

    return news_data

# ---------------- Streamlit App ----------------
def show_news_with_sentiment(ticker):
    st.header(f"Recent News & Sentiment Analysis for {ticker}")
    
    news = get_all_news(ticker, "AAAAAAAAAAAAAAAAAAAAADsv4AEAAAAAs4N4GxqZq081XWQcWX6qiH2Ugns%3DcASnHizOm98cF12ZC8X0Kb3RDunYbXs2pZxN3Rfv5JDP18Ulhi")
    if not news:
        st.info("No news found for this tsicker.")
        return

    sentiment_data = []
    for item in news:
        text = item['title'] + " " + item['summary']
        score = analyze_sentiment_with_finvader(text)
        sentiment_data.append({
            'Title': item['title'],
            'Source': item['source'],
            'Score': score,
            'Link': item['link']
        })

    # DataFrame and Average
    df_sentiment = pd.DataFrame(sentiment_data)
    avg_score = df_sentiment['Score'].mean()
    st.markdown(f"**Average Sentiment Score:** {avg_score:.3f}")
    if avg_score > 0.05:
        st.success("Overall news sentiment is positive 👍")
    elif avg_score < -0.05:
        st.error("Overall news sentiment is negative 👎")
    else:
        st.info("Overall news sentiment is neutral 😐")
    
    st.info("0.1 → slightly positive, 0.3 → moderately positive, 0.5 → strongly positive, 0.8+ → very strongly positive")
    st.markdown("---")

    # Display each article
    for item_data in sentiment_data:
        st.subheader(f"[{item_data['Title']}]({item_data['Link']})")
        st.write(f"Source: {item_data['Source']}")
        st.write(f"**Score: {item_data['Score']}**")
        st.write(f"Summary: {news[sentiment_data.index(item_data)]['summary']}")
        st.markdown("---")
