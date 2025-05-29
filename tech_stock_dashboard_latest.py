
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import datetime
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import streamlit as st

nltk.data.path.append('./nltk_data')
try:
    _ = nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', download_dir='./nltk_data')

sia = SentimentIntensityAnalyzer()

# Streamlit page config
st.set_page_config(page_title="Tech Stock Sentiment Dashboard", layout="wide")
st.title("📈 Tech Stock Analysis with Sentiment & Machine Learning")

# Step 1: Select tickers and date range
stocks = st.multiselect("Select Tech Stocks", ['AAPL', 'MSFT', 'GOOGL', 'NVDA'], default=['AAPL', 'MSFT'])
date_range = st.date_input("Select Date Range", [datetime.date(2010, 1, 1), datetime.date.today()])

@st.cache_data
def fetch_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    df['Ticker'] = ticker
    return df

# Simulated News headlines placeholder
def get_mock_headlines(date, ticker):
    return [
        f"{ticker} stock rises on {date} with strong earnings",
        f"{ticker} product launch sparks optimism on {date}"
    ]

# Feature Engineering using 'Close' column instead of 'Adj Close'
def engineer_features(df, ticker):
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    df['MA_200'] = df['Close'].rolling(window=200).mean()
    df['Volatility'] = df['Close'].rolling(window=20).std()
    df['Return'] = df['Close'].pct_change()
    df['Future_Return'] = df['Close'].shift(-20) / df['Close'] - 1
    df['Target'] = (df['Future_Return'] > 0.05).astype(int)

    sentiments = []
    for date in df.index:
        date_str = date.strftime('%Y-%m-%d')
        headlines = get_mock_headlines(date_str, ticker)
        scores = [sia.polarity_scores(h)['compound'] for h in headlines]
        sentiment_avg = np.mean(scores) if scores else 0
        sentiments.append(sentiment_avg)
    df['Sentiment'] = sentiments
    return df.dropna(subset=['MA_20', 'MA_50', 'MA_200', 'Volatility', 'Return', 'Sentiment', 'Target'])

# Data aggregation and feature processing
feature_data = []
for ticker in stocks:
    start = pd.to_datetime(date_range[0]).strftime('%Y-%m-%d')
    end = pd.to_datetime(date_range[1]).strftime('%Y-%m-%d')
    raw_df = fetch_data(ticker, start, end)

    if isinstance(raw_df.columns, pd.MultiIndex):
    raw_df.columns = raw_df.columns.get_level_values(1)

    raw_df = raw_df.loc[:, ~raw_df.columns.duplicated()]
    raw_df = raw_df.rename(columns={ticker: 'Close'})  # ✅ Normalize for modeling

    st.write(f"Raw data for {ticker} from {start} to {end}:")
    st.dataframe(raw_df.head(10))  # Only show first 10 rows

    if raw_df.empty or 'Close' not in raw_df.columns:
        st.warning(f"No data available for {ticker}. Skipping...")
        continue

    processed_df = engineer_features(raw_df.copy(), ticker)
    feature_data.append(processed_df)

    if not feature_data:
    st.error("No valid stock data available for modeling. Try adjusting the date range or tickers.")
    st.stop()

data = pd.concat(feature_data)

# Modeling
features = ['MA_20', 'MA_50', 'MA_200', 'Volatility', 'Return', 'Sentiment']
X = data[features]
y = data['Target']
train_size = int(len(X) * 0.7)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
data['Prediction'] = model.predict(X)

# Results
st.write("### Model Accuracy", accuracy_score(y_test, predictions))
report = classification_report(y_test, predictions, output_dict=True)
st.write("### Classification Report")
st.dataframe(pd.DataFrame(report).transpose())

# Visualizations
for ticker in stocks:
    df = data[data['Ticker'] == ticker]
    with st.expander(f"📊 {ticker} Chart & Buy Signals"):
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(df.index, df['Close'], label='Close')
        buy_signals = df[df['Prediction'] == 1]
        ax.plot(buy_signals.index, buy_signals['Close'], '^', markersize=10, color='g', label='Buy Signal')
        ax.set_title(f"{ticker} with Buy Signals from Model (incl. Sentiment)")
        ax.legend()
        st.pyplot(fig)
