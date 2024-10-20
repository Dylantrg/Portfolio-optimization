import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import requests
from datetime import datetime, timedelta

def get_recent_news_data(ticker, api_key, days=30):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    url = f"https://newsapi.org/v2/everything?q={ticker}&from={start_date.strftime('%Y-%m-%d')}&to={end_date.strftime('%Y-%m-%d')}&sortBy=publishedAt&apiKey={api_key}"
    
    response = requests.get(url)
    if response.status_code == 200:
        news_data = response.json()
        articles = news_data.get('articles', [])
        return pd.DataFrame(articles)
    else:
        print(f"Error fetching news data: {response.status_code}")
        return pd.DataFrame()

def get_historical_news_data(ticker, start_date, end_date):
    # This is a placeholder function. In a real scenario, you'd need to implement
    # a method to fetch historical news data from an alternative source.
    print(f"Fetching historical news data for {ticker} from {start_date} to {end_date}")
    # Return an empty DataFrame for now
    return pd.DataFrame(columns=['title', 'description', 'publishedAt'])

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and numbers
    text = ''.join([char for char in text if char.isalpha() or char.isspace()])
    return text

def train_sentiment_model(texts, labels, max_words=10000, max_len=200):
    # Tokenize the text
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    
    # Pad sequences
    X = pad_sequences(sequences, maxlen=max_len)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
    
    # Create the model
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(max_words, 16, input_length=max_len),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train the model
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), verbose=1)
    
    return model, tokenizer

if __name__ == "__main__":
    API_KEY = "c82d48e5444241cbac7bb12583dcb114"
    TICKER = "AAPL"
    START_DATE = "2014-01-01"
    END_DATE = datetime.now().strftime("%Y-%m-%d")
    
    # Fetch recent news data (last 30 days)
    recent_news_df = get_recent_news_data(TICKER, API_KEY)
    
    # Fetch historical news data
    historical_news_df = get_historical_news_data(TICKER, START_DATE, END_DATE)
    
    # Combine recent and historical news data
    news_df = pd.concat([historical_news_df, recent_news_df], ignore_index=True)
    
    if not news_df.empty:
        # Preprocess the text
        news_df['processed_text'] = news_df['description'].fillna('').apply(preprocess_text)
        
        # For demonstration, we'll create dummy sentiment labels
        # In a real scenario, you'd need labeled data or use a pre-trained model
        news_df['sentiment'] = np.random.choice([0, 1], size=len(news_df))
        
        # Train the model
        model, tokenizer = train_sentiment_model(news_df['processed_text'], news_df['sentiment'])
        
        # Save the processed news data
        news_df.to_csv("processed_news_data.csv", index=False)
        
        print("News data processed and saved. Model trained on dummy sentiment data.")
        print("\nSample of processed news data:")
        print(news_df[['title', 'processed_text', 'sentiment']].head())
    else:
        print("No news data to process.")
