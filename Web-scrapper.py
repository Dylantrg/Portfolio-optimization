import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

def get_stock_data(ticker, start_date, end_date, interval='1d'):
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date, interval=interval)
    return data

def calculate_technical_indicators(df):
    # Simple Moving Average (SMA)
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    
    # Exponential Moving Average (EMA)
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    
    # Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Daily Return
    df['Daily_Return'] = df['Close'].pct_change()
    
    # Daily Amplitude (DA)
    df['Daily_Amplitude'] = (df['High'] - df['Low']) / df['Low']
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    df['BB_Upper'] = df['BB_Middle'] + 2 * df['Close'].rolling(window=20).std()
    df['BB_Lower'] = df['BB_Middle'] - 2 * df['Close'].rolling(window=20).std()
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    return df

def get_financial_metrics(ticker):
    stock = yf.Ticker(ticker)
    
    # Get financial data
    balance_sheet = stock.balance_sheet
    income_stmt = stock.income_stmt
    cash_flow = stock.cashflow
    
    # Calculate metrics
    total_assets = balance_sheet.loc['Total Assets'].iloc[0]
    net_income = income_stmt.loc['Net Income'].iloc[0]
    total_debt = balance_sheet.loc['Total Debt'].iloc[0]
    total_equity = balance_sheet.loc['Total Stockholder Equity'].iloc[0]
    revenue = income_stmt.loc['Total Revenue'].iloc[0]
    
    metrics = {
        'Return_on_Asset': net_income / total_assets,
        'Debt_Asset_Ratio': total_debt / total_assets,
        'Earnings_Per_Share': stock.info.get('trailingEps', None),
        'Total_Assets_Growth_Rate': (balance_sheet.loc['Total Assets'].iloc[0] / balance_sheet.loc['Total Assets'].iloc[-1]) - 1,
        'Operational_Profit': income_stmt.loc['Operating Income'].iloc[0],
        'Revenue_Growth_Rate': (income_stmt.loc['Total Revenue'].iloc[0] / income_stmt.loc['Total Revenue'].iloc[-1]) - 1,
        'Asset_Turnover_Rate': revenue / total_assets,
        'Gross_Profit_Growth_Rate': (income_stmt.loc['Gross Profit'].iloc[0] / income_stmt.loc['Gross Profit'].iloc[-1]) - 1,
        'Price_to_Earnings_Ratio': stock.info.get('trailingPE', None),
        'Price_to_Book_Ratio': stock.info.get('priceToBook', None),
        'Price_to_Sales_Ratio': stock.info.get('priceToSalesTrailing12Months', None),
    }
    
    return pd.Series(metrics)

# Set date range
start_date = "2014-01-01"
end_date = datetime.now().strftime("%Y-%m-%d")

# Get Apple stock data
apple_data = get_stock_data("AAPL", start_date, end_date)

# Calculate technical indicators
apple_data = calculate_technical_indicators(apple_data)

# Get financial metrics
apple_metrics = get_financial_metrics("AAPL")

# Save data to CSV files
apple_data.to_csv("apple_stock_data_with_indicators.csv")
apple_metrics.to_csv("apple_financial_metrics.csv")

print("Data extraction and calculation complete. Files saved as CSV.")

print("\nApple Stock Data with Indicators (first 5 rows):")
print(apple_data.head())

print("\nApple Financial Metrics:")
print(apple_metrics)
