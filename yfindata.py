import yfinance as yf
import pandas as pd

# Load the company names and ticker symbols from the CSV file
company_info_df = pd.read_csv('D:/QAT_stock/company_names_and_tickers.csv')

# Create an empty DataFrame to store the stock data
stock_data_df = pd.DataFrame()

# Function to compute RSI
def compute_RSI(data, window):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    RS = gain / loss
    RSI = 100 - (100 / (1 + RS))
    return RSI

# Fetch and store stock data for each company
for index, row in company_info_df.iterrows():
    company_name = row['Company Name']
    ticker_symbol = row['Ticker Symbol']
    try:
        # Fetch stock data using yfinance
        stock_data = yf.Ticker(ticker_symbol + '.QA')  # Adding '.QA' for Qatar Exchange
        stock_price_data = stock_data.history(start='2010-01-01', end='2024-12-31')  # Fetching data from 2010 to 2024
        
        # Fetch fundamental data
        info = stock_data.info
        fundamentals = {
            'Company Name': company_name,
            'Ticker Symbol': ticker_symbol,
            'Market Cap': info.get('marketCap', 'N/A'),
            'PE Ratio': info.get('trailingPE', 'N/A'),
            'EPS': info.get('trailingEps', 'N/A'),
            'PB Ratio': info.get('priceToBook', 'N/A'),
            'Dividend Yield': info.get('dividendYield', 'N/A'),
            'ROE': info.get('returnOnEquity', 'N/A'),
            'Debt-to-Equity': info.get('debtToEquity', 'N/A')
        }
        
        # Convert stock price data to DataFrame
        stock_price_data.reset_index(inplace=True)
        
        # Add the fundamental data to each row of the stock price data
        for key, value in fundamentals.items():
            stock_price_data[key] = value
        
        # Calculate technical indicators
        stock_price_data['50-day MA'] = stock_price_data['Close'].rolling(window=50).mean()
        stock_price_data['200-day MA'] = stock_price_data['Close'].rolling(window=200).mean()
        stock_price_data['RSI'] = compute_RSI(stock_price_data, 14)
        stock_price_data['MACD'] = stock_price_data['Close'].ewm(span=12, adjust=False).mean() - stock_price_data['Close'].ewm(span=26, adjust=False).mean()
        stock_price_data['Signal Line'] = stock_price_data['MACD'].ewm(span=9, adjust=False).mean()
        
        # Append the fetched data to the stock_data_df
        stock_data_df = pd.concat([stock_data_df, stock_price_data])
    except Exception as e:
        print(f"Error fetching data for {company_name} ({ticker_symbol}): {e}")

# Save the stock data to a CSV file
stock_data_df.to_csv('D:/QAT_stock/stock_data.csv', index=False)

print("Stock data has been successfully fetched and saved to stock_data.csv")
