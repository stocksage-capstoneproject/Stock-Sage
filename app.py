import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import yfinance as yf
import streamlit.components.v1 as components

# Static exchange rates (update these as needed)
STATIC_EXCHANGE_RATES = {
    'USD': 1.0,   # US Dollar
    'EUR': 0.93,  # Euro
    'GBP': 0.80,  # British Pound
    'INR': 82.00, # Indian Rupee
    'JPY': 135.00,# Japanese Yen
    'AUD': 1.50,  # Australian Dollar
    'CAD': 1.35,  # Canadian Dollar
    'CHF': 0.91,  # Swiss Franc
    'CNY': 7.00,  # Chinese Yuan
    'MXN': 18.00, # Mexican Peso
    'BRL': 5.00,  # Brazilian Real
    'NZD': 1.60,  # New Zealand Dollar
    'SGD': 1.35,  # Singapore Dollar
    'KRW': 1350.00,# South Korean Won
    'ZAR': 19.00, # South African Rand
    'HKD': 7.85,  # Hong Kong Dollar
    'SEK': 10.20, # Swedish Krona
    'NOK': 9.15,  # Norwegian Krone
    'DKK': 6.90,  # Danish Krone
    'ILS': 3.60,  # Israeli New Shekel
    'TRY': 27.00, # Turkish Lira
    'RUB': 80.00, # Russian Ruble
    'SAR': 3.75,  # Saudi Riyal
    'MYR': 4.70,  # Malaysian Ringgit
    'PHP': 56.00, # Philippine Peso
    'TWD': 30.00, # New Taiwan Dollar
    'PLN': 4.00,  # Polish Zloty
    'CZK': 22.50, # Czech Koruna
    'HUF': 320.00,# Hungarian Forint
    'CLP': 800.00,# Chilean Peso
    'COP': 4100.00,# Colombian Peso
    'PEN': 3.70,  # Peruvian Nuevo Sol
    'ARS': 380.00,# Argentine Peso
    'VEF': 25.00, # Venezuelan Bolívar
    'DOP': 56.00, # Dominican Peso
    'BHD': 0.38,  # Bahraini Dinar
    'KWD': 0.31,  # Kuwaiti Dinar
    'OMR': 0.39,  # Omani Rial
    'QAR': 3.64,  # Qatari Riyal
    'JOD': 0.71,  # Jordanian Dinar
    'RSD': 117.00,# Serbian Dinar
    'MAD': 10.30, # Moroccan Dirham
    'TND': 3.05,  # Tunisian Dinar
}

# Example list of popular tickers (including some Indian company tickers)
POPULAR_TICKERS = [
    # Global
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'BRK-B', 'JPM', 'V',
    'MA', 'WMT', 'DIS', 'HD', 'KO', 'PFE', 'MRK', 'BA', 'C', 'CSCO', 'NKE', 'UNH',
    'INTC', 'T', 'ORCL', 'IBM', 'ADBE', 'CVX', 'XOM', 'MCD', 'PEP', 'ABT', 'NFLX',
    
    # Indian Companies
    'TATAMOTORS.BO', 'RELIANCE.BO', 'HDFCBANK.BO', 'INFY.BO', 'HDFC.BO', 'HDFC.NS', 'ICICIBANK.BO',
    'LT.BO', 'SBIN.BO', 'HINDUNILVR.BO', 'ITC.BO', 'KOTAKBANK.BO', 'BHARTIARTL.BO'
]

@st.cache_data(ttl=3600)
def update_ticker_list():
    """Fetch or update the list of tickers from an external source."""
    return list(set(POPULAR_TICKERS))  # Update list if needed

@st.cache_data(ttl=3600)
def fetch_exchange_rate(base_currency, target_currency):
    """Fetch the static conversion rate from base_currency to target_currency."""
    base_rate = STATIC_EXCHANGE_RATES.get(base_currency)
    target_rate = STATIC_EXCHANGE_RATES.get(target_currency)
    
    if base_rate and target_rate:
        return target_rate / base_rate
    else:
        st.warning(f"Exchange rate for {target_currency} not found.")
        return None

@st.cache_data(ttl=3600)
def fetch_data_yahoo(ticker):
    """Fetch historical stock data from Yahoo Finance."""
    try:
        df = yf.download(ticker)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        return df
    except Exception as e:
        st.error(f"Error fetching data from Yahoo Finance for ticker '{ticker}': {e}")
        return None

def preprocess_data(data, lookback_period):
    """Preprocess data by creating lag features."""
    data['Date'] = pd.to_datetime(data.index)
    data['Day'] = data['Date'].dt.day
    data['Month'] = data['Date'].dt.month
    data['Year'] = data['Date'].dt.year
    
    target = data['Close']
    features = []
    
    # Create lag features based on lookback period
    for i in range(1, min(lookback_period, 30) + 1):
        lag_feature = target.shift(i).rename(f'Lag_{i}')
        features.append(lag_feature)

    features_df = pd.concat(features, axis=1)
    features_df = features_df.dropna()
    
    # Align target with features
    target = target[features_df.index]
    
    if len(features_df) != len(target):
        st.warning("Mismatch between features and target lengths.")
        return None, None, None, None

    # Ensure there is enough data to split
    if len(features_df) < 10:  # Minimum of 10 samples for split
        st.warning("Not enough data to create training and test sets.")
        return None, None, None, None

    return train_test_split(features_df, target, test_size=0.2, shuffle=False)

def train_model(X_train, y_train):
    """Train the linear regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def predict_future(model, X_test, lookahead_days):
    """Predict future stock prices."""
    predictions = []
    last_known_data = X_test.iloc[-1:].copy()
    
    for _ in range(lookahead_days):
        pred = model.predict(last_known_data)[0]
        predictions.append(pred)
        
        # Shift features for the next prediction
        last_known_data = last_known_data.shift(-1, axis=1)
        last_known_data.iloc[0, -1] = pred
    
    return predictions

def calculate_profit_loss(predicted_prices, current_price):
    """Calculate profit or loss based on predictions."""
    profit_loss = []
    for price in predicted_prices:
        if price > current_price:
            profit_loss.append('Profit')
        elif price < current_price:
            profit_loss.append('Loss')
        else:
            profit_loss.append('No Change')
    
    # Provide investment advice
    if all(p > current_price for p in predicted_prices):
        advice = 'Strong Buy - All predicted prices are higher than the current price.'
    elif all(p < current_price for p in predicted_prices):
        advice = 'Strong Sell - All predicted prices are lower than the current price.'
    else:
        advice = 'Hold - Mixed predictions.'
    
    return profit_loss, advice

def plot_predictions(result):
    """Plot the predicted prices."""
    plt.figure(figsize=(14, 7))
    plt.plot(result['Date'], result['Predicted Price'], marker='o', linestyle='-', color='b', label='Predicted Price')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Stock Price Prediction')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)  # Use Streamlit to display the plot

def stock_price_prediction(ticker, lookback_period, lookahead_days, target_currency):
    """Combine all steps to predict stock prices."""
    data = fetch_data_yahoo(ticker)
    
    if data is None:
        st.warning(f"No data available for {ticker}.")
        return None, None, None
    
    X_train, X_test, y_train, y_test = preprocess_data(data, lookback_period)
    
    if X_train is None:
        st.warning("Insufficient data to create training and test sets.")
        return None, None, None
    
    model = train_model(X_train, y_train)
    predictions = predict_future(model, X_test, lookahead_days)
    
    current_price = data['Close'].iloc[-1]
    
    # Determine base currency of the ticker
    if ticker.endswith('.BO') or ticker.endswith('.NS'):
        base_currency = 'INR'
    else:
        base_currency = 'USD'
    
    if base_currency != target_currency:
        # Fetch static exchange rate for the target currency
        conversion_rate = fetch_exchange_rate(base_currency, target_currency)
        if conversion_rate is None:
            st.warning(f"Conversion rate for {target_currency} not found.")
            return None, None, None
        
        # Convert predictions to the target currency
        predictions_converted = [p * conversion_rate for p in predictions]
    else:
        conversion_rate = 1.0
        predictions_converted = predictions
    
    # Use pd.DateOffset to handle date calculations correctly
    future_dates = pd.date_range(start=data.index[-1] + pd.DateOffset(days=1), periods=lookahead_days)
    profit_loss, advice = calculate_profit_loss(predictions, current_price)
    
    # Prepare result DataFrame
    result = pd.DataFrame({
        'Date': future_dates,
        'Predicted Price': predictions_converted,
        'Profit/Loss': profit_loss
    })
    
    plot_predictions(result)
    
    return result, conversion_rate, advice

# Streamlit App
def main():
    st.title("Stock Sage")
    
    # User input
    ticker = st.selectbox('Select Ticker:', update_ticker_list(), index=0)
    lookback_period = st.slider('Lookback Period:', min_value=1, max_value=365, value=30)
    lookahead_days = st.slider('Lookahead Days:', min_value=1, max_value=365, value=20)
    target_currency = st.selectbox('Target Currency:', list(STATIC_EXCHANGE_RATES.keys()), index=0)
    
    run_prediction_button = st.button('Run Prediction')
    
    if run_prediction_button:
        st.write("Fetching and predicting...")
        result, conversion_rate, advice = stock_price_prediction(
            ticker,
            lookback_period,
            lookahead_days,
            target_currency
        )
        if result is not None:
            # Rename column to include target currency
            result.columns = ['Date', f'Predicted Price ({target_currency})', 'Profit/Loss']
            st.write(f"Conversion Rate: {conversion_rate}")
            st.write(f"Advice: {advice}")
            st.dataframe(result)

    # Embed the Dialogflow chatbot using components.html with fixed positioning
    components.html(
        """
        <style>
          #chatbot-container {
            position: fixed;
            bottom: 20px;  /* Fixed 20px from the bottom */
            right: 20px;   /* Fixed 20px from the right edge */
            z-index: 1000; /* Ensure it's on top of other content */
          }
        </style>
        <div id="chatbot-container">
          <script src="https://www.gstatic.com/dialogflow-console/fast/messenger/bootstrap.js?v=1"></script>
          <df-messenger
            chat-title="Stocksage_chatbot"
            agent-id="9cdbe2fd-eceb-481c-bd1a-4d2d8a2e9dc2"
            language-code="en">
          </df-messenger>
        </div>
        """,
        height=600
    )

if __name__ == "__main__":
    main()
