#Import Required Libraries
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf


# Utility Functions
# Function to calculate indiactor values
def calculate_indicators(data):
    """Add EMA, RSI, and MACD indicators to the dataframe."""
    data['EMA5'] = data['Close'].ewm(span=5, adjust=False).mean()
    data['EMA13'] = data['Close'].ewm(span=13, adjust=False).mean()
    
    # RSI
    delta = data['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema26 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = ema12 - ema26
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    
    return data

# Strategy Signal Generators
def ema_crossover_strategy(data):
    """EMA 5/13 crossover strategy"""
    data['Signal'] = np.where(data['EMA5'] > data['EMA13'], 1, 0)
    data['Position'] = data['Signal'].diff()
    return data

def rsi_strategy(data):
    """RSI > 60 = Buy, RSI < 40 = Sell"""
    data['Signal'] = np.where(data['RSI'] > 60, 1, np.where(data['RSI'] < 40, 0, np.nan))
    data['Signal'] = data['Signal'].ffill().fillna(0)
    data['Position'] = data['Signal'].diff()
    return data

def macd_strategy(data):
    """MACD crossover with Signal Line"""
    data['Signal'] = np.where(data['MACD'] > data['Signal_Line'], 1, 0)
    data['Position'] = data['Signal'].diff()
    return data

# Generate Buy/Sell Signal
def generate_signals(data):
    """Generate buy/sell signals based on EMA crossover """
    # When EMA5 line is above EMA13 generate Buy Signal
    conditions = (data['EMA5'] > data['EMA13'])
    data['Signal'] = np.where(conditions, 1, 0)  # 1 = Buy Signal
    data['Position'] = data['Signal'].diff()
    return data


   #conditions = (data['EMA5'] > data['EMA13']) & (data['RSI'] > 60) & (data['MACD'] > data['Signal_Line'])

# Calculate Risk Management
def calculate_risk_management(investment_amount, stock_price, stop_loss_pct, risk_pct):
    risk_amount = investment_amount * (risk_pct / 100)
    stop_loss_price = stock_price * (1 - stop_loss_pct / 100)
    risk_per_share = stock_price - stop_loss_price
    if risk_per_share <= 0:
        return 0, stop_loss_price, 0
    number_of_stocks = int(risk_amount / risk_per_share)
    return number_of_stocks, stop_loss_price, risk_amount

# Backtest the strategy on historical data
def backtest(data, investment_amount, stop_loss_pct, risk_pct):
    """Backtest strategy and calculate P/L."""
    capital = investment_amount
    shares = 0
    trade_log = []

    for i in range(1, len(data)):
        # Buy condition
        if data['Position'].iloc[i] == 1:
            stock_price = data['Close'].iloc[i]
            qty, stop_loss, risk_amt = calculate_risk_management(
                investment_amount, stock_price, stop_loss_pct, risk_pct
            )
            if qty > 0:
                shares = qty
                capital -= shares * stock_price
                trade_log.append({
                    'Date': data.index[i],
                    'Type': 'Buy',
                    'Price': stock_price,
                    'Shares': shares,
                    'Stop Loss': stop_loss
                })

        # Sell condition
        elif data['Position'].iloc[i] == -1 and shares > 0:
            stock_price = data['Close'].iloc[i]
            capital += shares * stock_price
            trade_log.append({
                'Date': data.index[i],
                'Type': 'Sell',
                'Price': stock_price,
                'Shares': shares,
                'PnL': shares * (stock_price - trade_log[-1]['Price'])
            })
            shares = 0

    # Final portfolio value
    if shares > 0:
        capital += shares * data['Close'].iloc[-1]
    return capital, trade_log


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Code for displaying data on Streamlit Dashboard
st.set_page_config(page_title="Trading Bot Dashboard", layout="wide")
st.title("📈 Automated Trading System Dashboard")
#st.write("EMA (5 & 13) Crossover Strategy with Risk Management Backtesting")
#st.write("EMA (5 & 13) Crossover + RSI + MACD Strategy with Risk Management")

# Sidebar inputs- User to select Investment amount, Stop loss%, Risk% for calculating number of stocks to be purchased and handle Stop loss
st.sidebar.header("Strategy Parameters")

strategy_choice = st.sidebar.radio("Choose Strategy", 
                                   ("EMA Crossover (5/13)", 
                                    "RSI (>60 Buy, <40 Sell)", 
                                    "MACD Signal Crossover"))

investment_amount = st.sidebar.number_input("💰 Investment Amount", min_value=1000.0, step=1000.0, value=100000.0)
stop_loss_pct = st.sidebar.slider("🛑 Stop Loss (%)", min_value=0.5, max_value=10.0, step=0.1, value=2.0)
risk_pct = st.sidebar.slider("⚠️ Risk per Trade (%)", min_value=0.5, max_value=5.0, step=0.1, value=1.0)

# Fetching Stock data from Yahoo Finance
# User to select Stock, Start Date and End Date for historical data range
st.sidebar.header("Data Source (Yahoo Finance)")
ticker = st.sidebar.text_input("📈 Stock Ticker (e.g., AAPL, TSLA, INFY.NS)", value="AAPL")
start_date = st.sidebar.date_input("📅 Start Date", value=pd.to_datetime("2024-01-01"))
end_date = st.sidebar.date_input("📅 End Date", value=pd.Timestamp.today())

if st.sidebar.button("Fetch Data"):
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)

        data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]  # Flatten column names
        data = data.dropna()
        #data.head

        if data.empty:
            st.error("❌ No data found for the selected ticker and date range.")
        else:
            #st.write(data)
            st.success(f"✅ Data fetched for {ticker} from {start_date} to {end_date}.")
            
            # Process data
            data = calculate_indicators(data)
            #data = generate_signals(data)

             # Apply Chosen Strategy
            if strategy_choice == "EMA Crossover (5/13)":
                data = ema_crossover_strategy(data)
            elif strategy_choice == "RSI (>60 Buy, <40 Sell)":
                data = rsi_strategy(data)
            else:
                data = macd_strategy(data)

            # Backtest Strategy
            final_capital, trade_log = backtest(data, investment_amount, stop_loss_pct, risk_pct)
            pnl = final_capital - investment_amount
            prctprofitloss = (pnl/investment_amount) * 100

            # Display Metrics - Total number of trades, Final Portfolio amount, P/L, %P/L
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("📦 Total Trades", len(trade_log)//2)
            col2.metric("💸 Final Portfolio Value", f"{final_capital:,.2f}")
            col3.metric("📈 Net P/L", f"{pnl:,.2f}")
            col4.metric("💲P/L", f"{prctprofitloss:,.2f}%")

            # Plotting chart for displaying stock data, 5/13 EMA, Buy signal(Green) & Sell signal(Red)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))
            fig.add_trace(go.Scatter(x=data.index, y=data['EMA5'], mode='lines', name='EMA 5'))
            fig.add_trace(go.Scatter(x=data.index, y=data['EMA13'], mode='lines', name='EMA 13'))
            buy_signals = data[data['Position'] == 1]
            sell_signals = data[data['Position'] == -1]
            fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Close'], mode='markers', name='Buy Signal', marker=dict(color='green', size=10)))
            fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Close'], mode='markers', name='Sell Signal', marker=dict(color='red', size=10)))
            fig.update_layout(title=f"📊 {ticker} Price with Buy/Sell Signals", xaxis_title="Date", yaxis_title="Price")
            st.plotly_chart(fig, use_container_width=True)

            # Trade Log
            st.subheader("📄 Trade Log")
            st.dataframe(pd.DataFrame(trade_log))

            # Disclaimer
            st.subheader("**Disclaimer**")
            st.write("** The system is a research and educational tool, not investment advice.")
            st.write("** Backtested results do not guarantee future performance.")
            st.write("** Users are responsible for their own investment decisions.")

    except Exception as e:
        st.error(f"Error fetching data: {e}")
else:
    st.info("👈 Enter stock details and click *Fetch Data* to begin.")





