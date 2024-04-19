import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import requests

import streamlit as st
import numpy as np
import requests
import pandas as pd
import plotly.express as px  # Import Plotly Express

# Function definitions
@st.cache(ttl=300)
def get_stock_data(symbol, API_KEY):
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={API_KEY}"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()['Time Series (Daily)']
    df = pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close'])
    
    for date, values in data.items():
        row = {'Date': date, 'Open': float(values['1. open']), 'High': float(values['2. high']),
               'Low': float(values['3. low']), 'Close': float(values['4. close'])}
        row_df = pd.DataFrame([row])  # Convert a single-row dict to DataFrame
        df = pd.concat([df, row_df], ignore_index=True)

    
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    return df

# Function to calculate the payoff for a call option
def calculate_call_payoff(prices, strike, asset_price):
    return np.maximum(prices - strike, 0) - asset_price

# Function to calculate the payoff for a put option
def calculate_put_payoff(prices, strike, asset_price):
    return np.maximum(strike - prices, 0) - asset_price

# Function to calculate the payoff for a straddle option
def calculate_straddle_payoff(asset_prices, strike, premium):
    call_payoff = np.maximum(asset_prices - strike, 0) - premium
    put_payoff = np.maximum(strike - asset_prices, 0) - premium
    return call_payoff + put_payoff

# Function to calculate the payoff for a covered call option
def calculate_covered_call_payoff(asset_prices, purchase_price, strike_price, premium_received):
    long_asset_payoff = asset_prices - purchase_price
    short_call_payoff = np.where(asset_prices > strike_price, strike_price - asset_prices + premium_received, premium_received)
    return long_asset_payoff + short_call_payoff

def calculate_married_put_payoff(asset_prices, purchase_price, strike_price, premium_paid):
    # Profit or loss from holding the stock
    stock_payoff = asset_prices - purchase_price
    # Payoff from the put option
    put_payoff = np.maximum(strike_price - asset_prices, 0) - premium_paid
    # The married put payoff is the sum of the stock and put option payoffs
    married_put_payoff = stock_payoff + put_payoff
    return married_put_payoff
# Function to calculate the payoff for a Bull Call Spread option
def calculate_bull_call_spread_payoff(asset_prices, strike_price_long_call, strike_price_short_call, premium_long_call, premium_short_call):
    # Payoff from the long call position
    long_call_payoff = np.maximum(asset_prices - strike_price_long_call, 0) - premium_long_call
    # Payoff from the short call position (negative because it's short)
    short_call_payoff = premium_short_call - np.maximum(asset_prices - strike_price_short_call, 0)
    # The bull call spread payoff is the sum of the long call and short call payoffs
    bull_call_spread_payoff = long_call_payoff + short_call_payoff
    return bull_call_spread_payoff
# Function to calculate the payoff for a Bull Put Spread option
def calculate_bull_put_spread_payoff(asset_prices, strike_price_short_put, strike_price_long_put, premium_short_put, premium_long_put):
    # Payoff from the short put position
    short_put_payoff = premium_short_put - np.maximum(strike_price_short_put - asset_prices, 0)
    # Payoff from the long put position (negative because we're buying it)
    long_put_payoff = np.maximum(strike_price_long_put - asset_prices, 0) - premium_long_put
    # The bull put spread payoff is the sum of the short put and long put payoffs
    bull_put_spread_payoff = short_put_payoff + long_put_payoff
    return bull_put_spread_payoff
# Function to calculate the payoff for a Protective Collar option
def calculate_protective_collar_payoff(asset_prices, purchase_price, strike_price_put, premium_put, strike_price_call, premium_call):
    # Profit or loss from holding the stock
    stock_payoff = asset_prices - purchase_price

    # Payoff from the long put position
    long_put_payoff = np.maximum(strike_price_put - asset_prices, 0) - premium_put

    # Payoff from the short call position (negative because it's short)
    short_call_payoff = premium_call - np.maximum(asset_prices - strike_price_call, 0)

    # The protective collar payoff is the sum of the stock, put, and call payoffs
    protective_collar_payoff = stock_payoff + long_put_payoff + short_call_payoff
    return protective_collar_payoff
# Function to calculate the payoff for a Long Call Butterfly Spread option
def calculate_long_call_butterfly_payoff(asset_prices, strike_price_low, strike_price_mid, strike_price_high, premium_low, premium_mid, premium_high):
    # Buying one low strike call
    long_call_low_payoff = np.maximum(asset_prices - strike_price_low, 0) - premium_low
    # Selling two mid strike calls
    short_call_mid_payoff = 2 * (premium_mid - np.maximum(asset_prices - strike_price_mid, 0))
    # Buying one high strike call
    long_call_high_payoff = np.maximum(asset_prices - strike_price_high, 0) - premium_high

    # Total payoff for the butterfly spread
    butterfly_payoff = long_call_low_payoff + short_call_mid_payoff + long_call_high_payoff
    return butterfly_payoff
# Function to calculate the payoff for an Iron Butterfly option
def calculate_iron_butterfly_payoff(asset_prices, strike_price_put, premium_put, strike_price_call, premium_call, premium_atm, strike_price_atm):
    # Payoff from the long out-of-the-money put
    long_put_payoff = np.maximum(strike_price_put - asset_prices, 0) - premium_put
    # Payoff from the short at-the-money put
    short_atm_put_payoff = premium_atm - np.maximum(strike_price_atm - asset_prices, 0)
    # Payoff from the short at-the-money call
    short_atm_call_payoff = premium_atm - np.maximum(asset_prices - strike_price_atm, 0)
    # Payoff from the long out-of-the-money call
    long_call_payoff = np.maximum(asset_prices - strike_price_call, 0) - premium_call
 # Total payoff for the Iron Butterfly
    iron_butterfly_payoff = long_put_payoff + short_atm_put_payoff + short_atm_call_payoff + long_call_payoff
    return iron_butterfly_payoff
# Function to calculate the payoff for an Iron Condor option
def calculate_iron_condor_payoff(asset_prices, strike_price_put_buy, premium_put_buy, strike_price_put_sell, premium_put_sell, strike_price_call_sell, premium_call_sell, strike_price_call_buy, premium_call_buy):
    # Payoff from the long put
    long_put_payoff = np.maximum(strike_price_put_buy - asset_prices, 0) - premium_put_buy
    # Payoff from the short put
    short_put_payoff = premium_put_sell - np.maximum(strike_price_put_sell - asset_prices, 0)
    # Payoff from the short call
    short_call_payoff = premium_call_sell - np.maximum(asset_prices - strike_price_call_sell, 0)
    # Payoff from the long call
    long_call_payoff = np.maximum(asset_prices - strike_price_call_buy, 0) - premium_call_buy

    # The iron condor payoff is the sum of the individual option payoffs
    iron_condor_payoff = long_put_payoff + short_put_payoff + short_call_payoff + long_call_payoff
    return iron_condor_payoff


# Streamlit app layout
st.title('Options Strategy Visualizer')

# API data fetch
API_KEY = st.secrets["API_KEY"]
symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "SPY", "QQQ", "DIA", "META", "NFLX", "NVDA", "TSLA", "AMD"]
selected_symbol = st.selectbox("Select Stock Symbol", symbols)

if st.button("Fetch Data"):
    stock_data = get_stock_data(selected_symbol, API_KEY)
    # Create a Plotly interactive line chart showing only the closing prices
    fig = px.line(stock_data, x='Date', y='Close', title='Stock Closing Prices', labels={'Close': 'Closing Price (USD)'})
    st.plotly_chart(fig)

    # Process and display more data if necessary, like futures_data, iv_data, etc.

# Strategy selection
strategy = st.selectbox("Select Strategy", ["Call", "Put", "Straddle", "Covered Call", "Married Put","Bull Call Spread","Bull Put Spread",
                                            "Protective Collar","Long Call Butterfly Spread","Iron Butterfly","Iron Condor"])

# Strategy parameters
strike_price = st.number_input('Strike Price', value=100, key=f'strike_{strategy}')
expiration = st.date_input('Expiration Date', key=f'expiry_{strategy}')
asset_price = st.number_input('Underlying Asset Price', value=100, key=f'asset_price_{strategy}')
premium = st.number_input('Premium',value=10, key=f'premium_{strategy}')

if strategy == "Covered Call":
    purchase_price = st.number_input('Purchase Price of Underlying Asset', value=100.0, key='purchase_price')
elif strategy == "Married Put":
    purchase_price = st.number_input('Purchase Price of Underlying Asset', value=100.0, key='purchase_price')
    premium_paid = st.number_input('Premium Paid for Put Option', value=10.0, key='premium_paid')
elif strategy == "Straddle":
    # Inputs for the Straddle strategy
    strike_price = st.number_input('Strike Price for Both Call and Put', min_value=0, value=100, key='strike_price_straddle')
    premium_call = st.number_input('Premium Paid for Call Option', min_value=0.0, value=5.0, key='premium_call_straddle')
    premium_put = st.number_input('Premium Paid for Put Option', min_value=0.0, value=5.0, key='premium_put_straddle')
elif strategy == "Bull Call Spread":
    # For Bull Call Spread, we need two strike prices and two premiums
    strike_price_long_call = st.number_input('Strike Price for Long Call', min_value=0, value=100, key='strike_price_long_call')
    premium_long_call = st.number_input('Premium for Long Call', min_value=0.0, value=10.0, key='premium_long_call')
    strike_price_short_call = st.number_input('Strike Price for Short Call', min_value=0, value=110, key='strike_price_short_call')
    premium_short_call = st.number_input('Premium for Short Call', min_value=0.0, value=5.0, key='premium_short_call')
elif strategy == "Bull Put Spread":
    # For Bull Put Spread, we need two strike prices and two premiums
    strike_price_short_put = st.number_input('Strike Price for Short Put', min_value=0, value=100, key='strike_price_short_put')
    premium_short_put = st.number_input('Premium for Short Put', min_value=0.0, value=10.0, key='premium_short_put')
    strike_price_long_put = st.number_input('Strike Price for Long Put', min_value=0, value=90, key='strike_price_long_put')
    premium_long_put = st.number_input('Premium for Long Put', min_value=0.0, value=5.0, key='premium_long_put')
# Inputs specific to Protective Collar
elif strategy == "Protective Collar":
    purchase_price = st.number_input('Purchase Price of Underlying Asset', value=100.0, key='purchase_price_collar')
    strike_price_put = st.number_input('Strike Price for Long Put', min_value=0, value=95, key='strike_price_put')
    premium_put = st.number_input('Premium for Long Put', min_value=0.0, value=5.0, key='premium_put')
    strike_price_call = st.number_input('Strike Price for Short Call', min_value=0, value=110, key='strike_price_call')
    premium_call = st.number_input('Premium for Short Call', min_value=0.0, value=5.0, key='premium_call')
elif strategy == "Long Call Butterfly Spread":
    # For Long Call Butterfly Spread, we need three strike prices and three premiums
    strike_price_low = st.number_input('Strike Price for Low Call', min_value=0, value=90, key='strike_price_low')
    premium_low = st.number_input('Premium for Low Call', min_value=0.0, value=3.0, key='premium_low')
    strike_price_mid = st.number_input('Strike Price for Mid Call', min_value=0, value=100, key='strike_price_mid')
    premium_mid = st.number_input('Premium for Mid Call', min_value=0.0, value=4.0, key='premium_mid')
    strike_price_high = st.number_input('Strike Price for High Call', min_value=0, value=110, key='strike_price_high')
    premium_high = st.number_input('Premium for High Call', min_value=0.0, value=8.0, key='premium_high')
# Inputs specific to Iron Butterfly
elif strategy == "Iron Butterfly":
    # For an Iron Butterfly, we need the strike price and premiums for the ATM options and the OTM put and call
    strike_price_atm = st.number_input('Strike Price for ATM Options', min_value=0, value=100, key='strike_price_atm')
    premium_atm = st.number_input('Premium for ATM Options', min_value=0.0, value=10.0, key='premium_atm')
    strike_price_otm_put = st.number_input('Strike Price for OTM Put', min_value=0, value=90, key='strike_price_otm_put')
    premium_otm_put = st.number_input('Premium for OTM Put', min_value=0.0, value=10.0, key='premium_otm_put')
    strike_price_otm_call = st.number_input('Strike Price for OTM Call', min_value=0, value=110, key='strike_price_otm_call')
    premium_otm_call = st.number_input('Premium for OTM Call', min_value=0.0, value=3.0, key='premium_otm_call')
    # Inputs specific to Iron Condor
elif strategy == "Iron Condor":
    strike_price_put_buy = st.number_input('Strike Price for Buy Put', min_value=0, value=80, key='strike_price_put_buy')
    premium_put_buy = st.number_input('Premium for Buy Put', min_value=0.0, value=1.0, key='premium_put_buy')
    
    strike_price_put_sell = st.number_input('Strike Price for Sell Put', min_value=0, value=90, key='strike_price_put_sell')
    premium_put_sell = st.number_input('Premium for Sell Put', min_value=0.0, value=2.0, key='premium_put_sell')
    
    strike_price_call_sell = st.number_input('Strike Price for Sell Call', min_value=0, value=110, key='strike_price_call_sell')
    premium_call_sell = st.number_input('Premium for Sell Call', min_value=0.0, value=2.0, key='premium_call_sell')
    
    strike_price_call_buy = st.number_input('Strike Price for Buy Call', min_value=0, value=120, key='strike_price_call_buy')
    premium_call_buy = st.number_input('Premium for Buy Call', min_value=0.0, value=1.0, key='premium_call_buy')
    # Define a label for the strategy
strategy_label = f'{strategy} Option Payoff'
# Calculation and plotting based on strategy
asset_prices = np.linspace(max(0, strike_price - 50), strike_price + 50, 100)
fig, ax = plt.subplots()

# Initialize payoffs to an empty array
payoffs = np.zeros_like(asset_prices)

# Calculate payoffs for each strategy
if strategy == "Call":
    payoffs = calculate_call_payoff(asset_prices, strike_price, premium)
    strategy_label = 'Long Call Payoff'
elif strategy == "Put":
    payoffs = calculate_put_payoff(asset_prices, strike_price, premium)
    strategy_label = 'Long Put Payoff'
elif strategy == "Straddle":
    payoffs = calculate_straddle_payoff(asset_prices, strike_price, premium)
    strategy_label = 'Straddle Payoff'
    break_even_up = strike_price + premium
    break_even_down = strike_price - premium
elif strategy == "Covered Call":
    payoffs = calculate_covered_call_payoff(asset_prices, purchase_price, strike_price, premium)
    strategy_label = 'Covered Call Payoff'
elif strategy == "Married Put":
    payoffs = calculate_married_put_payoff(asset_prices, purchase_price, strike_price, premium_paid)
    strategy_label = 'Married Put Payoff'
elif strategy == "Bull Call Spread":
    payoffs = calculate_bull_call_spread_payoff(asset_prices, strike_price_long_call, strike_price_short_call, premium_long_call, premium_short_call)
    strategy_label = 'Bull Call Spread Payoff'
elif strategy == "Bull Put Spread":
    payoffs = calculate_bull_put_spread_payoff(asset_prices, strike_price_short_put, strike_price_long_put, premium_short_put, premium_long_put)
    strategy_label = 'Bull Put Spread Payoff'
elif strategy == "Protective Collar": 
    payoffs = calculate_protective_collar_payoff(asset_prices, purchase_price, strike_price_put, premium_put, strike_price_call, premium_call)
    strategy_label = 'Protective Collar Payoff'
elif strategy == "Long Call Butterfly Spread":
    payoffs = calculate_long_call_butterfly_payoff(asset_prices, strike_price_low, strike_price_mid, strike_price_high, premium_low, premium_mid, premium_high)
    strategy_label = 'Long Call Butterfly Spread Payoff'
elif strategy == "Iron Butterfly":
    payoffs = calculate_iron_butterfly_payoff(asset_prices, strike_price_otm_put, premium_otm_put, strike_price_otm_call, premium_otm_call, premium_atm, strike_price_atm)
    strategy_label = 'Iron Butterfly Payoff'
elif strategy == "Iron Condor":
    # Calculate payoffs for the Iron Condor strategy
    payoffs = calculate_iron_condor_payoff(
        asset_prices,
        strike_price_put_buy,
        premium_put_buy,
        strike_price_put_sell,
        premium_put_sell,
        strike_price_call_sell,
        premium_call_sell,
        strike_price_call_buy,
        premium_call_buy
    )
    strategy_label = 'Iron Condor Payoff'    


# Common plot settings
ax.plot(asset_prices, payoffs, label=strategy_label)
ax.axhline(0, color='grey', lw=1)
ax.set_xlabel('Stock Price (USD)')
ax.set_ylabel('Profit / Loss (USD) x 100')
ax.set_title(f'{strategy} Payoff at Different Prices')

# Shading for profit/loss based on the strategy
# For strategies like Straddle, Butterfly, Iron Butterfly, Iron Condor, etc., where the profit/loss regions are not straightforward,
# we need to compute the specific conditions for profit and loss based on the strategy's payoff profile.
if strategy in ["Call", "Put", "Bull Call Spread", "Bull Put Spread", "Covered Call", "Married Put"]:
    ax.fill_between(asset_prices, payoffs, 0, where=(payoffs > 0), color='green', alpha=0.3, interpolate=True)
    ax.fill_between(asset_prices, payoffs, 0, where=(payoffs <= 0), color='red', alpha=0.3, interpolate=True)
elif strategy in ["Long Call Butterfly Spread", "Iron Butterfly", "Iron Condor"]:
    # Typically, these strategies have a profit region around the ATM strikes and losses elsewhere
    profit_indices = (payoffs > 0)
    ax.fill_between(asset_prices, payoffs, 0, where=profit_indices, color='green', alpha=0.3)
    ax.fill_between(asset_prices, payoffs, 0, where=~profit_indices, color='red', alpha=0.3)
    
    # For Iron Condor, max profit occurs between the short put and short call strike prices
elif strategy == "Iron Condor":
        profit_range = (asset_prices > strike_price_put_sell) & (asset_prices < strike_price_call_sell)
        ax.fill_between(asset_prices, payoffs, 0, where=profit_range, color='green', alpha=0.3)
        ax.fill_between(asset_prices, payoffs, 0, where=~profit_range, color='red', alpha=0.3)
# Strategy-specific annotations and markers
elif strategy == "Covered Call":
    break_even = purchase_price - premium
    max_profit = premium  # Maximum profit is the premium received
    ax.axvline(x=break_even, color='blue', linestyle='--')
    ax.text(break_even, 0, f' Break-Even\n ${break_even}', horizontalalignment='right')
    ax.axhline(y=max_profit, color='blue', linestyle='--')
    ax.text(asset_prices[-1], max_profit, f' Max Profit: ${max_profit}', verticalalignment='bottom')
# Apply shading logic for strategies with more complex payoff structures like Butterfly spreads
elif strategy in ["Long Call Butterfly Spread", "Iron Butterfly"]:
    # Identify profit and loss indices
    profit_indices = (asset_prices > strike_price_low) & (asset_prices < strike_price_high)
    loss_indices = ~profit_indices

    ax.fill_between(asset_prices, payoffs, 0, where=profit_indices, color='green', alpha=0.3)
    ax.fill_between(asset_prices, payoffs, 0, where=loss_indices, color='red', alpha=0.3)

elif strategy == "Protective Collar":
    # Calculate payoffs
    payoffs = calculate_protective_collar_payoff(asset_prices, purchase_price, strike_price_put, premium_put, strike_price_call, premium_call)

# Strategy-specific annotations and markers
    strategy_label = 'Protective Collar Payoff'
# Calculate maximum profit and maximum loss
    max_profit = (strike_price_call - purchase_price) - (premium_put - premium_call)
    max_loss = (purchase_price - strike_price_put) - (premium_put - premium_call)
    
    # Create plot
    fig, ax = plt.subplots()
    ax.plot(asset_prices, payoffs, label=strategy_label)
    
# Shading for profit and loss areas
    ax.fill_between(asset_prices, payoffs, where=(payoffs >= 0), color='green', alpha=0.3, interpolate=True)
    ax.fill_between(asset_prices, payoffs, where=(payoffs < 0), color='red', alpha=0.3, interpolate=True)
    
# Max profit and loss lines and annotations
    ax.axhline(y=max_profit, color='blue', linestyle='--', label=f'Max Profit: ${max_profit:.2f}')
    # The max loss should be plotted as a negative value because it represents a loss
    ax.axhline(y=-max_loss, color='orange', linestyle='--', label=f'Max Loss: ${-max_loss:.2f}')
    # Set axis labels and title
    ax.set_xlabel('Stock Price (USD)')
    ax.set_ylabel('Profit / Loss (USD)')
    ax.set_title('Protective Collar Strategy Payoff')

elif strategy == "Straddle":
    # Calculate break-even points
    upper_break_even = strike_price + premium
    lower_break_even = strike_price - premium

    profit_indices = (asset_prices < break_even_down) | (asset_prices > break_even_up)
    ax.fill_between(asset_prices, payoffs, 0, where=(asset_prices >= upper_break_even) & (payoffs > 0), color='green', alpha=0.3)
    ax.fill_between(asset_prices, payoffs, 0, where=(asset_prices <= lower_break_even) & (payoffs > 0), color='green', alpha=0.3)
    ax.fill_between(asset_prices, payoffs, 0, where=(payoffs <= 0), color='red', alpha=0.3)


# Break-even lines
    ax.axvline(x=break_even_up, color='blue', linestyle='--', label=f'Break-Even Up ${break_even_up:.2f}')
    ax.axvline(x=break_even_down, color='purple', linestyle='--', label=f'Break-Even Down ${break_even_down:.2f}')
    ax.text(break_even_up, 0, f' ${break_even_up:.2f}', horizontalalignment='right')
    ax.text(break_even_down, 0, f' ${break_even_down:.2f}', horizontalalignment='left')

# Add legend and layout adjustments
ax.legend()
plt.tight_layout()

st.pyplot(fig)
