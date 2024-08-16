import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import requests
import pandas as pd
import plotly.express as px 
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime
from scipy.stats import norm

# Function definitions
@st.cache_data(ttl=300)
def get_stock_data(symbols):
    # Download historical market data from Yahoo Finance
    stock_data = yf.download(symbols, period='6mo', interval='1d',progress= False)  
    stock_data.reset_index(inplace=True)  
    stock_data = stock_data[['Date', 'Open', 'High', 'Low', 'Close']]  
    return stock_data

def get_underlying_asset_price(symbols):
    # Fetch the most recent closing price
    stock_data = yf.download(symbols, period='5d', interval='1d',progress= False) 
    most_recent_close = stock_data['Close'][-1]  # Get the most recent close price
    return most_recent_close
# Black-Scholes formula for Call option
def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price
def black_scholes_put(S, K, T, r, sigma):
    # Black-Scholes formula for put option price
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = (K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))
    return put_price

def calculate_call_payoff(asset_prices, strike_price, T, r, sigma, premium):
    option_prices = [black_scholes_call(S, strike_price, T, r, sigma) for S in asset_prices]
    payoffs = [option_price - premium for option_price in option_prices]
    return payoffs

def calculate_put_payoff(asset_prices, strike_price, T, r, sigma, premium):
    option_prices = [black_scholes_put(S, strike_price, T, r, sigma) for S in asset_prices]
    payoffs = [option_price - premium for option_price in option_prices]
    return payoffs

# Function to calculate the payoff for a straddle option
def calculate_straddle_payoff(asset_prices, strike_price, T, r, sigma, premium_call, premium_put):
    call_prices = [black_scholes_call(S, strike_price, T, r, sigma) for S in asset_prices]
    put_prices = [black_scholes_put(S, strike_price, T, r, sigma) for S in asset_prices]
    payoffs = [(call_price + put_price) - (premium_call + premium_put) for call_price, put_price in zip(call_prices, put_prices)]
    
    return payoffs


def calculate_covered_call_payoff_bs(asset_prices, purchase_price, strike_price, T, r, sigma, premium):
    call_option_prices = np.array([black_scholes_call(S, strike_price, T, r, sigma) for S in asset_prices])
    long_asset_payoff = asset_prices - strike_price
    short_call_payoff = premium - call_option_prices
    payoffs = long_asset_payoff + short_call_payoff
    
    return payoffs

def calculate_married_put_payoff_bs(asset_prices, purchase_price, strike_price, T, r, sigma, premium_paid):
    put_option_prices = np.array([black_scholes_put(S, strike_price, T, r, sigma) for S in asset_prices])
    stock_payoff = asset_prices - purchase_price
    put_payoff = put_option_prices - premium_paid
    payoffs = stock_payoff + put_payoff
    return payoffs
    
def calculate_bull_call_spread_payoff_bs(asset_prices, strike_price_long_call, strike_price_short_call, T, r, sigma, premium_long_call, premium_short_call):
    # Calculate call option prices using the Black-Scholes formula
    long_call_prices = np.array([black_scholes_call(S, strike_price_long_call, T, r, sigma) for S in asset_prices])
    short_call_prices = np.array([black_scholes_call(S, strike_price_short_call, T, r, sigma) for S in asset_prices])
    
    # Payoff from the long call position
    long_call_payoff = long_call_prices - premium_long_call
    # Payoff from the short call position (negative because it's short)
    short_call_payoff = premium_short_call - short_call_prices
    
    # The bull call spread payoff is the sum of the long call and short call payoffs
    bull_call_spread_payoff = long_call_payoff + short_call_payoff
    return bull_call_spread_payoff

# Function to calculate the payoff for a Bull Put Spread option
def calculate_bull_put_spread_payoff_bs(asset_prices, strike_price_short_put, strike_price_long_put, T, r, sigma, premium_short_put, premium_long_put):
    # Calculate put option prices using the Black-Scholes formula
    short_put_prices = np.array([black_scholes_put(S, strike_price_short_put, T, r, sigma) for S in asset_prices])
    long_put_prices = np.array([black_scholes_put(S, strike_price_long_put, T, r, sigma) for S in asset_prices])
    
    # Payoff from the short put position
    short_put_payoff = premium_short_put - short_put_prices
    # Payoff from the long put position (negative because we're buying it)
    long_put_payoff = long_put_prices - premium_long_put
    
    # The bull put spread payoff is the sum of the short put and long put payoffs
    bull_put_spread_payoff = short_put_payoff + long_put_payoff
    return bull_put_spread_payoff
# Function to calculate the payoff for a Protective Collar option
def calculate_protective_collar_payoff_bs(asset_prices, purchase_price, strike_price_put, premium_put, strike_price_call, premium_call, T, r, sigma):
    # Calculate put and call option prices using the Black-Scholes formula
    put_prices = np.array([black_scholes_put(S, strike_price_put, T, r, sigma) for S in asset_prices])
    call_prices = np.array([black_scholes_call(S, strike_price_call, T, r, sigma) for S in asset_prices])
    premium_put = black_scholes_put(purchase_price, strike_price_put, T, r, sigma)
    premium_call = black_scholes_call(purchase_price, strike_price_call, T, r, sigma)
    # Profit or loss from holding the stock
    stock_payoff = asset_prices - purchase_price
    # Payoff from the long put position (using Black-Scholes put prices to reflect premiums)
    long_put_payoff = np.maximum(strike_price_put - asset_prices, 0) - premium_put
    # Payoff from the short call position (using Black-Scholes call prices to reflect premiums)
    short_call_payoff = premium_call - np.maximum(asset_prices - strike_price_call, 0)
    
    # The protective collar payoff is the sum of the stock, put, and call payoffs
    protective_collar_payoff = stock_payoff + long_put_payoff + short_call_payoff
    return protective_collar_payoff
# Function to calculate the payoff for a Long Call Butterfly Spread option
def calculate_long_call_butterfly_payoff_bs(asset_prices, strike_price_low, strike_price_mid, strike_price_high, T, r, sigma, premium_low, premium_mid, premium_high):
    # Calculate call option prices using the Black-Scholes formula
    call_prices_low = np.array([black_scholes_call(S, strike_price_low, T, r, sigma) for S in asset_prices])
    call_prices_mid = np.array([black_scholes_call(S, strike_price_mid, T, r, sigma) for S in asset_prices])
    call_prices_high = np.array([black_scholes_call(S, strike_price_high, T, r, sigma) for S in asset_prices])
    
    # Payoff for buying one low strike call
    long_call_low_payoff = call_prices_low - premium_low
    # Payoff for selling two mid strike calls
    short_call_mid_payoff = 2 * (premium_mid - call_prices_mid)
    # Payoff for buying one high strike call
    long_call_high_payoff = call_prices_high - premium_high

    # Total payoff for the butterfly spread
    butterfly_payoff = long_call_low_payoff + short_call_mid_payoff + long_call_high_payoff
    return butterfly_payoff
# Function to calculate the payoff for an Iron Butterfly option
def calculate_iron_butterfly_payoff_bs(asset_prices, strike_price_atm, strike_price_otm_put, strike_price_otm_call, T, r, sigma, premium_atm, premium_otm_put, premium_otm_call):
    # Calculate option prices using the Black-Scholes formula
    call_price_atm = np.array([black_scholes_call(S, strike_price_atm, T, r, sigma) for S in asset_prices])
    put_price_atm = np.array([black_scholes_put(S, strike_price_atm, T, r, sigma) for S in asset_prices])
    call_price_otm_call = np.array([black_scholes_call(S, strike_price_otm_call, T, r, sigma) for S in asset_prices])
    put_price_otm_put = np.array([black_scholes_put(S, strike_price_otm_put, T, r, sigma) for S in asset_prices])

    # Payoff from the long out-of-the-money put
    long_put_payoff = put_price_otm_put - premium_otm_put
    # Payoff from the short at-the-money put
    short_atm_put_payoff = premium_atm - put_price_atm
    # Payoff from the short at-the-money call
    short_atm_call_payoff = premium_atm - call_price_atm
    # Payoff from the long out-of-the-money call
    long_call_payoff = call_price_otm_call - premium_otm_call

    # Total payoff for the Iron Butterfly
    iron_butterfly_payoff = long_put_payoff + short_atm_put_payoff + short_atm_call_payoff + long_call_payoff
    return iron_butterfly_payoff
# Function to calculate the payoff for an Iron Condor option
def calculate_iron_condor_payoff_bs(asset_prices, strike_price_put_buy, strike_price_put_sell, strike_price_call_buy, strike_price_call_sell, T, r, sigma, premium_put_buy, premium_put_sell, premium_call_buy, premium_call_sell):
    # Calculate option prices using the Black-Scholes formula
    put_price_buy = np.array([black_scholes_put(S, strike_price_put_buy, T, r, sigma) for S in asset_prices])
    put_price_sell = np.array([black_scholes_put(S, strike_price_put_sell, T, r, sigma) for S in asset_prices])
    call_price_buy = np.array([black_scholes_call(S, strike_price_call_buy, T, r, sigma) for S in asset_prices])
    call_price_sell = np.array([black_scholes_call(S, strike_price_call_sell, T, r, sigma) for S in asset_prices])

    # Payoff from the long put
    long_put_payoff = put_price_buy - premium_put_buy
    # Payoff from the short put
    short_put_payoff = premium_put_sell - put_price_sell
    # Payoff from the short call
    short_call_payoff = premium_call_sell - call_price_sell
    # Payoff from the long call
    long_call_payoff = call_price_buy - premium_call_buy

    # The iron condor payoff is the sum of the individual option payoffs
    iron_condor_payoff = long_put_payoff + short_put_payoff + short_call_payoff + long_call_payoff
    return iron_condor_payoff

# Streamlit app layout
st.title('Options Strategy Visualizer')

# API data fetch
symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "SPY", "QQQ", "DIA", "META", "NFLX", "NVDA", "TSLA", "AMD"]
selected_symbol = st.selectbox("Select Stock Symbol", symbols)

# Display a placeholder for the most recent close price
most_recent_close = 0.00

if selected_symbol:
    stock_data = get_stock_data(selected_symbol)
    
    if not stock_data.empty:
        # Create and display the candlestick chart
        fig_candlestick = go.Figure(data=[go.Candlestick(x=stock_data['Date'],
                                                         open=stock_data['Open'],
                                                         high=stock_data['High'],
                                                         low=stock_data['Low'],
                                                         close=stock_data['Close'])])
        fig_candlestick.update_layout(title=f'Candlestick Chart for {selected_symbol}', xaxis_title='Date', yaxis_title='Price (USD)')
        st.plotly_chart(fig_candlestick)
        
        # Update and display the most recent adjusted close price
        most_recent_close = get_underlying_asset_price(selected_symbol)
        if most_recent_close is not None:
            st.write(f"Most recent adjusted close price for {selected_symbol}: ${most_recent_close:.2f}")
        else:
            st.warning("Unable to fetch the most recent adjusted close price.")
    else:
            st.error(f"No data available for {selected_symbol}. Please try again later.")

# Strategy selection
strategy = st.selectbox("Select Strategy", ["Call", "Put", "Straddle", "Covered Call", "Married Put","Bull Call Spread","Bull Put Spread",
                                            "Protective Collar","Long Call Butterfly Spread","Iron Butterfly","Iron Condor"])
# Strategy parameters
asset_price = st.number_input('Underlying Asset Price', value=most_recent_close, key=f'asset_price_{selected_symbol}')
strike_price = st.number_input('Strike Price', value=int(round(asset_price, 0)), step=1, key=f'strike_{strategy}')
premium = st.number_input('Premium',value=10.0, step = 0.01, key=f'premium_{strategy}')
current_date = st.date_input('Current Date', key=f'current_{strategy}')
expiration_date = st.date_input('Expiration Date', key=f'expiry_{strategy}')

T = (expiration_date - current_date).days / 365  # Time to expiration in years
r = 0.05  # Risk-free interest rate
sigma = 0.25  # Volatility

if strategy == "Covered Call":
    purchase_price = st.number_input('Purchase Price of Underlying Asset', value= asset_price, key='purchase_price')
elif strategy == "Married Put":
    purchase_price = st.number_input('Purchase Price of Underlying Asset', value= asset_price, key='purchase_price')
    premium_paid = st.number_input('Premium Paid for Put Option', value=10.0, key='premium_paid')
elif strategy == "Straddle":
    # Inputs for the Straddle strategy
    strike_price = st.number_input('Strike Price for Both Call and Put', min_value=0, value = strike_price, key='strike_price_straddle')
    premium_call = st.number_input('Premium Paid for Call Option', min_value=0.0, value=5.0, key='premium_call_straddle')
    premium_put = st.number_input('Premium Paid for Put Option', min_value=0.0, value=5.0, key='premium_put_straddle')
elif strategy == "Bull Call Spread":
    # For Bull Call Spread, we need two strike prices and two premiums
    strike_price_long_call = st.number_input('Strike Price for Long Call', min_value=0, value = strike_price, key='strike_price_long_call')
    premium_long_call = st.number_input('Premium for Long Call', min_value=0.0, value=10.0, key='premium_long_call')
    strike_price_short_call = st.number_input('Strike Price for Short Call', min_value=0, value = strike_price + 10, key='strike_price_short_call')
    premium_short_call = st.number_input('Premium for Short Call', min_value=0.0, value=5.0, key='premium_short_call')
elif strategy == "Bull Put Spread":
    # For Bull Put Spread, we need two strike prices and two premiums
    strike_price_short_put = st.number_input('Strike Price for Short Put', min_value=0, value=strike_price, key='strike_price_short_put')
    premium_short_put = st.number_input('Premium for Short Put', min_value=0.0, value=10.0, key='premium_short_put')
    strike_price_long_put = st.number_input('Strike Price for Long Put', min_value=0, value=strike_price - 10, key='strike_price_long_put')
    premium_long_put = st.number_input('Premium for Long Put', min_value=0.0, value=5.0, key='premium_long_put')
# Inputs specific to Protective Collar
elif strategy == "Protective Collar":
    purchase_price = st.number_input('Purchase Price of Underlying Asset', value=asset_price, key='purchase_price_collar')
    strike_price_put = st.number_input('Strike Price for Long Put', min_value=0, value= strike_price -5, key='strike_price_put')
    premium_put = st.number_input('Premium for Long Put', min_value=0.0, value=5.0, key='premium_put')
    strike_price_call = st.number_input('Strike Price for Short Call', min_value=0, value= strike_price + 10, key='strike_price_call')
    premium_call = st.number_input('Premium for Short Call', min_value=0.0, value=5.0, key='premium_call')
elif strategy == "Long Call Butterfly Spread":
    # For Long Call Butterfly Spread, we need three strike prices and three premiums
    strike_price_low = st.number_input('Strike Price for Low Call', min_value=0, value=strike_price - 10, key='strike_price_low')
    premium_low = st.number_input('Premium for Low Call', min_value=0.0, value=3.0, key='premium_low')
    strike_price_mid = st.number_input('Strike Price for Mid Call', min_value=0, value=strike_price, key='strike_price_mid')
    premium_mid = st.number_input('Premium for Mid Call', min_value=0.0, value=4.0, key='premium_mid')
    strike_price_high = st.number_input('Strike Price for High Call', min_value=0, value=strike_price + 10, key='strike_price_high')
    premium_high = st.number_input('Premium for High Call', min_value=0.0, value=8.0, key='premium_high')
# Inputs specific to Iron Butterfly
elif strategy == "Iron Butterfly":
    # For an Iron Butterfly, we need the strike price and premiums for the ATM options and the OTM put and call
    strike_price_atm = st.number_input('Strike Price for ATM Options', min_value=0, value=strike_price, key='strike_price_atm')
    premium_atm = st.number_input('Premium for ATM Options', min_value=0.0, value=10.0, key='premium_atm')
    strike_price_otm_put = st.number_input('Strike Price for OTM Put', min_value=0, value=strike_price - 10, key='strike_price_otm_put')
    premium_otm_put = st.number_input('Premium for OTM Put', min_value=0.0, value=10.0, key='premium_otm_put')
    strike_price_otm_call = st.number_input('Strike Price for OTM Call', min_value=0, value=strike_price + 10, key='strike_price_otm_call')
    premium_otm_call = st.number_input('Premium for OTM Call', min_value=0.0, value=3.0, key='premium_otm_call')
    # Inputs specific to Iron Condor
elif strategy == "Iron Condor":
    strike_price_put_buy = st.number_input('Strike Price for Buy Put', min_value=0, value=strike_price - 20, key='strike_price_put_buy')
    premium_put_buy = st.number_input('Premium for Buy Put', min_value=0.0, value=1.0, key='premium_put_buy')
    
    strike_price_put_sell = st.number_input('Strike Price for Sell Put', min_value=0, value=strike_price -10, key='strike_price_put_sell')
    premium_put_sell = st.number_input('Premium for Sell Put', min_value=0.0, value=2.0, key='premium_put_sell')
    
    strike_price_call_sell = st.number_input('Strike Price for Sell Call', min_value=0, value=strike_price +10, key='strike_price_call_sell')
    premium_call_sell = st.number_input('Premium for Sell Call', min_value=0.0, value=2.0, key='premium_call_sell')
    
    strike_price_call_buy = st.number_input('Strike Price for Buy Call', min_value=0, value=strike_price +20, key='strike_price_call_buy')
    premium_call_buy = st.number_input('Premium for Buy Call', min_value=0.0, value=1.0, key='premium_call_buy')
    # Define a label for the strategy
strategy_label = f'{strategy} Option Payoff'
# Calculation and plotting based on strategy
asset_prices = np.linspace(max(0, strike_price - 100), strike_price + 100, 100)
fig, ax = plt.subplots()

# Initialize payoffs to an empty array
payoffs = np.zeros_like(asset_prices)

# Calculate payoffs for each strategy
if strategy == "Call":
    payoffs = calculate_call_payoff(asset_prices, strike_price, T, r, sigma, premium)
    strategy_label = 'Long Call Payoff'
elif strategy == "Put":
    payoffs = calculate_put_payoff(asset_prices, strike_price, T, r, sigma, premium)
    strategy_label = 'Long Put Payoff'
elif strategy == "Straddle":
    payoffs = calculate_straddle_payoff(asset_prices, strike_price, T, r, sigma, premium_call, premium_put)
    strategy_label = 'Straddle Payoff'
    break_even_up = strike_price + premium
    break_even_down = strike_price - premium
elif strategy == "Covered Call":
    payoffs = calculate_covered_call_payoff_bs(asset_prices,purchase_price,strike_price,T,r,sigma,premium)    
    strategy_label = 'Covered Call Payoff'
elif strategy == "Married Put":
    payoffs = calculate_married_put_payoff_bs(asset_prices, purchase_price, strike_price, T, r, sigma, premium_paid)
    strategy_label = 'Married Put Payoff'
elif strategy == "Bull Call Spread":
    payoffs = calculate_bull_call_spread_payoff_bs(asset_prices, strike_price_long_call, strike_price_short_call, T, r, sigma, premium_long_call, premium_short_call)
    strategy_label = 'Bull Call Spread Payoff'
elif strategy == "Bull Put Spread":
    payoffs = calculate_bull_put_spread_payoff_bs(asset_prices, strike_price_short_put, strike_price_long_put, T, r, sigma, premium_short_put, premium_long_put)
    strategy_label = 'Bull Put Spread Payoff'
elif strategy == "Protective Collar": 
    payoffs = calculate_protective_collar_payoff_bs(asset_prices, purchase_price, strike_price_put, premium_put, strike_price_call, premium_call, T, r, sigma)
    strategy_label = 'Protective Collar Payoff'
elif strategy == "Long Call Butterfly Spread":
    payoffs = calculate_long_call_butterfly_payoff_bs(asset_prices, strike_price_low, strike_price_mid, strike_price_high, T, r, sigma, premium_low, premium_mid, premium_high)
    strategy_label = 'Long Call Butterfly Spread Payoff'
elif strategy == "Iron Butterfly":
    payoffs = calculate_iron_butterfly_payoff_bs(asset_prices, strike_price_atm, strike_price_otm_put, strike_price_otm_call, T, r, sigma, premium_atm, premium_otm_put, premium_otm_call)
    strategy_label = 'Iron Butterfly Payoff'
elif strategy == "Iron Condor":
    # Calculate payoffs for the Iron Condor strategy
    payoffs = calculate_iron_condor_payoff_bs(asset_prices, strike_price_put_buy, strike_price_put_sell, strike_price_call_buy, strike_price_call_sell, T, r, sigma, premium_put_buy, premium_put_sell, premium_call_buy, premium_call_sell)
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
    ax.fill_between(asset_prices, payoffs, where=(np.array(payoffs) > 0), color='green', alpha=0.3)
    ax.fill_between(asset_prices, payoffs, where=(np.array(payoffs) <= 0), color='red', alpha=0.3)
elif strategy in ["Covered Call"]:
    break_even = purchase_price - premium
    max_profit = premium  # Maximum profit is the premium received
    ax.axvline(x=break_even, color='blue', linestyle='--')
    ax.text(break_even, 0, f' Break-Even\n ${break_even}', horizontalalignment='right')
    ax.axhline(y=max_profit, color='blue', linestyle='--')
    ax.text(asset_prices[-1], max_profit, f' Max Profit: ${max_profit}', verticalalignment='bottom')
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
# Apply shading logic for strategies with more complex payoff structures like Butterfly spreads
elif strategy in ["Long Call Butterfly Spread", "Iron Butterfly"]:
    # Identify profit and loss indices
    profit_indices = (asset_prices > strike_price_low) & (asset_prices < strike_price_high)
    loss_indices = ~profit_indices

    ax.fill_between(asset_prices, payoffs, 0, where=profit_indices, color='green', alpha=0.3)
    ax.fill_between(asset_prices, payoffs, 0, where=loss_indices, color='red', alpha=0.3)

elif strategy == "Protective Collar":
    # Calculate payoffs
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

    total_premium = premium * 2
    # Calculate break-even points
    upper_break_even = strike_price + total_premium
    lower_break_even = strike_price - total_premium

    # Calculate the profit indices (where the strategy is profitable)
    profit_indices = (asset_prices < lower_break_even) | (asset_prices > upper_break_even)
    
    # Fill between for profit areas
    ax.fill_between(asset_prices, payoffs, 0, where=profit_indices, color='green', alpha=0.3)
    
    # Fill between for loss areas
    ax.fill_between(asset_prices, payoffs, 0, where=~profit_indices, color='red', alpha=0.3)

    # Break-even lines
    ax.axvline(x=upper_break_even, color='blue', linestyle='--', label=f'Break-Even Up ${upper_break_even:.2f}')
    ax.axvline(x=lower_break_even, color='purple', linestyle='--', label=f'Break-Even Down ${lower_break_even:.2f}')
    
    # Text for break-even points
    ax.text(upper_break_even, 0, f' ${upper_break_even:.2f}', horizontalalignment='right', verticalalignment='bottom')
    ax.text(lower_break_even, 0, f' ${lower_break_even:.2f}', horizontalalignment='left', verticalalignment='bottom')

# Add legend and layout adjustments
ax.legend()
plt.tight_layout()

st.pyplot(fig)
