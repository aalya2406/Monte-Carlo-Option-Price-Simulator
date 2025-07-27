import streamlit as st
import matplotlib.pyplot as plt
from monte_carlo import monte_carlo_option_price

st.set_page_config(page_title="Monte Carlo Option Pricing", layout="wide")
st.title("Monte Carlo Option Pricing Simulator")

st.sidebar.header("Input Parameters")

# Sidebar Inputs
S = st.sidebar.number_input("Initial Stock Price (S)", value=100.0)
K = st.sidebar.number_input("Strike Price (K)", value=100.0)
T = st.sidebar.number_input("Time to Maturity (T in years)", value=1.0)
r = st.sidebar.number_input("Risk-Free Interest Rate (r)", value=0.05)
sigma = st.sidebar.number_input("Volatility (σ)", value=0.2)
num_simulations = st.sidebar.slider("Number of Simulations", 1000, 100000, 10000, step=1000)

if st.button("Run Simulation"):

    # Run both simulations
    call_price, ST_call = monte_carlo_option_price(S, K, T, r, sigma, num_simulations, option_type='call')
    put_price, ST_put = monte_carlo_option_price(S, K, T, r, sigma, num_simulations, option_type='put')

    # Show results side-by-side
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="📈 Call Option Price", value=f"${call_price:.2f}")
    with col2:
        st.metric(label="📉 Put Option Price", value=f"${put_price:.2f}")

    # Show both histograms side by side
    col3, col4 = st.columns(2)

    with col3:
        fig1, ax1 = plt.subplots()
        ax1.hist(ST_call, bins=50, alpha=0.7, color='skyblue')
        ax1.axvline(K, color='red', linestyle='--', label='Strike Price')
        ax1.set_title(f"Simulated Call Payoffs (n={num_simulations})")
        ax1.set_xlabel("Final Asset Price")
        ax1.set_ylabel("Frequency")
        ax1.legend()
        st.pyplot(fig1)

    with col4:
        fig2, ax2 = plt.subplots()
        ax2.hist(ST_put, bins=50, alpha=0.7, color='lightcoral')
        ax2.axvline(K, color='red', linestyle='--', label='Strike Price')
        ax2.set_title(f"Simulated Put Payoffs (n={num_simulations})")
        ax2.set_xlabel("Final Asset Price")
        ax2.set_ylabel("Frequency")
        ax2.legend()
        st.pyplot(fig2)
