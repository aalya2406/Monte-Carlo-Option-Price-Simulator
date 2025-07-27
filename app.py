import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from monte_carlo import monte_carlo_option_price, simulate_price_paths

# https://intro.quantecon.org/monte_carlo.html 

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

# Optional: path visualization parameters even before clicking "Run"
st.sidebar.markdown("---")
st.sidebar.subheader("Price Path Visualization Settings")
num_paths = st.sidebar.slider("Number of Sample Paths", min_value=1, max_value=50, value=10)
num_steps = st.sidebar.slider("Number of Time Steps", min_value=10, max_value=500, value=100)

if st.button("Run Simulation"):

    # Monte Carlo option pricing for call and put
    call_price, call_ci, ST_call = monte_carlo_option_price(S, K, T, r, sigma, num_simulations, option_type='call')
    put_price, put_ci, ST_put = monte_carlo_option_price(S, K, T, r, sigma, num_simulations, option_type='put')

    # Show Call and Put Prices side by side
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="📈 Call Option Price", value=f"${call_price:.2f}")
    with col2:
        st.metric(label="📉 Put Option Price", value=f"${put_price:.2f}")

    # Show histograms side by side
    col3, col4 = st.columns(2)

    with col3:
        fig1, ax1 = plt.subplots()
        ax1.hist(ST_call, bins=50, alpha=0.7, color='skyblue')
        ax1.axvline(K, color='red', linestyle='--', label='Strike Price')
        ax1.set_title(f"Simulated Call Final Prices (n={num_simulations})")
        ax1.set_xlabel("Final Asset Price")
        ax1.set_ylabel("Frequency")
        ax1.legend()
        st.pyplot(fig1)
        st.write(f"95% Confidence Interval: ${call_ci[0]:.2f} - ${call_ci[1]:.2f}")

    with col4:
        fig2, ax2 = plt.subplots()
        ax2.hist(ST_put, bins=50, alpha=0.7, color='lightcoral')
        ax2.axvline(K, color='red', linestyle='--', label='Strike Price')
        ax2.set_title(f"Simulated Put Final Prices (n={num_simulations})")
        ax2.set_xlabel("Final Asset Price")
        ax2.set_ylabel("Frequency")
        ax2.legend()
        st.pyplot(fig2)
        st.write(f"95% Confidence Interval: ${put_ci[0]:.2f} - ${put_ci[1]:.2f}")

    # Simulate and plot price paths
    price_paths = simulate_price_paths(S, T, r, sigma, num_steps=num_steps, num_paths=num_paths)

    fig3, ax3 = plt.subplots()
    time_grid = np.linspace(0, T, num_steps + 1)
    for i in range(num_paths):
        ax3.plot(time_grid, price_paths[i], lw=1, alpha=0.7)
    ax3.set_xlabel("Time (years)")
    ax3.set_ylabel("Stock Price")
    ax3.set_title(f"{num_paths} Simulated Stock Price Paths (Geometric Brownian Motion)")
    st.pyplot(fig3)
