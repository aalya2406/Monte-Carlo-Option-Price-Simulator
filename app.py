import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from monte_carlo import monte_carlo_option_price, simulate_price_paths
from monte_carlo import generate_heatmap_data

# Set Streamlit config
st.set_page_config(page_title="Monte Carlo Option Pricing", layout="wide")
st.title("Monte Carlo Option Pricing Simulator")

# Sidebar Inputs
st.sidebar.header("Input Parameters")
S = st.sidebar.number_input("Initial Stock Price (S)", value=100.0)
K = st.sidebar.number_input("Strike Price (K)", value=100.0)
T = st.sidebar.number_input("Time to Maturity (T in years)", value=1.0)
r = st.sidebar.number_input("Risk-Free Interest Rate (r)", value=0.05)
sigma = st.sidebar.number_input("Volatility (σ)", value=0.2)
num_simulations = st.sidebar.slider("Number of Simulations", 1000, 100000, 10000, step=1000)

st.sidebar.markdown("---")
st.sidebar.subheader("Price Path Visualization Settings")
num_paths = st.sidebar.slider("Number of Sample Paths", min_value=1, max_value=50, value=10)
num_steps = st.sidebar.slider("Number of Time Steps", min_value=10, max_value=500, value=100)

# Run simulation button
if st.button("Run Simulation"):

    # Option pricing
    call_price, call_ci, ST_call = monte_carlo_option_price(S, K, T, r, sigma, num_simulations, option_type='call')
    put_price, put_ci, ST_put = monte_carlo_option_price(S, K, T, r, sigma, num_simulations, option_type='put')

    # Display metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="📈 Call Option Price", value=f"${call_price:.2f}")
    with col2:
        st.metric(label="📉 Put Option Price", value=f"${put_price:.2f}")

    # Display histograms
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
        st.markdown(f"**95% Confidence Interval (Call):** `${call_ci[0]:.2f} - {call_ci[1]:.2f}`")

    with col4:
        fig2, ax2 = plt.subplots()
        ax2.hist(ST_put, bins=50, alpha=0.7, color='lightcoral')
        ax2.axvline(K, color='red', linestyle='--', label='Strike Price')
        ax2.set_title(f"Simulated Put Final Prices (n={num_simulations})")
        ax2.set_xlabel("Final Asset Price")
        ax2.set_ylabel("Frequency")
        ax2.legend()
        st.pyplot(fig2)
        st.markdown(f"**95% Confidence Interval (Put):** `${put_ci[0]:.2f} - {put_ci[1]:.2f}`")


    # Simulate price paths
    price_paths = simulate_price_paths(S, T, r, sigma, num_steps=num_steps, num_paths=num_paths)

    # Plot price paths
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    time_grid = np.linspace(0, T, num_steps + 1)
    for i in range(num_paths):
        ax3.plot(time_grid, price_paths[i], lw=1, alpha=0.6)
    ax3.set_xlabel("Time (Years)")
    ax3.set_ylabel("Stock Price")
    ax3.set_title(f"{num_paths} Simulated Stock Price Paths (Geometric Brownian Motion)")
    ax3.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig3)

    # Plot histogram with KDE for final prices
    final_prices = price_paths[:, -1]
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    sns.histplot(final_prices, bins=50, kde=True, stat="density", color='skyblue', edgecolor='black', ax=ax4)
    ax4.axvline(np.mean(final_prices), color='green', linestyle='dotted', linewidth=2, label='Mean')
    ax4.axvline(K, color='red', linestyle='dashed', linewidth=2, label='Strike Price')
    ax4.set_title("Histogram & Density of Final Simulated Stock Prices")
    ax4.set_xlabel("Final Stock Price")
    ax4.set_ylabel("Density")
    ax4.legend()
    ax4.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig4)


   # --- Option Price Heatmap Section ---


    # --- Generate Heatmaps for Call and Put Together ---
    with st.spinner("Generating heatmaps for Call and Put options..."):
        
        heatmap_call, sigmas, maturities = generate_heatmap_data(S, K, r, sigma, T, option_type='call', num_simulations=num_simulations)

        heatmap_put, _, _ = generate_heatmap_data(S, K, r, sigma, T, option_type='put', num_simulations=num_simulations)




        # Two columns to display side-by-side
        col_h1, col_h2 = st.columns(2)

        with col_h1:
            fig_call, ax_call = plt.subplots(figsize=(6, 5))
            sns.heatmap(
                heatmap_call,
                xticklabels=np.round(sigmas, 2),
                yticklabels=np.round(maturities, 2),
                cmap="YlGnBu",
                cbar_kws={'label': 'Call Option Price'},
                ax=ax_call
            )
            ax_call.set_xlabel("Volatility (σ)")
            ax_call.set_ylabel("Time to Maturity (T)")
            ax_call.set_title("Call Option Price Heatmap")
            st.pyplot(fig_call)

        with col_h2:
            fig_put, ax_put = plt.subplots(figsize=(6, 5))
            sns.heatmap(
                heatmap_put,
                xticklabels=np.round(sigmas, 2),
                yticklabels=np.round(maturities, 2),
                cmap="YlOrRd",
                cbar_kws={'label': 'Put Option Price'},
                ax=ax_put
            )
            ax_put.set_xlabel("Volatility (σ)")
            ax_put.set_ylabel("Time to Maturity (T)")
            ax_put.set_title("Put Option Price Heatmap")
            st.pyplot(fig_put)
