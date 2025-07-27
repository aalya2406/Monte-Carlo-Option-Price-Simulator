import numpy as np

def monte_carlo_option_price(S, K, T, r, sigma, num_simulations=10000, option_type='call'):
    """
    Monte Carlo simulation to price a European option.
    """
    np.random.seed(42)  # for reproducibility

    # Simulate asset price at maturity using Geometric Brownian Motion
    Z = np.random.standard_normal(num_simulations)
    ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

    if option_type == 'call':
        payoff = np.maximum(ST - K, 0)
    elif option_type == 'put':
        payoff = np.maximum(K - ST, 0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    # Discounted average payoff
    price = np.exp(-r * T) * np.mean(payoff)
    return price, ST  # return paths too for plotting
