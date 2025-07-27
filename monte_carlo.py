import numpy as np
from scipy.stats import norm

def monte_carlo_option_price(S, K, T, r, sigma, num_simulations=10000, option_type='call', confidence_level=0.95):
    """
    Monte Carlo simulation to price a European option with confidence intervals.
    
    Returns:
        price: Estimated option price (mean)
        conf_interval: Tuple (lower_bound, upper_bound) of confidence interval
        ST: Simulated end prices (for plotting)
    """
    np.random.seed(42)

    Z = np.random.standard_normal(num_simulations)
    ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

    if option_type == 'call':
        payoffs = np.maximum(ST - K, 0)
    elif option_type == 'put':
        payoffs = np.maximum(K - ST, 0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    discounted_payoffs = np.exp(-r * T) * payoffs
    price = discounted_payoffs.mean()

    # Confidence interval calculation
    std_dev = discounted_payoffs.std(ddof=1)
    std_err = std_dev / np.sqrt(num_simulations)

    z = norm.ppf(0.5 + confidence_level / 2)  # e.g., 1.96 for 95%
    conf_lower = price - z * std_err
    conf_upper = price + z * std_err

    return price, (conf_lower, conf_upper), ST


def simulate_price_paths(S, T, r, sigma, num_steps=100, num_paths=10):
    """
    Simulate multiple stock price paths using Geometric Brownian Motion.
    
    Parameters:
        S: initial stock price
        T: time to maturity (years)
        r: risk-free rate
        sigma: volatility
        num_steps: number of time intervals
        num_paths: number of simulation paths
    
    Returns:
        paths: numpy array of shape (num_paths, num_steps+1)
    """
    dt = T / num_steps
    paths = np.zeros((num_paths, num_steps + 1))
    paths[:, 0] = S

    for t in range(1, num_steps + 1):
        Z = np.random.standard_normal(num_paths)  # random shocks
        paths[:, t] = paths[:, t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    
    return paths