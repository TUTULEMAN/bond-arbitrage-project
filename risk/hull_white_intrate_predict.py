from src.utils.common_imports import np, plt, pd, os, interp1d

def create_yield_curve_interpolator(csv_path):
    """"
    Assumes the CSV includes a date column ('NEW_DATE' or 'Date') and yield columns:
      'BC_1MONTH', 'BC_3MONTH', 'BC_6MONTH', 'BC_1YEAR',
      'BC_2YEAR', 'BC_3YEAR', 'BC_5YEAR', 'BC_7YEAR', 'BC_10YEAR'.
    
    Maturities (in years) are assumed to be:
      1/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10.

    """
    df = pd.read_csv(csv_path)
    if 'NEW_DATE' in df.columns:
        df['NEW_DATE'] = pd.to_datetime(df['NEW_DATE'])
        df.sort_values('NEW_DATE', inplace=True)
        latest_row = df.iloc[-1]
    elif 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.sort_values('Date', inplace=True)
        latest_row = df.iloc[-1]
    else:
        latest_row = df.iloc[-1]

    maturities = [1/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10]
    yield_cols = ['BC_1MONTH', 'BC_3MONTH', 'BC_6MONTH', 'BC_1YEAR',
                  'BC_2YEAR', 'BC_3YEAR', 'BC_5YEAR', 'BC_7YEAR', 'BC_10YEAR']
    
    yields = []
    for col in yield_cols:
        try:
            # Convert to float and (if necessary) from percentage to decimal
            y_val = float(latest_row[col])
            yields.append(y_val / 100.0)
        except Exception:
            yields.append(np.nan)
    
    # Remove any entries with missing data
    maturities = np.array(maturities)
    yields = np.array(yields)
    valid = ~np.isnan(yields)
    maturities = maturities[valid]
    yields = yields[valid]
    
    # Create a linear interpolation function for the yield curve
    yield_interp = interp1d(maturities, yields, kind='linear', fill_value='extrapolate')
    return yield_interp

def calibrated_theta(t, a, sigma, constant_theta, yield_interp):
    """
    theta(t) = df0/dt + a * f0(t) + (sigma^2/(2*a)) * (1 - exp(-2*a*t))
    """
    eps = 1e-4
    f0_t = yield_interp(t)
    # Ensure t-eps is non-negative
    t_minus = max(t - eps, 0)
    f0_t_plus = yield_interp(t + eps)
    f0_t_minus = yield_interp(t_minus)
    dfdt = (f0_t_plus - f0_t_minus) / (2 * eps)
    return dfdt + a * f0_t + (sigma**2 / (2 * a)) * (1 - np.exp(-2 * a * t))

def simulate_hull_white(r0, theta, a, sigma, T, dt, n_paths, theta_func=None):
    """
    Simulates the Hull White short rate model with an optional time-dependent theta.
    
    Model:
        dr(t) = (theta(t) - a * r(t)) dt + sigma * dW(t)
    
    Parameters:
        r0         : Initial short rate.
        theta      : Constant theta value (used if theta_func is None).
        a          : Mean reversion speed.
        sigma      : Volatility.
        T          : Time horizon (years).
        dt         : Time step (years).
        n_paths    : Number of simulation paths.
        theta_func : Optional function theta_func(t, a, sigma, theta) returning theta(t).
                     If None, theta is treated as constant.
    
    Returns:
        times : 1D array of time points.
        rates : 2D array of simulated short rates (shape: [n_steps+1, n_paths]).

    More information on material in 'research/hullwhite_rp.pdf/'
    """
    n_steps = int(T / dt)
    times = np.linspace(0, T, n_steps + 1)
    rates = np.zeros((n_steps + 1, n_paths))
    rates[0, :] = r0

    for i in range(1, n_steps + 1):
        t = times[i - 1]
        theta_val = theta_func(t, a, sigma, constant_theta)
        dW = np.random.normal(0, np.sqrt(dt), size=n_paths)
        rates[i, :] = rates[i - 1, :] + (theta_val - a * rates[i - 1, :]) * dt + sigma * dW

    return times, rates

def plot_simulation(times, rates):
    """
    Plots sample simulation paths, the mean forecast, and a 5th–95th percentile envelope.
    """
    n_paths = rates.shape[1]
    mean_rate = np.mean(rates, axis=1)
    perc05 = np.percentile(rates, 5, axis=1)
    perc95 = np.percentile(rates, 95, axis=1)
    
    plt.figure(figsize=(10, 6))
    for i in range(min(n_paths, 10)):
        plt.plot(times, rates[:, i], lw=1, alpha=0.7, label=f'Path {i+1}' if i == 0 else "")
    plt.plot(times, mean_rate, 'k-', lw=2, label='Mean Forecast')
    plt.fill_between(times, perc05, perc95, color='gray', alpha=0.3, label='5th–95th Percentile')
    plt.xlabel("Time (Years)")
    plt.ylabel("Short Rate")
    plt.title("Hull–White Model Simulation with Calibrated Theta")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    """
    Model parameters (tweak or calibrate these with market data as needed)
    - Interest rate: 4%
    - Const theta: 5%
    - Mean reversion speed: 10%
    - Volatility: 2% (might change depending on market environment)
    - Time horizon: 5 years
    - Daily time steps: 1/252
    - Simulation(s): 10,000
    """
    r0 = 0.04                
    constant_theta = 0.05    
    a = 0.1                  
    sigma = 0.01             
    T = 5.0                  
    dt = 1 / 252             
    n_paths = 10000           
    
    csv_path=os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'yield_curve_data.csv')
    try:
        yield_interp = create_yield_curve_interpolator(csv_path)
        print("Yield curve data loaded successfully.")
    except Exception as e:
        print("Error loading yield curve data:", e)
        yield_interp = None

    if yield_interp is not None:
        theta_func = lambda t, a, sigma, constant_theta: calibrated_theta(t, a, sigma, constant_theta, yield_interp)
    else:
        theta_func = lambda t, a, sigma, constant_theta: constant_theta

    # Run simulation
    times, rates = simulate_hull_white(r0, constant_theta, a, sigma, T, dt, n_paths, theta_func)
    plot_simulation(times, rates)
    
    predicted_rate = np.mean(rates[-1, :])
    print(f"Predicted short rate at T = {T} years: {predicted_rate:.4f}")

if __name__ == '__main__':
    main()
