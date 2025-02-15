from src.utils.common_imports import yf, pd, os, np, plt

TRADING_DAYS = 252  #number of trading days in a year

class BondArbBacktest:
    def __init__(self, data_path, window=20, threshold=1.5):
        self.data_path = data_path
        self.window = window
        self.threshold = threshold

    def load_data(self):
        df = pd.read_csv(self.data_path, parse_dates=["date"])
        df = df.sort_values("date").reset_index(drop=True)
        df["spread"] = df["bond1_price"] - df["bond2_price"]
        df["rolling_mean"] = df["spread"].rolling(self.window, min_periods=1).mean()
        df["rolling_std"] = df["spread"].rolling(self.window, min_periods=1).std()
        self.df = df

    def backtest(self):
        df = self.df.copy()
        pos = 0
        positions = []
        for i, row in df.iterrows():
            if i == 0:
                positions.append(0)
                continue
            mean, std, spread = row["rolling_mean"], row["rolling_std"], row["spread"]
            upper, lower = mean + self.threshold * std, mean - self.threshold * std
            if pos == 0:
                if spread > upper:
                    pos = -1
                elif spread < lower:
                    pos = 1
            else:
                if (pos == 1 and spread >= mean) or (pos == -1 and spread <= mean):
                    pos = 0
            positions.append(pos)
        df["position"] = positions
        df["spread_diff"] = df["spread"].diff().fillna(0)
        # Use previous day's position for today's PnL
        df["pnl"] = df["position"].shift(1).fillna(0) * df["spread_diff"]
        df["cum_pnl"] = df["pnl"].cumsum()
        self.df = df

    def performance(self):
        df = self.df
        total = df["cum_pnl"].iloc[-1]
        daily = df["pnl"]
        ann_return = daily.mean() * TRADING_DAYS
        ann_vol = daily.std() * np.sqrt(TRADING_DAYS)
        sharpe = ann_return / ann_vol if ann_vol else np.nan
        max_dd = (df["cum_pnl"].cummax() - df["cum_pnl"]).max()
        return {"Total PnL": total, "Annualized Return": ann_return,
                "Annualized Volatility": ann_vol, "Sharpe Ratio": sharpe,
                "Maximum Drawdown": max_dd}

    def plot(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.df["date"], self.df["cum_pnl"], label="Cumulative PnL", color="blue")
        plt.xlabel("Date")
        plt.ylabel("Cumulative PnL")
        plt.title("Bond Arbitrage Strategy Backtest")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def main():
    # expects processed data at data/processed/processed_bond_data.csv
    data_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'processed_bond_data.csv')
    backtester = BondArbBacktest(data_file)
    backtester.load_data()
    backtester.backtest()
    perf = backtester.performance()
    
    print("Performance Metrics:")
    for key, value in perf.items():
        print(f"  {key}: {value:.2f}")
    
    backtester.plot()

if __name__ == '__main__':
    main()
