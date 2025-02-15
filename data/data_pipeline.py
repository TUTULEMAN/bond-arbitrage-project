from src.utils.common_imports import yf, pd, os

#when try running, ModuleNotFoundError: No module named 'src'
#we obv have that but somehow its not compiling

def fetch_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    return df[['Adj Close']].rename(columns={'Adj Close': 'price'})

def main():
    start_date = "2010-01-01"
    end_date = "2025-01-01"
    
    # Tickers for data sources:
    # IEF: iShares 7-10 Year Treasury Bond ETF (proxy for 10-year US bond prices)
    # ZN=F: 10-Year US Treasury Note Futures on Yahoo Finance
    bond_ticker = "IEF"
    futures_ticker = "ZN=F"
    
    # Fetching info
    bond_data = fetch_data(bond_ticker, start_date, end_date)
    futures_data = fetch_data(futures_ticker, start_date, end_date)
    
    # Renaming columns
    # 'bond1_price' for the bond proxy 
    # 'bond2_price' for the futures contract.
    bond_data = bond_data.rename(columns={'price': 'bond1_price'})
    futures_data = futures_data.rename(columns={'price': 'bond2_price'})
    
    df = pd.merge(bond_data, futures_data, left_index=True, right_index=True, how='inner')
    
    df = df.reset_index().rename(columns={'Date': 'date'})
    df['date'] = pd.to_datetime(df['date'])
    
    # Define the output directory and file path
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'processed_bond_data.csv')
    
    # Save the DataFrame to CSV
    df.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")

if __name__ == '__main__':
    main()
