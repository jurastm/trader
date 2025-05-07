import pandas as pd
import numpy as np

def load_and_preprocess_data(file_path):
    """
    Loads historical Bitcoin price data from a CSV file.
    Expected CSV columns: unix, date, symbol, open, high, low, close, Volume BTC, Volume USD
    """
    df = pd.read_csv(file_path)
    # Convert 'date' to datetime objects
    df['date'] = pd.to_datetime(df['date'])
    # Sort in chronological order
    df = df.sort_values('date').reset_index(drop=True)
    # Forward fill missing values
    # df.fillna(method='ffill', inplace=True)
    df.ffill(inplace=True)
    # Ensure the 'close' column is numeric
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df.dropna(subset=['close'], inplace=True)
    return df


def fft(inputs, use_log_returns=True, window_size=64):
    if use_log_returns:
        # For window_size data points, you have (window_size - 1) returns:
        returns = np.log(inputs[1:] / np.maximum(inputs[:-1], 1e-8))
    else:
        # Simple difference: p_t - p_{t-1}
        returns = inputs[1:] - inputs[:-1]
        
    
    returns -= np.mean(returns)
    padded_returns = np.pad(returns, (0, 1), 'constant', constant_values=0)
    
    
    fft_result = np.fft.fft(padded_returns, n=window_size)
    magnitudes = np.abs(fft_result)
    phases = np.angle(fft_result)

    return magnitudes, phases, fft_result