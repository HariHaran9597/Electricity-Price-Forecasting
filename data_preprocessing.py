import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_data(file_paths):
    """Load and combine data from multiple CSV files."""
    dfs = []
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def preprocess_data(df):
    """Clean and preprocess the data."""
    # Convert timestamp to datetime
    df['datetime_beginning_ept'] = pd.to_datetime(df['datetime_beginning_ept'])
    
    # Sort by timestamp
    df = df.sort_values('datetime_beginning_ept')
    
    # Select only numeric price column and convert to float
    df['price'] = pd.to_numeric(df['total_lmp_da'], errors='coerce')
    
    # Add time-based features
    df['hour'] = df['datetime_beginning_ept'].dt.hour
    df['day_of_week'] = df['datetime_beginning_ept'].dt.dayofweek
    df['month'] = df['datetime_beginning_ept'].dt.month
    
    # Drop any rows with missing values
    df = df.dropna(subset=['price'])
    
    # Create lag features after resampling to avoid memory issues
    return df[['datetime_beginning_ept', 'price', 'hour', 'day_of_week', 'month']]
    
    # Create lag features
    for i in range(1, 25):
        df[f'price_lag_{i}'] = df['price'].shift(i)
    
    # Drop rows with NaN values
    df = df.dropna()
    
    return df

def create_sequences(data, sequence_length=24):
    """Create sequences for LSTM model."""
    n_samples = len(data) - sequence_length
    n_features = data.shape[1]
    
    # Pre-allocate arrays
    X = np.zeros((n_samples, sequence_length, n_features))
    y = np.zeros(n_samples)
    
    # Fill arrays
    for i in range(n_samples):
        X[i] = data[i:(i + sequence_length)]
        y[i] = data[i + sequence_length, 0]
    
    return X, y

def prepare_data(file_paths, test_size=0.2, sample_interval='1H'):
    """Prepare data for model training."""
    # Load and preprocess data
    df = load_data(file_paths)
    df_processed = preprocess_data(df)
    
    # Resample data to reduce memory usage
    df_processed.set_index('datetime_beginning_ept', inplace=True)
    df_processed = df_processed.resample(sample_interval).mean().dropna()
    df_processed.reset_index(inplace=True)
    
    # Create lag features
    for i in range(1, 25):
        df_processed[f'price_lag_{i}'] = df_processed['price'].shift(i)
    
    # Drop rows with NaN values from lag creation
    df_processed = df_processed.dropna()
    
    # Select features
    features = ['price'] + [f'price_lag_{i}' for i in range(1, 25)] + ['hour', 'day_of_week', 'month']
    data = df_processed[features].values
    
    # Scale data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Create sequences
    X, y = create_sequences(scaled_data)
    
    # Split into train and test sets
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    return X_train, X_test, y_train, y_test, scaler
    
    return X_train, X_test, y_train, y_test, scaler