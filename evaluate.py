import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import load_model
from data_preprocessing import prepare_data
from glob import glob
import os

def evaluate_model():
    """Evaluate model performance and make 24-hour predictions."""
    # Create results directory if it doesn't exist
    results_dir = os.path.join('results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Load data
    data_dir = os.path.join('Dataset')
    file_paths = glob(os.path.join(data_dir, '*.csv'))
    X_train, X_test, y_train, y_test, scaler = prepare_data(file_paths)
    
    # Load model
    model = load_model('price_forecasting_model.h5')
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Inverse transform predictions and actual values
    dummy = np.zeros((len(y_test), X_test.shape[2]))
    dummy[:, 0] = y_pred.flatten()
    y_pred = scaler.inverse_transform(dummy)[:, 0]
    
    dummy[:, 0] = y_test
    y_test = scaler.inverse_transform(dummy)[:, 0]
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    # Save metrics to file
    metrics = {
        'Mean Absolute Error': f'${mae:.2f}',
        'Root Mean Squared Error': f'${rmse:.2f}',
        'Mean Absolute Percentage Error': f'{mape:.2f}%'
    }
    
    with open(os.path.join(results_dir, 'model_metrics.txt'), 'w') as f:
        f.write('Model Performance Metrics:\n')
        for metric, value in metrics.items():
            f.write(f'{metric}: {value}\n')
    
    print('\nModel Performance Metrics:')
    for metric, value in metrics.items():
        print(f'{metric}: {value}')
    
    # Plot recent predictions
    plot_predictions(y_test[-100:], y_pred[-100:], 'Last 100 Hours - Actual vs Predicted Prices', results_dir)
    
    # Predict next 24 hours
    last_sequence = X_test[-1]
    next_24_hours = []
    
    for _ in range(24):
        # Make prediction
        pred = model.predict(last_sequence.reshape(1, 24, -1))
        next_24_hours.append(pred[0, 0])
        
        # Update sequence
        last_sequence = np.roll(last_sequence, -1, axis=0)
        last_sequence[-1, 0] = pred[0, 0]
    
    # Inverse transform predictions
    dummy = np.zeros((24, X_test.shape[2]))
    dummy[:, 0] = next_24_hours
    next_24_hours_inv = scaler.inverse_transform(dummy)[:, 0]
    
    # Save predictions to CSV
    predictions_df = pd.DataFrame({
        'Hour': range(1, 25),
        'Predicted_Price': next_24_hours_inv
    })
    predictions_df.to_csv(os.path.join(results_dir, '24_hour_forecast.csv'), index=False)
    
    # Plot and print predictions
    plot_forecast(next_24_hours_inv, results_dir)
    
    return next_24_hours_inv

def plot_predictions(y_true, y_pred, title='Model Predictions vs Actual Values', results_dir=None):
    """Plot predicted vs actual values."""
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='Actual', alpha=0.7)
    plt.plot(y_pred, label='Predicted', alpha=0.7)
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True)
    
    if results_dir:
        plt.savefig(os.path.join(results_dir, 'actual_vs_predicted.png'))
    plt.show()

def plot_forecast(predictions, results_dir=None, title='24-Hour Price Forecast'):
    """Plot price forecast for next 24 hours."""
    plt.figure(figsize=(12, 6))
    plt.plot(range(24), predictions, marker='o')
    plt.title(title)
    plt.xlabel('Hour')
    plt.ylabel('Predicted Price ($)')
    plt.grid(True)
    
    if results_dir:
        plt.savefig(os.path.join(results_dir, '24_hour_forecast.png'))
    plt.show()
    
    # Print and save hourly predictions
    print('\nHourly Price Predictions for Next 24 Hours:')
    with open(os.path.join(results_dir, 'hourly_predictions.txt'), 'w') as f:
        f.write('Hourly Price Predictions for Next 24 Hours:\n')
        for hour, price in enumerate(predictions, 1):
            prediction_line = f'Hour {hour:2d}: ${price:.2f}'
            print(prediction_line)
            f.write(prediction_line + '\n')

if __name__ == '__main__':
    print('Evaluating model and generating predictions...')
    next_24_hours = evaluate_model()