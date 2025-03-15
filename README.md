# Electricity Price Forecasting

This project implements a deep learning model for forecasting electricity prices over a 24-hour horizon using historical price data. The model utilizes LSTM (Long Short-Term Memory) neural networks to capture temporal dependencies in electricity price patterns.

## Project Overview

The system is designed to:
- Predict electricity prices for the next 24 hours
- Analyze historical price patterns
- Provide detailed performance metrics and visualizations
- Support decision-making in energy trading and planning

## Features

- 24-hour ahead price forecasting
- Data preprocessing and normalization
- Model training with LSTM architecture
- Comprehensive evaluation metrics
- Visualization of predictions and actual prices
- Hourly price forecasts export

## Installation

1. Clone this repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Data Requirements

Place your electricity price data CSV files in the `Dataset` folder. The data should include hourly electricity prices in a structured format.

## Usage

### Training the Model

To train the model on your dataset:

```bash
python train.py
```

This will:
- Process the data from the Dataset folder
- Train the LSTM model
- Save the trained model as 'price_forecasting_model.h5'

### Evaluating and Forecasting

To evaluate the model and generate predictions:

```bash
python evaluate.py
```

This will:
- Load the trained model
- Generate performance metrics
- Create visualizations
- Produce 24-hour price forecasts

## Output

The evaluation process generates several outputs in the `results` directory:

- `model_metrics.txt`: Performance metrics including MAE, RMSE, and MAPE
- `24_hour_forecast.csv`: Detailed hourly price predictions
- `actual_vs_predicted.png`: Visualization comparing actual vs predicted prices
- `24_hour_forecast.png`: Plot of the 24-hour price forecast
- `hourly_predictions.txt`: Formatted hourly price predictions

## Model Architecture

The model uses a sequence-to-sequence LSTM architecture designed to capture both short-term and long-term patterns in electricity price data. It processes 24-hour sequences to predict the next hour's price.

## Performance

The model's performance is evaluated using multiple metrics:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)

Detailed performance metrics are available in the results directory after evaluation.

## Contributing

Contributions to improve the model's performance or add new features are welcome. Please feel free to submit pull requests or open issues for discussion.

## License

This project is open-source and available under the MIT License.