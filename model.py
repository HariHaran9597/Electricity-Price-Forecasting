import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def create_model(sequence_length, n_features, lstm_units=50, dropout_rate=0.2):
    """Create and compile LSTM model for time series forecasting."""
    model = Sequential([
        LSTM(lstm_units, activation='relu', input_shape=(sequence_length, n_features), 
             return_sequences=True),
        Dropout(dropout_rate),
        LSTM(lstm_units, activation='relu'),
        Dropout(dropout_rate),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(), metrics=['mae'])
    return model

def train_model(model, X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
    """Train the LSTM model with early stopping."""
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=1
    )
    
    return history

def predict_next_24_hours(model, recent_data, scaler):
    """Make predictions for the next 24 hours."""
    # Scale the recent data
    scaled_data = scaler.transform(recent_data)
    
    # Reshape for prediction
    X = scaled_data.reshape(1, 24, recent_data.shape[1])
    
    # Make prediction
    pred_scaled = model.predict(X)
    
    # Inverse transform prediction
    dummy = np.zeros((1, recent_data.shape[1]))
    dummy[:, 0] = pred_scaled.flatten()
    predictions = scaler.inverse_transform(dummy)[:, 0]
    
    return predictions