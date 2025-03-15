import os
from glob import glob
from data_preprocessing import prepare_data
from model import create_model, train_model

def main():
    # Get all CSV files
    data_dir = os.path.join('Dataset')
    file_paths = glob(os.path.join(data_dir, '*.csv'))
    
    # Prepare data
    print('Preparing data...')
    X_train, X_test, y_train, y_test, scaler = prepare_data(file_paths)
    
    # Create and train model
    print('\nCreating model...')
    sequence_length = 24
    n_features = X_train.shape[2]
    model = create_model(sequence_length, n_features)
    
    print('\nTraining model...')
    history = train_model(model, X_train, y_train, X_test, y_test)
    
    # Save model
    model.save('price_forecasting_model.h5')
    print('\nModel saved as price_forecasting_model.h5')

if __name__ == '__main__':
    main()