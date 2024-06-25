import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data(lane1_file, lane2_file):
    lane1_data = pd.read_csv(lane1_file)
    lane2_data = pd.read_csv(lane2_file)

    lane1_data['Date'] = pd.to_datetime(lane1_data['Date'], dayfirst=True)
    lane2_data['Date'] = pd.to_datetime(lane2_data['Date'], dayfirst=True)

    label_encoder = LabelEncoder()
    for column in lane1_data.columns[1:]:
        lane1_data[column] = label_encoder.fit_transform(lane1_data[column])
        lane2_data[column] = label_encoder.fit_transform(lane2_data[column])
    
    return lane1_data, lane2_data, label_encoder

def prepare_lstm_data(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i+n_steps])
        y.append(data[i+n_steps])
    return np.array(X), np.array(y)

def build_and_train_lstm(X, y, n_steps, n_features):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    model.fit(X, y, epochs=200, verbose=0)
    return model

def make_predictions(model, X, label_encoder):
    predictions = model.predict(X, verbose=0).round().astype(int).flatten()
    
    print("Unique prediction values:", np.unique(predictions))
    
    known_labels = label_encoder.classes_
    predictions = np.clip(predictions, 0, len(known_labels) - 1)
    
    predicted_labels = label_encoder.inverse_transform(predictions)
    return predicted_labels

lane1_data, lane2_data, label_encoder = load_and_preprocess_data('lane1_traffic_density.csv', 'lane2_traffic_density.csv')

n_steps = 24

lane1_values = lane1_data.iloc[:, 1:].values.flatten()
lane2_values = lane2_data.iloc[:, 1:].values.flatten()

X_lane1, y_lane1 = prepare_lstm_data(lane1_values, n_steps)
X_lane2, y_lane2 = prepare_lstm_data(lane2_values, n_steps)

n_features = 1
X_lane1 = X_lane1.reshape((X_lane1.shape[0], X_lane1.shape[1], n_features))
X_lane2 = X_lane2.reshape((X_lane2.shape[0], X_lane2.shape[1], n_features))

model_lane1 = build_and_train_lstm(X_lane1, y_lane1, n_steps, n_features)
model_lane2 = build_and_train_lstm(X_lane2, y_lane2, n_steps, n_features)

predicted_lane1 = make_predictions(model_lane1, X_lane1, label_encoder)
predicted_lane2 = make_predictions(model_lane2, X_lane2, label_encoder)

start_date_lane1 = lane1_data['Date'].iloc[n_steps]
start_date_lane2 = lane2_data['Date'].iloc[n_steps]

predicted_dates_lane1 = [start_date_lane1 + pd.DateOffset(hours=i) for i in range(len(predicted_lane1))]
predicted_dates_lane2 = [start_date_lane2 + pd.DateOffset(hours=i) for i in range(len(predicted_lane2))]

predicted_lane1_df = pd.DataFrame({'Date': predicted_dates_lane1, 'Predicted Density': predicted_lane1})
predicted_lane2_df = pd.DataFrame({'Date': predicted_dates_lane2, 'Predicted Density': predicted_lane2})

predicted_lane1_df.to_csv('predicted_lane1_traffic_density.csv', index=False)
predicted_lane2_df.to_csv('predicted_lane2_traffic_density.csv', index=False)

print("Predictions saved to CSV files:")
print("predicted_lane1_density.csv")
print("predicted_lane2_density.csv")


  
