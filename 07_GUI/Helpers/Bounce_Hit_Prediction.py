import pandas as pd
import numpy as np
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler
import joblib

def calculate_Features(df):
    for index in range(1, len(df)-1):
        current_position = (df.loc[index, 'x'], df.loc[index, 'y'])
        previous_position = (df.loc[index-1, 'x'], df.loc[index-1, 'y'])

        df.loc[index, 'velocity'] = calculateVelocity(current_position, previous_position)
        df.loc[index, 'direction'] = calculateDirection(current_position, previous_position)
        df.loc[index, 'slope'] = calculateSlope(current_position, previous_position)

    return df


def calculateVelocity(current_position, previous_position):
    # check if value is missing
    if pd.isnull(current_position[0]) or pd.isnull(previous_position[0]):
        return -1

    distane_travelled = distance.euclidean(current_position, previous_position)

    return distane_travelled


def calculateDirection(current_position, previous_position):
   # check if value is missing
    if pd.isnull(current_position[0]) or pd.isnull(previous_position[0]):
        return -1
    
    x1, y1 = previous_position
    x2, y2 = current_position

    # Calculate the change in x and y
    delta_x = x2 - x1
    delta_y = y2 - y1

    # Compute the angle using atan2
    angle = np.arctan2(delta_y, delta_x)

    # Convert the angle to degrees and normalize to 0–360
    direction = (np.degrees(angle) + 360) % 360

    return direction

def calculateSlope(current_position, previous_position):
   # check if value is missing
    if pd.isnull(current_position[0]) or pd.isnull(previous_position[0]):
        return -1

    x1, y1 = previous_position
    x2, y2 = current_position

    slope = (y2 - y1) / (x2 - x1) if x2 - x1 != 0 else 0
    return slope

def preprocess_df(ball_positions):
    # convert ball positions to dataframe
    df = pd.DataFrame(ball_positions, columns=['x', 'y'])

    df = calculate_Features(df)

    return df

def convert_to_time_series_data_dynamic(input_df, window_size=5, features=['x', 'y', 'velocity', 'direction']):
    df = input_df.copy()
    rows = []

    for index, row in df.iterrows():
        if index <= window_size or index > len(df) - window_size:
            continue

        previous_frames = {}
        for i in range(window_size, 1, -1):
            lag_stamp = 'previous_' + str(i) + '_'
            for feature in features:
                previous_frames[lag_stamp + feature] = df.loc[index-i, feature]

        current_frame = {}
        for feature in features:
            current_frame[feature] = df.loc[index, feature]

        future_frames = {}
        for i in range(1, window_size):
            lag_stamp = 'following_' + str(i) + '_'
            for feature in features:
                future_frames[lag_stamp + feature] = df.loc[index+i, feature]

        series = {
            **current_frame,
            **previous_frames,
            **future_frames
        }
        rows.append(series)

    return pd.DataFrame(rows)

def bounce_and_hit_inference_XGBoost(ball_positions):
    df = preprocess_df(ball_positions)

    window_size = 10
    features = ['x', 'y', 'velocity', 'direction']

    # Convert the data to time series
    df_time_series = convert_to_time_series_data_dynamic(df, window_size=window_size, features=features)
    x_columns = features + ['previous_' + str(i) + '_' + feature for i in range(window_size, 1, -1) for feature in features] + ['following_' + str(i) + '_' + feature for i in range(1, window_size) for feature in features]
    Input = df_time_series[x_columns]

    scaler = joblib.load("models/scaler.pkl")
    Input = scaler.fit_transform(Input)

    # load XGBModel
    loaded_model = joblib.load("models/BounceHit_best_xgboost.pkl")

    # Predict on the validation set
    predictions = loaded_model.predict(Input)

    # add leading and training 0 for the first window size predictions
    predictions = np.concatenate([np.zeros(window_size), predictions])
    predictions = np.concatenate([predictions, np.zeros(window_size)])


    return predictions