import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
import random
from scipy.interpolate import CubicSpline
from scipy.spatial import distance

def fill_gaps_with_spline(ball_positions, frame_width, frame_height, window_size=4, max_gap_length=5):
    # Prepare x and y positions, replacing None with NaN
    x_positions = np.array([pos[0] if pos is not None else np.nan for pos in ball_positions], dtype=float)
    y_positions = np.array([pos[1] if pos is not None else np.nan for pos in ball_positions], dtype=float)

    # Remove points outside the frame
    valid_in_frame = (
        (x_positions >= 0) & (x_positions <= frame_width) & 
        (y_positions >= 0) & (y_positions <= frame_height)
    )
    x_positions[~valid_in_frame] = np.nan
    y_positions[~valid_in_frame] = np.nan

    # Time (frame indices)
    time = np.arange(len(ball_positions))

    # Helper function to find gaps
    def find_gaps(data):
        is_nan = np.isnan(data)
        gaps = []
        start = None
        for i, nan in enumerate(is_nan):
            if nan and start is None:
                start = i
            elif not nan and start is not None:
                gaps.append((start, i))
                start = None
        if start is not None:
            gaps.append((start, len(data)))
        return gaps

    # Identify gaps in the data
    gaps = find_gaps(x_positions)

    # Fill each gap with an enlarged time window
    for start, end in gaps:
        gap_length = end - start
        # Only process gaps smaller than or equal to max_gap_length
        if gap_length <= max_gap_length:
            # Define the interpolation range: Include more points before and after the gap
            interp_start = max(0, start - window_size)  # Include window_size points before
            interp_end = min(len(x_positions), end + window_size)  # Include window_size points after

            # Valid indices for interpolation
            valid_range = time[interp_start:interp_end][~np.isnan(x_positions[interp_start:interp_end])]
            valid_x = x_positions[interp_start:interp_end][~np.isnan(x_positions[interp_start:interp_end])]
            valid_y = y_positions[interp_start:interp_end][~np.isnan(y_positions[interp_start:interp_end])]

            if len(valid_x) >= 3:  # Need at least 3 points for cubic spline
                # Fit splines for x and y
                spline_x = CubicSpline(valid_range, valid_x, bc_type='natural')
                spline_y = CubicSpline(valid_range, valid_y, bc_type='natural')

                # Fill the gap
                gap_range = time[start:end]
                x_positions[start:end] = spline_x(gap_range)
                y_positions[start:end] = spline_y(gap_range)

    # Convert back to original positions
    filled_positions = np.column_stack((x_positions, y_positions))
    
    return filled_positions

def detect_outliers(ball_positions, max_change = 200, threshold = 50, window_size = 5):

    # Check for valid input
    if len(ball_positions) < window_size:
        raise ValueError("Number of positions must be greater than the sliding window size.")

    # Method 1: Positional change check
    large_change_indexes = positional_change_check(ball_positions, max_change)

    # remove the detected outliers
    ball_positions = [pos if i not in large_change_indexes else (None, None) for i, pos in enumerate(ball_positions)]

    # Method 2: Cubic spline sliding window
    spline_outlier_indexes = cubic_spline_outlier_detection(ball_positions, threshold, window_size)

    # Combine results from both methods
    outliers = np.unique(np.concatenate([large_change_indexes, spline_outlier_indexes]))

    return outliers

def positional_change_check(ball_positions, max_change):
    outlier_indexes = []
    #iterate over the ball positions and calculate the euclidean distance between each pair of consecutive positions
    for i in range(1, len(ball_positions)):
        #calculate the euclidean distance between the current position and the previous position
        current_position = ball_positions[i]
        previous_position = ball_positions[i-1]
        #check if both have a value
        if current_position[0] is not None and current_position[1] is not None and previous_position[0] is not None and previous_position[1] is not None:

            distance = np.linalg.norm(np.array(ball_positions[i]) - np.array(ball_positions[i-1]))
            #if the distance is greater than the max_change threshold add to the outlier indexes
            if distance > max_change:
                outlier_indexes.append(i)

    # if i have indexes with a gap of 1 in between liek 166 and 168 add 167 to the outlier indexes
    for i in range(1, len(outlier_indexes)):
        if outlier_indexes[i] - outlier_indexes[i-1] == 2:
            outlier_indexes.append(outlier_indexes[i] - 1)

    filtered_outliers = []

    # Iterate and check for three consecutive outliers
    for i in range(len(outlier_indexes)):
        if i >= 2 and outlier_indexes[i] - outlier_indexes[i-1] == 1 and outlier_indexes[i-1] - outlier_indexes[i-2] == 1:
            # Skip the current outlier if it's part of a sequence of three
            continue
        filtered_outliers.append(outlier_indexes[i])

    outlier_indexes = filtered_outliers

    return outlier_indexes

def predict_next_position_cubic(positions):
    # Extract x and y coordinates, replacing None with np.nan for consistency
    x_positions = np.array([pos[0] if pos[0] is not None else np.nan for pos in positions], dtype=float)
    y_positions = np.array([pos[1] if pos[1] is not None else np.nan for pos in positions], dtype=float)

    # Time (frame indices)
    time = np.arange(len(positions))

    # Count the number of NaN values
    num_nan_x = np.isnan(x_positions).sum()
    num_nan_y = np.isnan(y_positions).sum()

    # Check if more than half the points are NaN
    if num_nan_x > len(positions) // 2 or num_nan_y > len(positions) // 2:
        return (np.nan, np.nan), False # Too many missing values

    # Interpolate missing values with linear interpolation
    x_positions = np.interp(time, time[~np.isnan(x_positions)], x_positions[~np.isnan(x_positions)])
    y_positions = np.interp(time, time[~np.isnan(y_positions)], y_positions[~np.isnan(y_positions)])

    # Fit cubic splines
    spline_x = CubicSpline(time, x_positions, bc_type='natural')
    spline_y = CubicSpline(time, y_positions, bc_type='natural')

    # Predict the next position
    next_time = time[-1] + 1
    next_x = spline_x(next_time)
    next_y = spline_y(next_time)

    return (next_x, next_y), True

def cubic_spline_outlier_detection(ball_positions, threshold, window_size):
    n_positions = len(ball_positions)
    outlier_indexes = []

    for i in range(n_positions):
        # Skip boundary positions where we can't apply the sliding window
        if i < window_size or i >= n_positions - window_size:
            continue

        # If the current position is None, skip
        if ball_positions[i][0] is None or ball_positions[i][1] is None:
            continue

        # Fit cubic splines for preceding and subsequent windows
        prev_indices = range(i - window_size, i)
        next_indices = range(i + 1, i + 1 + window_size)

        # Filter valid positions for both windows
        positions_before = np.array([ball_positions[idx] for idx in prev_indices])
        positions_after = np.array([ball_positions[idx] for idx in next_indices])

        # reverse the positions after 
        positions_after = positions_after[::-1]

        # predict the next position using the cubic spline
        prediction_prev, success_prev = predict_next_position_cubic(positions_before)
        predictions_after, success_after = predict_next_position_cubic(positions_after)

        # If either prediction fails, skip
        if not success_prev or not success_after:
            continue

        # Convert current position to numpy array
        current_position = np.array(ball_positions[i])

        # Calculate Euclidean distances
        dist_prev = np.linalg.norm(prediction_prev - current_position)
        dist_next = np.linalg.norm(predictions_after - current_position)

        # Use the smaller of the two distances
        distance = min(dist_prev, dist_next)

        # If the distance is greater than the threshold, mark as an outlier
        if distance > threshold:
            outlier_indexes.append(i)

    return outlier_indexes

# pipeline 
def smoothen_trajectory(ball_positions, frame_width, frame_height, max_change=200, threshold=50, window_size=5, max_gap_length=3):
    # Detect outliers
    outliers = detect_outliers(ball_positions, max_change, threshold, window_size)

    # Replace outliers with None
    ball_positions = [pos if i not in outliers else (None, None) for i, pos in enumerate(ball_positions)]


    # Fill gaps with cubic spline interpolation
    ball_positions = fill_gaps_with_spline(ball_positions, frame_width, frame_height, window_size, max_gap_length)

    return ball_positions

