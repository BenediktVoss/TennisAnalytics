import numpy as np
import matplotlib.pyplot as plt
import cv2
import numpy as np


def transform_points_to_model(points, matrix):
    transformed_points = []
    for point in points:
        # Convert the point to the required format for perspectiveTransform
        input_point = np.array([[[point[0], point[1]]]], dtype=np.float32)
        
        # Transform the point using the homography matrix
        model_point = cv2.perspectiveTransform(input_point, matrix)
        
        # Extract the x and y from the transformed result
        x, y = model_point[0][0][0], model_point[0][0][1]
        transformed_points.append((int(x), int(y)))  # Convert to integer for pixel coordinates
    
    return transformed_points


def draw_points_on_model(model_court, points, point_color=(255, 0, 0), point_radius=10):
    # Make a copy of the model court image to draw on
    model_court_copy = model_court.copy()
    for i, point in enumerate(points):
        x, y = point
        # Check bounds to ensure we don’t draw outside the image
        if 0 <= x < model_court_copy.shape[1] and 0 <= y < model_court_copy.shape[0]:
            # Draw a circle at each point
            cv2.circle(model_court_copy, (x, y), point_radius, point_color, -1)
    
    return model_court_copy


def get_ordered_keypoints(keypoints):
    mp_dict = {
        'top_left_corner': [352,602],
        'top_left_singles': [484,602],
        'top_right_singles': [1312,602],
        'top_right_corner': [1444,602],
        'bottom_left_corner': [352,2974],
        'bottom_left_singles': [484,2974],
        'bottom_right_singles': [1312,2974],
        'bottom_right_corner': [1444,2974],
        'service_top_left': [484,1150],
        'service_top_right': [1312,1150],
        'service_bottom_left': [484,2425],
        'service_bottom_right': [1312,2425],
        'service_center_top': [898,1150],
        'service_center_bottom': [898,2425],
    }

    keypoint_predictions = []
    keypoints_model = []

    for idx, keypoint in enumerate(mp_dict.keys()):
        if not np.isnan(keypoints[idx][0]):
            keypoint_predictions.append(keypoints[idx])
            keypoints_model.append((mp_dict[keypoint][0], mp_dict[keypoint][1]))

    #convert to numpy array
    keypoint_predictions = np.array(keypoint_predictions)
    keypoints_model = np.array(keypoints_model)

    return keypoint_predictions, keypoints_model


def get_homography_for_keypoints(keypoints, method=0):

    # get existing keypoints
    keypoint_predictions, keypoints_model = get_ordered_keypoints(keypoints)

    # Compute the homography matrix using RANSAC for robustness
    keypoint_matrix, status = cv2.findHomography(keypoint_predictions, keypoints_model, method)

    return keypoint_matrix

     
# perform homography
def calculate_transformed_position(matrix, positions):
    # transform 
    transformed_positions = transform_points_to_model(positions, matrix)

    return transformed_positions


if __name__ == "__main__":
    predictions = [(522, 323), (np.nan, np.nan), (703, 325), (729, 322), (255, 449), (364, 450), (969, 450), (1065, 449), (535, 332), (720, 332), (465, 382), (827, 382), (627, 332), (650, 381), (635, 350)]

    # load image
    model_court = cv2.imread('models/tennis_court_with_precise_lines.png')

    # calculate homography
    keypoint_matrix = get_homography_for_keypoints(predictions)

    # transform the points
    transformed_positions = calculate_transformed_position(keypoint_matrix, predictions)

    # draw the points on the model
    model_with_points = draw_points_on_model(model_court, transformed_positions)

    # Plot the model court with points
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(model_with_points, cv2.COLOR_BGR2RGB))
    plt.axis("off")  # Hide axis
    plt.show()
