import gradio as gr
import pandas as pd
import numpy as np
import json
import cv2
import os
import numpy as np
from Helpers import Inference_Ball_TrackNet
from Helpers.Smoothing import smoothen_trajectory
from Helpers.Bounce_Hit_Prediction import bounce_and_hit_inference_XGBoost
from Helpers.Inference_Court_Detection import inference_ball_tracknet
from Helpers.Inference_Player import get_player_predictions
from Helpers.Homography import get_homography_for_keypoints, calculate_transformed_position, draw_points_on_model 
import copy
import random
import time
import shutil


def split_video_into_frames(video_file, temp_folder="_temp"):
    # Load the video using OpenCV
    video = cv2.VideoCapture(video_file.name)

    # get fps information
    fps = video.get(cv2.CAP_PROP_FPS)

    # generate uuid
    uuid = str(random.randint(100000, 999999))
    
    # Iterate over all frames in the video
    frame_number = 0
    while True:
        # Read the next frame
        success, frame = video.read()
        
        # If the frame was not successfully read, we have reached the end of the video
        if not success:
            break
        
        # save the frame as a jpg file
        frame_path = os.path.join(temp_folder, uuid, str(frame_number) + ".jpg")
        os.makedirs(os.path.dirname(frame_path), exist_ok=True)
        # resize to 1280x720
        frame = cv2.resize(frame, (1280, 720))
        cv2.imwrite(frame_path, frame)

        
        # Increment the frame number
        frame_number += 1
    
    # Release the video object
    video.release()

    print(f"Video split into {frame_number} frames")
    print(f"Frames saved in {temp_folder}/{uuid}")
    
    return frame_number, fps, uuid

    
def extract_frame(path, frame_number):
    # Construct the full path to the frame
    frame_path = os.path.join(path, str(frame_number) + ".jpg")
    
    # Check if the file exists
    if not os.path.exists(frame_path):
        raise FileNotFoundError(f"The frame {frame_number} does not exist")
    
    # Load the frame using OpenCV
    frame = cv2.imread(frame_path)
    
    # Return the frame as a NumPy array
    return frame

def hex_to_bgr(hex_color):
    # Remove the '#' if it exists
    hex_color = hex_color.lstrip('#')
    
    # Convert hex to RGB, then reverse to get BGR for OpenCV
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    bgr = rgb[::-1]  # Reverse the tuple for BGR
    return bgr

# function to draw all players in a frame
def draw_players(frame, players, color):
    # Create a copy of the frame
    frame_copy = frame.copy()
    
    # Draw a bounding box for each player
    for player in players:
        # Get the coordinates of the bounding box round float to int
        x1 = int(float(player['xtl']))
        y1 = int(float(player['ytl']))
        x2 = int(float(player['xbr']))
        y2 = int(float(player['ybr']))
        
        # Draw the bounding box
        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)

    return frame_copy

def draw_balls(frame, balls, color, fill=True, outline=True):
    # Create a copy of the frame
    frame_copy = frame.copy()
    
    # Draw a bounding box for each ball
    for ball in balls:
        # Get the coordinates of the bounding box round float to int
        x = ball[0]
        y = ball[1]
        
        # Draw the bounding box
        if fill:
            cv2.circle(frame_copy, (x, y), 4, color, -1, lineType=cv2.LINE_AA)
        # Draw a larger circle around the ball in black
        if outline:
            cv2.circle(frame_copy, (x, y), 7, (0, 0, 0), 1, lineType=cv2.LINE_AA)

    return frame_copy

def draw_keypoints(frame, keypoints, color, alpha=0.8):
    # Create a copy of the frame
    frame_copy = frame.copy()

    # Draw circles for keypoints directly on the overlay
    for value in keypoints:
        if not np.isnan(value[0]):
            # Convert float to integer coordinates
            x, y = map(int, map(float, value))
            # Draw the keypoint on the overlay
            cv2.circle(frame_copy, (x, y), 3, color, -1, lineType=cv2.LINE_AA)

    return frame_copy

def convert_player_positions_to_points(player_positions):
    positions = []
    for player in player_positions:
        # Calculate bottom center
        bottom_center_x = (player['xtl'] + player['xbr']) // 2
        # moove the bottom point 4 px up for better visualization
        bottom_center_y = player['ybr'] - 4

        positions.append((bottom_center_x, bottom_center_y))
    return positions


def generate_minimap(homography_matrix, player_positions, ball_position, trajectory_class, detected_bounces):
    # load image
    model_court = cv2.imread('models/tennis_court_with_precise_lines_thicker.png')

    # Player -----------------------------------------------------------------

    # convert player positions from bounding box to point
    player_positions = convert_player_positions_to_points(player_positions)

    # transform player positions
    player_positions = calculate_transformed_position(homography_matrix, player_positions)

    # draw player positions on model
    model_court = draw_points_on_model(model_court, player_positions, hex_to_bgr("#2800ff"), point_radius=30)

    # Ball -------------------------------------------------------------------

    # show ball position only on trajectory class 2 (Bounce)
    if trajectory_class == 2:
        ball_position = calculate_transformed_position(homography_matrix, [ball_position])
        detected_bounces.append(ball_position[0])
    
    # draw all detected bounces
    model_court = draw_points_on_model(model_court, detected_bounces, hex_to_bgr("#ffff00"), point_radius=30)

    return model_court, detected_bounces

def create_video_from_inference(number_frames, 
                                fps,
                                path,
                                ball_positions, 
                                bounces_and_hits,
                                player_predictions,
                                court_positions, 
                                output_video_path="_temp/output_video.mp4",
                                minimap_enabled=True):
    # Extract the first frame to determine video dimensions
    height, width = (720, 1280)  

    # Initialize the video writer with an MP4-compatible codec
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4 format
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # if ball_positions is longer than frames, cut the ball_positions
    ball_positions = ball_positions[:number_frames]

    # calculate the homography matrix
    homography_matrix = get_homography_for_keypoints(court_positions)
    detected_bounces = []

    # Process and write each frame
    for number in range(number_frames):
        try:
            # Extract the frame using the existing function
            frame = extract_frame(path, number)

            # check if ball position is valid
            if not np.isnan(ball_positions[number][0]):
                # draw circle on the ball position
                ball_position = (int(ball_positions[number][0]), int(ball_positions[number][1]))
                frame = draw_balls(frame, [ball_position], hex_to_bgr("#ffff00"))
            
            # draw court keypoints
            if len(court_positions) > 0:
                frame = draw_keypoints(frame, court_positions, hex_to_bgr("#34eb61"))

            # draw players
            if len(player_predictions[number]) > 0:
                frame = draw_players(frame, player_predictions[number], hex_to_bgr("#2800ff"))

            if minimap_enabled:
                # generate minimap
                minimap, detected_bounces = generate_minimap(homography_matrix, player_predictions[number], ball_positions[number], bounces_and_hits[number], detected_bounces=detected_bounces)

                # Original size of the minimap
                minimap_height, minimap_width, _ = minimap.shape

                # Resize the minimap depending on image width so width of minimat is 10% of the image width
                ratio = max(round(int(width * 0.1) / minimap_width, 2), 0.1)
                minimap = cv2.resize(minimap, (int(minimap_width * ratio), int(minimap_height * ratio)))

                # Add the minimap to the top-right corner of the frame
                frame[0:minimap.shape[0], -minimap.shape[1]:] = minimap

            # Write the annotated frame to the video
            video_writer.write(frame)

        except FileNotFoundError as e:
            print(f"Error processing frame {number}: {e}")
            continue

    # Release the video writer
    video_writer.release()

    return output_video_path


def process_and_display_random_video_inference(video_file):
        # Timing dictionary to store step timings
    timing_info = {}
    start_time = time.time()

    # process the video
    number_frames, fps, uuid = split_video_into_frames(video_file, "_temp")
    path = os.path.join("_temp", uuid)
    timing_info['video_conversion'] = time.time() - start_time

    # Generate random frames
    step_start_time = time.time()

    # Step 1: Perform ball tracking inference
    step_start_time = time.time()
    print("Start ball tracking")
    ball_positions = Inference_Ball_TrackNet.inference_ball_tracknet(path, "models/BallTracking.pt", 1280)
    timing_info['ball_tracking'] = time.time() - step_start_time

    # Step 2: Smoothen trajectory
    step_start_time = time.time()
    print("Start smoothen trajectory")
    ball_positions = smoothen_trajectory(ball_positions, 1280, 720)
    timing_info['smoothen_trajectory'] = time.time() - step_start_time

    # Step 3: Perform bounce and hit prediction
    step_start_time = time.time()
    print("Start bounce and hit")
    bounces_and_hits = bounce_and_hit_inference_XGBoost(ball_positions)
    timing_info['bounce_and_hit'] = time.time() - step_start_time

    # Step 4: Predict court positions
    step_start_time = time.time()
    print("Start court detection")
    court_positions = inference_ball_tracknet(path, "models/CourtTracking.pth", 1280)
    timing_info['court_detection'] = time.time() - step_start_time

    # Step 5: Get player bounding boxes
    step_start_time = time.time()
    print("Start player detection")
    player_predictions = get_player_predictions(model_path="models/YOLOv11_Player.pt", image_path=path)
    timing_info['player_detection'] = time.time() - step_start_time

    # Step 6: Create a video from the frames
    step_start_time = time.time()
    print("Start video creation")
    video_path = create_video_from_inference(number_frames,
                                             fps, 
                                             path,
                                             ball_positions, 
                                             bounces_and_hits, 
                                             player_predictions,
                                             court_positions)
    timing_info['video_creation'] = time.time() - step_start_time

    # Total execution time
    timing_info['total_time'] = time.time() - start_time

    # Generate the inference details string with Markdown table
    inference_details = (
        f"Timing Information:\n"
        f"| Step                       | Time (seconds) |\n"
        f"|----------------------------|----------------|\n"
        f"| Video Conversion           | {timing_info['video_conversion']:.2f} |\n"
        f"| Ball Tracking              | {timing_info['ball_tracking']:.2f} |\n"
        f"| Smoothen Trajectory        | {timing_info['smoothen_trajectory']:.2f} |\n"
        f"| Bounce and Hit Prediction  | {timing_info['bounce_and_hit']:.2f} |\n"
        f"| Court Detection            | {timing_info['court_detection']:.2f} |\n"
        f"| Player Detection           | {timing_info['player_detection']:.2f} |\n"
        f"| Video Creation             | {timing_info['video_creation']:.2f} |\n"
        f"| Total Time                 | {timing_info['total_time']:.2f} |\n"
    )

    print(path)
    # clean up the temp folder
    shutil.rmtree(path)
        
    return video_path, inference_details


with gr.Blocks() as InferenceOwn:
    with gr.Row():
        # Input Section
        with gr.Column(scale=1):
            gr.Markdown("""### 🎥 **Inference on Your Own Data**""")
            gr.Markdown("""
            Upload your own video to perform tennis analytics inference. The video will be processed and annotated with the following features:

            - **Ball Tracking**: Tracks the ball throughout the video.
            - **Player Detection**: Identifies player positions with bounding boxes.
            - **Court Detection**: Overlays court lines for better visualization.
            - **Bounce & Hit Prediction**: Detects bounces and hits on the court.

            **How to Use**:
            1. Upload a video file (supported formats: `.mp4`, `.avi`, `.mov`).
            2. Click **Run Inference** to analyze and annotate your video.
            """)

            video_upload = gr.File(
                label="📤 Upload Video File",
                file_types=[".mp4", ".avi", ".mov"]
            )

            generate_button = gr.Button("🚀 Run Inference")

        # Output Section
        with gr.Column(scale=2):
            gr.Markdown("### 🎬 **Inference Output**")
            video_output = gr.Video(label="🎥 Annotated Video")
            details_output = gr.Markdown(label="ℹ️ Inference Details")

    # Functionality: Process the uploaded video
    generate_button.click(
        fn=process_and_display_random_video_inference,
        inputs=[video_upload],
        outputs=[video_output, details_output]
    )