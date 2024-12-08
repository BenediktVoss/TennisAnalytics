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

# Load the dataset once at the start
with open("../00_Dataset/annotations.json", "r") as file:  # Replace with your dataset JSON file path
    dataset = json.load(file)
    dataset['subsets'] = dataset['subsets'][:2]  

def get_object_for_frame(clip, frame):
    objects = {}
    objects['players'] = []
    objects['balls'] = []
    objects['court_lines'] = []
    objects['nets'] = []
    objects['keypoints'] = {}
    objects['players'] = clip['frames_with_objects'][frame]['players']
    objects['balls'] = clip['frames_with_objects'][frame]['balls']
    objects['court_lines'] = clip['frames_with_objects'][frame]['court_lines']
    objects['nets'] = clip['frames_with_objects'][frame]['nets']
    if len(clip['frames_with_objects'][frame]['keypoints'])>0:
        objects['keypoints'] = clip['frames_with_objects'][frame]['keypoints'][0]['points']

    return objects


# function that returns random frames with specific conditions
def get_random_video(dataset, subset_name):
    deep_copy = copy.deepcopy(dataset)

    frames = []

    if subset_name != "All Subsets":
        deep_copy["subsets"] = [subset for subset in deep_copy["subsets"] if subset["name"] == subset_name]

    # filter in subset Tracknet for videos "game4", "game6", "game8"
    for subset in deep_copy["subsets"]:
        if subset["name"] == "TrackNet":
            subset["videos"] = [video for video in subset["videos"] if video["name"] in ["game4", "game6", "game8"]]

    subset = np.random.choice(deep_copy["subsets"])
    #select random video
    video = np.random.choice(subset['videos'])
    #select random clip
    clip = np.random.choice(video['clips'])

    # loop over all frames in the clip
    for frame in clip['frames_with_objects']:
        # get all objects at this frame
        objects = get_object_for_frame(clip, frame)
        frames.append((frame, objects))

    return subset['name'], video['name'], clip['name'], frames
    
def extract_frame(subset, video_name, clip_name, frame_number, dataset_folder = "dataset"):
    # Construct the full path to the frame
    frame_path = os.path.join(dataset_folder, subset, video_name, clip_name, str(frame_number) + ".jpg")
    
    # Check if the file exists
    if not os.path.exists(frame_path):
        raise FileNotFoundError(f"The frame {frame_number} in {video_name}/{clip_name} does not exist")
    
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

def create_video_from_inference(subset_name, 
                                video_name, 
                                clip_name, 
                                frames, 
                                ball_positions, 
                                bounces_and_hits,
                                player_predictions,
                                court_positions, 
                                dataset_folder="../00_Dataset", 
                                output_video_path="_temp/output_video.mp4",
                                minimap_enabled=True):
    # Extract the first frame to determine video dimensions
    first_frame_number = frames[0][0]
    first_frame = extract_frame(subset_name, video_name, clip_name, first_frame_number, dataset_folder)
    height, width, _ = first_frame.shape

    fps = 30.0 if subset_name == "Amateur" else 25.0

    # Initialize the video writer with an MP4-compatible codec
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4 format
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # if ball_positions is longer than frames, cut the ball_positions
    ball_positions = ball_positions[:len(frames)]

    # calculate the homography matrix
    homography_matrix = get_homography_for_keypoints(court_positions)
    detected_bounces = []

    # Process and write each frame
    for idx, (frame_number, objects) in enumerate(frames):
        try:
            # Extract the frame using the existing function
            frame = extract_frame(subset_name, video_name, clip_name, frame_number, dataset_folder)

            # check if ball position is valid
            if not np.isnan(ball_positions[idx][0]):
                # draw circle on the ball position
                ball_position = (int(ball_positions[idx][0]), int(ball_positions[idx][1]))
                frame = draw_balls(frame, [ball_position], hex_to_bgr("#ffff00"))
            
            # draw court keypoints
            if len(court_positions) > 0:
                frame = draw_keypoints(frame, court_positions, hex_to_bgr("#34eb61"))

            # draw players
            if len(player_predictions[idx]) > 0:
                frame = draw_players(frame, player_predictions[idx], hex_to_bgr("#2800ff"))

            if minimap_enabled:
                # generate minimap
                minimap, detected_bounces = generate_minimap(homography_matrix, player_predictions[idx], ball_positions[idx], bounces_and_hits[idx], detected_bounces=detected_bounces)

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
            print(f"Error processing frame {frame_number}: {e}")
            continue

    # Release the video writer
    video_writer.release()

    return output_video_path


def process_and_display_random_video_inference(subset_name, dataset_folder="../00_Dataset"):
        # Timing dictionary to store step timings
    timing_info = {}
    start_time = time.time()

    # Generate random frames
    step_start_time = time.time()
    subset_name, video_name, clip_name, frames = get_random_video(dataset, subset_name)
    timing_info['random_video_generation'] = time.time() - step_start_time

    if len(frames) == 0:
        return None, "No frames found."
    
    path = os.path.join(dataset_folder, subset_name, video_name, clip_name)

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
    video_path = create_video_from_inference(subset_name, 
                                             video_name, 
                                             clip_name, 
                                             frames, 
                                             ball_positions, 
                                             bounces_and_hits, 
                                             player_predictions,
                                             court_positions, 
                                             dataset_folder)
    timing_info['video_creation'] = time.time() - step_start_time

    # Total execution time
    timing_info['total_time'] = time.time() - start_time

    # Generate the inference details string with Markdown table
    inference_details = (
        f"### Selected Video\n"
        f"- **Subset**: {subset_name}\n"
        f"- **Video**: {video_name}\n"
        f"- **Clip**: {clip_name}\n\n"
        f"Timing Information:\n"
        f"| Step                       | Time (seconds) |\n"
        f"|----------------------------|----------------|\n"
        f"| Random Video Selection     | {timing_info['random_video_generation']:.2f} |\n"
        f"| Ball Tracking              | {timing_info['ball_tracking']:.2f} |\n"
        f"| Smoothen Trajectory        | {timing_info['smoothen_trajectory']:.2f} |\n"
        f"| Bounce and Hit Prediction  | {timing_info['bounce_and_hit']:.2f} |\n"
        f"| Court Detection            | {timing_info['court_detection']:.2f} |\n"
        f"| Player Detection           | {timing_info['player_detection']:.2f} |\n"
        f"| Video Creation             | {timing_info['video_creation']:.2f} |\n"
        f"| Total Time                 | {timing_info['total_time']:.2f} |\n"
    )
    
    return video_path, inference_details


with gr.Blocks() as Inference:
    # Get all names of the subsets
    subset_names = [subset['name'] for subset in dataset['subsets']]
    subset_names.append("All Subsets")  # Add "All Subsets" option

    with gr.Row():
        # Input Section
        with gr.Column():
            gr.Markdown("""### 🧠 **Inference Explorer**""")
            gr.Markdown("""
            Visualize the performance of the tennis analytics system directly on the dataset by running full inference on random video clips.         
            |               | Ball Tracking               | Bounce & Hit        | Court Detection               | Player Detection  |
            |---------------|-----------------------------|---------------------|-------------------------------|-------------------|
            | **Model**     | TrackNet v2 (5-in-5-out)    | XGBoost             | Modified TrackNetv2           | YOLOv11           |
                        
            **Features**:
            - **Track the Ball**: Detect and refine ball movement in real-time.
            - **Detect Court & Players**: Overlay court lines and track player positions.
            - **Generate Minimap**: Visualize player positions and ball bounces on a dynamic minimap.
            - **Analyze Inference Speed**: View detailed processing times for each operation.
                        
            **How to Use**:
            1. Select a dataset subset from the dropdown.
            2. Click **Run Inference** to generate an annotated video and processing details.
            """)

            subset_dropdown = gr.Dropdown(
                choices=subset_names,
                label="🎯 Select Dataset Subset",
                value="All Subsets",
            )

            generate_button = gr.Button("🚀 Run Inference")

        # Output Section
        with gr.Column():
            video_output = gr.Video(label="🎬 Annotated Video")
            details_output = gr.Markdown(label="ℹ️ Inference Details")

    # Functionality: Generate a random video and its inference details
    generate_button.click(
        fn=process_and_display_random_video_inference,
        inputs=[subset_dropdown],
        outputs=[video_output, details_output],
    )
