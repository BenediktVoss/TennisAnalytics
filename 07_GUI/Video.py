import gradio as gr
import pandas as pd
import numpy as np
import json
import cv2
import os
import numpy as np
import random

# Load the dataset once at the start
with open("dataset/annotations.json", "r") as file:  # Replace with your dataset JSON file path
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
    frames = []

    if subset_name != "All Subsets":
        dataset["subsets"] = [subset for subset in dataset["subsets"] if subset["name"] == subset_name]

    subset = np.random.choice(dataset["subsets"])
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
        x = int(float(ball['x']))
        y = int(float(ball['y']))
        
        # Draw the bounding box
        if fill:
            cv2.circle(frame_copy, (x, y), 4, color, -1, lineType=cv2.LINE_AA)
        # Draw a larger circle around the ball in black
        if outline:
            cv2.circle(frame_copy, (x, y), 7, (0, 0, 0), 1, lineType=cv2.LINE_AA)

    return frame_copy

def draw_court_lines(frame, court_lines, color):
    # Create a copy of the frame
    frame_copy = frame.copy()
    
    # Draw a bounding box for each court line
    for line in court_lines:
        # Get the coordinates of the bounding box round float to int
        x1 = int(float(line['x1']))
        y1 = int(float(line['y1']))
        x2 = int(float(line['x2']))
        y2 = int(float(line['y2']))
        
        # Draw the bounding box
        cv2.line(frame_copy, (x1, y1), (x2, y2), color, 2)

    return frame_copy

def draw_keypoints(frame, keypoints, color):
    # Create a copy of the frame
    frame_copy = frame.copy()

    # Draw a circle for each keypoint
    for key,value in keypoints.items():
        if value is not None:
            # Get the coordinates of the keypoint round float to int
            x = int(float(value[0]))
            y = int(float(value[1]))
        
        # Draw the keypoint
        cv2.circle(frame_copy, (x, y), 4, color, -1, lineType=cv2.LINE_AA)

    return frame_copy

def draw_nets(frame, nets, color):
    # Create a copy of the frame
    frame_copy = frame.copy()
    
    # Draw a bounding box for each net
    for net in nets:
        points = net['points']
        polygon_points = np.array(
            [[int(round(float(point['x']))), int(round(float(point['y'])))] for point in points],
            np.int32
        )
        polygon_points = polygon_points.reshape((-1, 1, 2))
        
        # Draw the polygon on the frame copy
        cv2.polylines(frame_copy, [polygon_points], isClosed=True, color=color, thickness=2)

    return frame_copy


def draw_objects(data, frame, objects, player=True, ball=True, court_line=True, net=True, keypoint=True):
    # Create a copy of the frame
    frame_copy = frame.copy()
    
    # draw players
    if player:
        players = objects['players']
        player_color = hex_to_bgr(data['labels'][0]['color'])
        frame_copy = draw_players(frame_copy, players, player_color)
    # draw balls
    if ball:
        balls = objects['balls']
        ball_color = hex_to_bgr(data['labels'][1]['color'])
        frame_copy = draw_balls(frame_copy, balls, ball_color)
    # draw court lines
    if court_line:
        court_lines = objects['court_lines']
        court_line_color = hex_to_bgr(data['labels'][2]['color'])
        frame_copy = draw_court_lines(frame_copy, court_lines, court_line_color)
    # draw nets
    if net:
        nets = objects['nets']
        net_color = hex_to_bgr(data['labels'][3]['color'])
        frame_copy = draw_nets(frame_copy, nets, net_color)
    # draw keypoints
    if keypoint:
        keypoints = objects['keypoints']
        keypoint_color = hex_to_bgr(data['labels'][4]['color'])
        frame_copy = draw_keypoints(frame_copy, keypoints, keypoint_color)

    return frame_copy


def create_video_from_frames(subset_name, video_name, clip_name, frames, dataset_folder="../00_Dataset", output_video_path="_temp/output_video.mp4"):

    # Extract the first frame to determine video dimensions
    first_frame_number = frames[0][0]
    first_frame = extract_frame(subset_name, video_name, clip_name, first_frame_number, dataset_folder)
    height, width, _ = first_frame.shape

    # Initialize the video writer with an MP4-compatible codec
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4 format
    video_writer = cv2.VideoWriter(output_video_path, fourcc, 30.0, (width, height))

    # Process and write each frame
    for frame_number, objects in frames:
        try:
            # Extract the frame using the existing function
            frame = extract_frame(subset_name, video_name, clip_name, frame_number, dataset_folder)

            # Draw objects on the frame using the existing function
            annotated_frame = draw_objects(dataset, frame, objects)

            # Write the annotated frame to the video
            video_writer.write(annotated_frame)

        except FileNotFoundError as e:
            print(f"Error processing frame {frame_number}: {e}")
            continue

    # Release the video writer
    video_writer.release()

    return output_video_path


def process_and_display_random_video(subset_name, dataset_folder="dataset"):
    # Generate random frames with specified conditions (None for default behavior)
    subset_name, video_name, clip_name, frames = get_random_video(dataset, subset_name)

    if len(frames) == 0:
        return None, "No frames found."

    # Create a video from the frames
    video_path = create_video_from_frames(subset_name, video_name, clip_name, frames, dataset_folder)

    # Return the output image path and the frame details
    frame_details = (
        f"Subset: {subset_name}\n"
        f"Video: {video_name}\n"
        f"Clip: {clip_name}\n"
    )
    return video_path, frame_details


# Creating the Gradio Blocks interface with subset filtering
with gr.Blocks() as Video:
    gr.Markdown("# Video generation")
    gr.Markdown("### Visualize entire videos in subsets New and TrackNet")

    # Get all names of the subsets
    subset_names = [subset['name'] for subset in dataset['subsets']]
    subset_names.append("All Subsets")  # Add "All Subsets" option

    with gr.Row():
        # Left Column
        with gr.Column():
            # Dropdown input for selecting subsets
            subset_dropdown = gr.Dropdown(
                choices=subset_names,
                label="Select Subset",
                value="All Subsets",
            )

            # Button to generate random frame visualization
            random_video_button = gr.Button("Generate Random Video")

            # Output for the frame details
            frame_details_output = gr.Textbox(label="Video Details", interactive=False)

        # Right Column
        with gr.Column():
            # Output for the processed image
            video_output = gr.Video(label="Random Video with Annotations")

    # Functionality: Generate a random frame on button click
    random_video_button.click(
        fn=process_and_display_random_video,
        inputs=[subset_dropdown],
        outputs=[video_output, frame_details_output],
    )
