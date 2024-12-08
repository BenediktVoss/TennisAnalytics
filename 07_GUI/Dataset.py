import gradio as gr
import pandas as pd
import numpy as np
import json
import cv2
import os
import numpy as np
import random

# Load the dataset once at the start
with open("../00_Dataset/annotations.json", "r") as file:  # Replace with your dataset JSON file path
    dataset = json.load(file)

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
def get_random_frames(dataset, subset_name, ball_visibility= None, ball_trajectory= None, limit = 35):
    clips = []

    if subset_name != "All Subsets":
        dataset["subsets"] = [subset for subset in dataset["subsets"] if subset["name"] == subset_name]

    while len(clips) < limit:
        subset = np.random.choice(dataset["subsets"])
        #select random video
        video = np.random.choice(subset['videos'])
        #select random clip
        clip = np.random.choice(video['clips'])

        clip
        
        # filter frames with objects based on ball visibility and trajectory
        frames = []

        for frame_id, frame in clip['frames_with_objects'].items():
            # if both conditions are None, add all frames
            if ball_visibility == None and ball_trajectory == None:
                frames.append(frame_id)
                continue
            # check if balls are visible
            if len(frame['balls']) == 0:
                continue
            for ball in frame['balls']:
                if (ball['visibility'] == ball_visibility or ball_visibility == None) and (ball['trajectory'] == ball_trajectory or ball_trajectory == None):
                    frames.append(frame_id)
                    break
            
        # select random frame
        if len(frames) > 0:
            frame = np.random.choice(frames)
        else:
            continue

        # get all objects at this frame
        objects = get_object_for_frame(clip, frame)

        return subset['name'], video['name'], clip['name'], frame, objects
    
    return None

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


def process_and_display_random_frame(subset_name, dataset_folder="../00_Dataset"):
    # Generate random frames with specified conditions (None for default behavior)
    result = get_random_frames(dataset, subset_name)

    if not result:
        return None, "No frames matched the conditions."

    subset_name, video_name, clip_name, frame_number, objects = result

    # Extract the frame
    frame = extract_frame(subset_name, video_name, clip_name, frame_number, dataset_folder)

    # Create visualization by drawing objects on the frame
    frame_with_objects = draw_objects(dataset, frame, objects)

    # Save the processed frame to display it
    output_path = "_temp/output_frame.jpg"
    cv2.imwrite(output_path, frame_with_objects)

    # Format the `objects` dictionary as a pretty string
    objects_pretty = json.dumps(objects, indent=4)

    # Return the output image path and the frame details
    frame_details = (
        f"Subset: {subset_name}\n"
        f"Video: {video_name}\n"
        f"Clip: {clip_name}\n"
        f"Frame: {frame_number}\n"
        f"Objects:\n{objects_pretty}"
    )
    return output_path, frame_details

with gr.Blocks() as Dataset:
    # Get all names of the subsets
    subset_names = [subset['name'] for subset in dataset['subsets']]
    subset_names.append("All Subsets")  # Add "All Subsets" option
   
    with gr.Row():
        # Input Section
        with gr.Column():
            gr.Markdown("""### 🎾 **Tennis Frame Explorer**""")
            gr.Markdown("""
            Explore the tennis analytics dataset by generating **random annotated frames**.

            **Features**:
            - Displays a random frame with annotations for:
                - **Players**: Bounding boxes for players.
                - **Balls**: Ball positions and trajectories.
                - **Court Lines & Nets**: Overlays for court and net structures.
            - Provides detailed metadata and object information for each frame.

            **How to Use**:
            1. Select a dataset subset from the dropdown.
            2. Click **Generate Frame** to view a random annotated frame and its details.
            """)

            subset_dropdown = gr.Dropdown(
                choices=subset_names,
                label="🎯 Select Dataset Subset",
                value="All Subsets",
            )

            generate_button = gr.Button("🚀 Generate Frame")

        # Output Section
        with gr.Column():
            frame_output = gr.Image(label="📸 Annotated Frame")
            details_output = gr.Textbox(
                label="ℹ️ Frame Details",
                lines=10,
                interactive=False,
            )

    # Functionality: Generate a random frame
    generate_button.click(
        fn=process_and_display_random_frame,
        inputs=[subset_dropdown],
        outputs=[frame_output, details_output],
    )