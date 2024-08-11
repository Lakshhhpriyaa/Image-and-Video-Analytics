#!/usr/bin/env python
# coding: utf-8

# In[3]:


from google.colab.patches import cv2_imshow
import cv2
import numpy as np

video_path = "/content/video.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()


frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
duration = frame_count / fps

print(f"Frame Count: {frame_count}")
print(f"Frame Rate (FPS): {fps}")
print(f"Duration (seconds): {duration}")


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the frame using cv2_imshow
    cv2_imshow(gray_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()


# In[3]:


import subprocess

def get_video_resolution(video_path):
    result = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height', '-of', 'csv=s=x:p=0', video_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )

    resolution = result.stdout.strip()
    return resolution

def main():
    video_path = 'video.mp4'
    resolution = get_video_resolution(video_path)
    print(f"Video Resolution: {resolution}")

if __name__ == '__main__':
    main()


# In[1]:


pip install ffmpeg-python


# In[2]:


import cv2


# In[1]:


import cv2
import subprocess
from google.colab.patches import cv2_imshow

def get_frame_types(video_path):
    result = subprocess.run(
        ['ffprobe', '-select_streams', 'v', '-show_frames', '-show_entries', 'frame=pict_type', '-of', 'csv', video_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )

    frame_types = [line for line in result.stdout.splitlines() if line]
    return frame_types


def display_and_save_frames(video_path, frame_types):
    cap = cv2.VideoCapture(video_path)
    frame_index = 0
    saved_frame_count = 0

    while cap.isOpened() and saved_frame_count < 20:
        ret, frame = cap.read()
        if not ret:
            break

        # Ensure frame_index is within bounds
        if frame_index < len(frame_types):
            frame_type = frame_types[frame_index]
            frame_type = frame_type.split(',')[1]  # Extract the frame type (I, P, B)

            # Display the frame with its type
            cv2.putText(frame, f'Frame Type: {frame_type}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2_imshow(frame)  # Use cv2_imshow for Google Colab

            time_to_save = {'I': 'I_frame_{}.png', 'P': 'P_frame_{}.png', 'B': 'B_frame_{}.png'}
            if frame_type in time_to_save:
                cv2.imwrite(time_to_save[frame_type].format(saved_frame_count), frame)
                saved_frame_count += 1

        frame_index += 1

    cap.release()

# Main function
def main():
    video_path = 'video.mp4'
    frame_types = get_frame_types(video_path)
    display_and_save_frames(video_path, frame_types)

if __name__ == '__main__':
    main()


# In[2]:


import subprocess

# Function to get frame types using ffprobe
def get_frame_types(video_path):
    result = subprocess.run(
        ['ffprobe', '-select_streams', 'v', '-show_frames', '-show_entries', 'frame=pict_type', '-of', 'csv', video_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )

    frame_types = [line.split(',')[1] for line in result.stdout.splitlines() if line]
    return frame_types

# Function to count frame types
def count_frame_types(frame_types):
    i_count = frame_types.count('I')
    p_count = frame_types.count('P')
    b_count = frame_types.count('B')
    return i_count, p_count, b_count


def main():
    video_path = 'video.mp4'
    frame_types = get_frame_types(video_path)
    i_count, p_count, b_count = count_frame_types(frame_types)
    print(f"Number of I frames: {i_count}")
    print(f"Number of P frames: {p_count}")
    print(f"Number of B frames: {b_count}")

if __name__ == '__main__':
    main()


# In[ ]:




