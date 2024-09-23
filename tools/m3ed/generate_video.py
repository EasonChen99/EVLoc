import cv2
import os
import tqdm
import numpy as np

# Parameters
image_folder = '/home/eason/WorkSpace/EventbasedVisualLocalization/tools/m3ed/visualization/falcon_indoor'  # Replace with your folder path
output_video = 'output_video.mp4'           # Name of the output video file
frame_rate = 12                          # Frame rate of the video

# Get all images in the folder
images = [img for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
images.sort()  # Sort images by filename to maintain order

# Read the first image to get the dimensions
first_image_path = os.path.join(image_folder, images[0])
frame = cv2.imread(first_image_path)
height, width, layers = frame.shape

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' for .mp4, or 'XVID' for .avi
video = cv2.VideoWriter(output_video, fourcc, frame_rate, (width*2, height))

# Add all images to the video
# for image in tqdm.tqdm(images):
# for i in range(len(images) // 2):
i = 0
while i < len(images) - 1:
    TS_path = os.path.join(image_folder, images[i+1])
    TSTS_path = os.path.join(image_folder, images[i])
    TS_frame = cv2.imread(TS_path)
    TSTS_frame = cv2.imread(TSTS_path)
    frame = np.hstack((TS_frame, TSTS_frame)) # event representation
    # frame = np.hstack((TSTS_frame, TS_frame)) # overlay

    
    text = f"TS"
    org = (frame.shape[1]-1900, frame.shape[0]-550)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (255, 255, 255)
    thickness = 2
    # Add text to the image
    cv2.putText(frame, text, org, font, fontScale, color, thickness, cv2.LINE_AA)
    org = (frame.shape[1]-940, frame.shape[0]-550)
    text = f"TSTS"
    cv2.putText(frame, text, org, font, fontScale, color, thickness, cv2.LINE_AA)


    video.write(frame)
    i += 2

# Release the VideoWriter object
video.release()

print(f"Video saved as {output_video}")
