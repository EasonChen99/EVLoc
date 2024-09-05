import cv2
import os
import tqdm

# Parameters
image_folder = '/home/eason/WorkSpace/EventbasedVisualLocalization/E2D_Loc/visualization/odometry'  # Replace with your folder path
output_video = 'output_video.mp4'           # Name of the output video file
frame_rate = 8                             # Frame rate of the video

# Get all images in the folder
images = [img for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
images.sort()  # Sort images by filename to maintain order

# Read the first image to get the dimensions
first_image_path = os.path.join(image_folder, images[0])
frame = cv2.imread(first_image_path)
height, width, layers = frame.shape

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' for .mp4, or 'XVID' for .avi
video = cv2.VideoWriter(output_video, fourcc, frame_rate, (width, height))

# Add all images to the video
for image in tqdm.tqdm(images):
    image_path = os.path.join(image_folder, image)
    frame = cv2.imread(image_path)
    video.write(frame)

# Release the VideoWriter object
video.release()

print(f"Video saved as {output_video}")
