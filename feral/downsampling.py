import cv2
from pathlib import Path

def downsample_video(input_path, output_path, target_fps=30):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Could not open input video.")
        return

    # Get original video properties
    source_fps = cap.get(cv2.CAP_PROP_FPS)  # Should be 200
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Configure the output video writer (using MP4V codec)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, target_fps, (width, height))

    # Time tracking variables
    source_interval = 1.0 / source_fps
    target_interval = 1.0 / target_fps
    accumulated_time = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Check if current frame aligns with the target timeline
        if accumulated_time >= target_interval:
            out.write(frame)
            accumulated_time -= target_interval
        
        # Advance time by one frame step
        accumulated_time += source_interval

    # Clean up resources
    cap.release()
    out.release()
    print("Downsampling complete!")

source_name = '/groups/karashchuk/home/dengd3/feral_project/allen_mouse_805164'
source_path = Path(source_name)

target_name = "/groups/karashchuk/home/dengd3/feral_project/allen_mouse_805164_downsampled"
target_path = Path(target_name)

for item in source_path.iterdir():
    item_name = item.name
    downsample_video(f"{source_name}/{item_name}", f"{target_name}/{item_name}_30hz.mp4", target_fps=30)