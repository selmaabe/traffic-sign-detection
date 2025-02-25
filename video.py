import platform
import pathlib
import torch
import cv2

# Fix the Path class according to the platform
if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath
else:
    pathlib.WindowsPath = pathlib.PosixPath

# Load custom YOLOv5 model
model_path = 'best.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

# Load a video file
video_path = r'C:\Users\selma\PycharmProjects\PythonProject\sagadonusyasak.mp4'
# Note: 'sagadonusyasak.mp4' is a test video used for demonstration purposes.
# This video is included only for testing the functionality of the code.
# Users should use their own test videos to run the script.
cap = cv2.VideoCapture(video_path)

# Check if the video was successfully loaded
if not cap.isOpened():
    print(f"Error: Could not open video file: {video_path}")
    exit()

# Process the video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video or cannot fetch the frame.")
        break

    # Convert the frame to RGB (YOLOv5 expects RGB images)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform object detection
    results = model(img_rgb)

    # Get detection results as a pandas dataframe
    detections = results.pandas().xyxy[0]  # xmin, ymin, xmax, ymax, confidence, class, name

    # Draw bounding boxes and labels if detections are found
    for _, detection in detections.iterrows():
        # Extract box coordinates, confidence, and class name
        x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
        confidence = detection['confidence']
        class_name = detection['name']

        # Display confidence and name
        label = f"{class_name} {confidence:.2f}"

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Put label near the bounding box
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Display the frame with annotations
    cv2.imshow('YOLOv5 Video Detection', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting on user request.")
        break

# Release video capture and close display windows
#cap.release()
#cv2.destroyAllWindows()
