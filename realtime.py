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

# Start the webcam
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to get a frame from the camera!")
        break

    # Convert frame to RGB (YOLOv5 expects RGB images)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform object detection
    results = model(img_rgb)

    # Get detection results as a pandas dataframe
    detections = results.pandas().xyxy[0]  # xmin, ymin, xmax, ymax, confidence, class, name

    # Draw bounding boxes and labels
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
    cv2.imshow('YOLOv5 Real-Time Detection', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
