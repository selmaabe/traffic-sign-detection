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

# Load an image from file
image_path = r'C:\Users\selma\PycharmProjects\PythonProject\deneme1.jpg'
# Note: 'deneme1.jpg' is a test video used for demonstration purposes.
# This video is included only for testing the functionality of the code.
# Users should use their own test videos to run the script.
image = cv2.imread(image_path)

# Convert image to RGB (YOLOv5 expects RGB images)
img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Perform object detection
results = model(img_rgb)

# Get detection results as a pandas dataframe
detections = results.pandas().xyxy[0]  # xmin, ymin, xmax, ymax, confidence, class, name

# Check if any detections were made
if detections.empty:
    print("No detections found.")
else:
    print(f"Detected {len(detections)} objects.")

# Draw bounding boxes and labels
for _, detection in detections.iterrows():
    # Extract box coordinates, confidence, and class name
    x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
    confidence = detection['confidence']
    class_name = detection['name']

    # Display confidence and name
    label = f"{class_name} {confidence:.2f}"

    # Draw bounding box with a different color (blue)
    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Put label near the bounding box
    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# Display the image with annotations
cv2.imshow('YOLOv5 Object Detection', image)

# Wait until a key is pressed to close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
