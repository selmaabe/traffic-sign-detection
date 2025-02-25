# # Traffic Sign Detection Using YOLOv5

## Overview
This project is a real-time traffic sign detection system built using the YOLOv5 model. It detects various traffic signs, such as stop signs and speed limit signs, from images, videos, and webcam feeds. The system is designed to assist autonomous driving applications by improving traffic sign recognition.

## Features
- Real-time traffic sign detection
- Uses YOLOv5 for high-accuracy object detection
- Supports image, video, and webcam inputs
- Achieves **89.3% accuracy** and **90.7% recall**
- Implemented with OpenCV and PyTorch

## Installation

### Prerequisites
Make sure you have the following installed:
- Python 3.8+
- pip
- Git

### Clone the Repository
```sh
git clone https://github.com/your-username/traffic-sign-detection.git
cd traffic-sign-detection
```

### Install Dependencies
```sh
pip install -r requirements.txt
```

## Usage

### Running the Model
To start real-time detection using a webcam:
```sh
python detect.py
```

To detect traffic signs in an image:
```sh
python detect.py --source path/to/image.jpg
```

To detect traffic signs in a video:
```sh
python detect.py --source path/to/video.mp4
```

## Model Training
The YOLOv5 model was trained on Google Colab for 50 epochs using a dataset of annotated traffic sign images. The dataset includes various lighting conditions and angles to ensure robustness.

## Results
The model achieved:
- **89.3% accuracy**
- **90.7% recall**
- Effective detection under various conditions

## Contributing
Feel free to contribute to this project by submitting issues or pull requests.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
- Ultralytics for YOLOv5
- OpenCV for image processing
- PyTorch for deep learning capabilities


 
