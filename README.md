# Facial Expression Detection using OpenCV and PyTorch

## Overview
This Python application detects facial expressions in real-time using a webcam. It leverages computer vision techniques with OpenCV for face detection and PyTorch for deep learning-based emotion recognition using a pre-trained ResNet18 model.

## Features
- **Real-time Emotion Detection**: Utilizes webcam feed to detect and predict emotions on faces.
- **Face Detection**: Uses OpenCV's Haar Cascade Classifier to detect faces in frames captured by the camera.
- **Emotion Recognition**: Employs a ResNet18 model fine-tuned for emotion recognition among seven classes (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral).
- **User Interface**: Displays real-time video feed with bounding boxes around detected faces and overlays predicted emotions with confidence scores.

## Dependencies
- Python 3.x
- Libraries:
  - OpenCV (cv2)
  - PyTorch
  - torchvision
  - PIL (Pillow)
  
## Setup
1. **Installation**:
   - Ensure Python 3.x is installed.
   - Install required libraries:
     ```
     pip install torch torchvision opencv-python pillow
     ```

2. **Pre-trained Model**:
   - The application uses a pre-trained ResNet18 model for emotion recognition.
   - The weights are loaded from the torchvision model zoo during initialization.

3. **Face Cascade Classifier**:
   - OpenCV's Haar Cascade Classifier for face detection is used (`haarcascade_frontalface_default.xml`).

## Usage
1. **Running the Application**:
   - Execute the Python script `facial_expression_detection.py`.
   - Ensure your webcam is connected and accessible.

2. **Interaction**:
   - The application will open a window showing the webcam feed with real-time emotion predictions and bounding boxes around detected faces.
   - Press `Q` or `ESC` to exit the application.

3. **Output**:
   - Detected faces will be highlighted with rectangles, and above each face, the predicted emotion along with confidence (probability) will be displayed.

## Troubleshooting
- **Camera Issues**: If the webcam feed does not appear, ensure it is not being used by another application.
- **Dependencies**: Make sure all required libraries are installed correctly.

## Contribution
Contributions are welcome! Feel free to fork the repository, make improvements, and submit pull requests.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Utilizes PyTorch for deep learning functionalities.
- Uses OpenCV for computer vision tasks.
- Model based on ResNet18 architecture for facial expression recognition.
