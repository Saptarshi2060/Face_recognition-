import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image
import numpy as np

# Define the CNN model architecture
class FacialExpressionModel(nn.Module):
    def __init__(self, num_classes):
        super(FacialExpressionModel, self).__init__()
        self.resnet = resnet18()  # Initialize ResNet18 without pre-trained weights
        self.load_pretrained_weights()  # Load pre-trained weights
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.resnet(x)
    
    def load_pretrained_weights(self):
        # Load pre-trained weights from torchvision model zoo
        state_dict = torch.hub.load_state_dict_from_url(
            'https://download.pytorch.org/models/resnet18-5c106cde.pth',
            progress=True,  # Show download progress
        )
        self.resnet.load_state_dict(state_dict)

# Initialize the model
model = FacialExpressionModel(num_classes=7)  # Adjust num_classes according to your model
model.eval()

# Load the face cascade for detecting faces
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Labels for the emotions (modify according to your model)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Function to preprocess the image
def preprocess_image(image):
    image = image.convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)

# Function to detect and predict facial expressions from the camera
def detect_emotion():
    # Open default camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Loop to capture frames from the camera
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip frame for mirror effect
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            roi_color = frame[y:y + h, x:x + w]
            roi_pil = Image.fromarray(cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB))
            roi_tensor = preprocess_image(roi_pil)

            # Predict emotion
            with torch.no_grad():
                outputs = model(roi_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                predicted = torch.argmax(probabilities, dim=1)
                label = emotion_labels[predicted.item()]
                confidence = torch.max(probabilities).item()

            # Display prediction and confidence as text on the frame
            cv2.putText(frame, f"{label} ({confidence:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('Facial Expression Detector', frame)

        # Check for ESC key press or window close event to exit
        key = cv2.waitKey(1) & 0xFF
        if key in [27, ord('q'), ord('Q')]:
            break

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Main function to start detecting facial expressions
if __name__ == "__main__":
    try:
        detect_emotion()
    except Exception as e:
        print(f"An error occurred: {e}")
        cv2.destroyAllWindows()
