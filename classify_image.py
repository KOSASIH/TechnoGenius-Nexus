import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Load pre-trained ResNet model
model = models.resnet50(pretrained=True)
model.eval()

# Define the image pre-processing transformation
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to perform image classification
def classify_image(image_path):
    # Load and pre-process the image
    image = Image.open(image_path)
    image = preprocess(image)
    image = image.unsqueeze(0)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        output = model(image)

    # Get the predicted class label
    _, predicted_idx = torch.max(output, 1)
    predicted_label = predicted_idx.item()

    return predicted_label

# Example usage
image_path = 'path/to/your/image.jpg'
predicted_label = classify_image(image_path)
print(f"Predicted label: {predicted_label}")
