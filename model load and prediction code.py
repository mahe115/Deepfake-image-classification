from facenet_pytorch import InceptionResnetV1
import torch
from torchvision import transforms
from PIL import Image

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Running on device: {device}')

# Define the transformation
trans = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    lambda x: x * 255 - 0.5  # Reverse of fixed_image_standardization
])

# Function to load the model
def load_model(model_path, class_num):
    model = InceptionResnetV1(classify=True, num_classes=class_num).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# Function to predict a single image
def predict_single_image(image_path, model, transform):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    idx_to_class = {0: 'real', 1: 'fake'}
    return idx_to_class[predicted.item()]

# Load the model
model_path = 'my_own_model25ola.pt'
class_num = 2
model = load_model(model_path, class_num)


# Path to your single image
image_path = 'ai-images-news (1).jpg'  # Update this path to your specific image

# Predicting the class of the single image
predicted_class = predict_single_image(image_path, model, trans)
print(f'Predicted class of {image_path}: {predicted_class}')
