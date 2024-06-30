import torch
from torchvision import transforms
from PIL import Image
from model import get_model

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Number of classes (ensure this matches the number of classes used during training)
num_classes = 2  # Replace this with the actual number of classes used during training

# Class names mapping
class_names = ['Abyssinian', 'Bengal']  # Replace with your actual class names

# Load model and move to GPU if available
model = get_model(num_classes=num_classes)
model.load_state_dict(torch.load('../saved_models/cat_breed_model.pth'))
model.to(device)
model.eval()

def predict(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()


image_path = r'../data/test/cannoli_box.jpg'  
predicted_class = predict(image_path)
print(f'Predicted class: {predicted_class} ({class_names[predicted_class]})')
