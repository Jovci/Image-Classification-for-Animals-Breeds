import torch
from torchvision import transforms
from PIL import Image
from model import get_model
import torch.nn as nn

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
num_classes = 37 

# Class names mapping
class_names = {
    0: 'Abyssinian',
    1: 'American Bulldog',
    2: 'American Pit Bull Terrier',
    3: 'Basset Hound',
    4: 'Beagle',
    5: 'Bengal',
    6: 'Birman',
    7: 'Bombay',
    8: 'Boxer',
    9: 'British Shorthair',
    10: 'Chihuahua',
    11: 'Egyptian Mau',
    12: 'English Cocker Spaniel',
    13: 'English Setter',
    14: 'German Shorthaired Pointer',
    15: 'Great Pyrenees',
    16: 'Havanese',
    17: 'Japanese Chin',
    18: 'Keeshond',
    19: 'Leonberger',
    20: 'Maine Coon',
    21: 'Miniature Pinscher',
    22: 'Newfoundland',
    23: 'Persian',
    24: 'Pomeranian',
    25: 'Pug',
    26: 'Ragdoll',
    27: 'Russian Blue',
    28: 'Samoyed',
    29: 'Scottish Terrier',
    30: 'Shiba Inu',
    31: 'Siamese',
    32: 'Sphynx',
    33: 'Staffordshire Bull Terrier',
    34: 'Wheaten Terrier',
    35: 'Yorkshire Terrier',
    36: 'Saint Bernard'
}

# Load model and move to GPU if available
model = get_model(num_classes=num_classes)
model.load_state_dict(torch.load('../saved_models/cat_breed_model_finetuned.pth'))
model.to(device)
model.eval()

def predict(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

# Usage
image_path = r'../data/test/siamese.jpg'  
predicted_class = predict(image_path)
class_name = class_names[predicted_class]
print(f'Predicted class ID: {predicted_class}, Class Name: {class_name}')
