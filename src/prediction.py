import torch
from torchvision import transforms
from PIL import Image
from model import get_model

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

def predict(image_path, model_path, transform):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    
    # Load model
    model = get_model(num_classes=len(class_names))
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    
    # Preprocess the image
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    
    class_id = predicted.item()
    class_name = class_names[class_id]
    
    return class_id, class_name