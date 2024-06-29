import torch
from torchvision import transforms
from PIL import Image
from src.model import get_model

def predict(image_path, model_path, transform, num_classes):
    # Load the model
    model = get_model(num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Load and preprocess the image
    image = Image.open(image_path)
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension

    # Make prediction
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    return predicted.item()

# Example usage
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
prediction = predict('PREDICT IMAGE HERE RAHHHHHHHHHH', 'saved_models/cat_breed_model.pth', transform, num_classes=12)
print(f'Predicted breed: {prediction}')
