import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report
from dataset import CatBreedDataset
from model import get_model
from torchvision import transforms

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
test_dataset = CatBreedDataset('../data/processed/test_labels.csv', '../data/images', transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load model and move to GPU if available
model = get_model(num_classes=len(test_dataset.img_labels['breed_id'].unique()))  # Adjust num_classes
model.load_state_dict(torch.load('../saved_models/cat_breed_model.pth'))
model.to(device)
model.eval()

# Evaluate the model
all_preds = []
all_labels = []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Calculate accuracy and classification report
accuracy = accuracy_score(all_labels, all_preds)
report = classification_report(all_labels, all_preds, target_names=test_dataset.img_labels['breed_id'].unique().astype(str))

print(f'Accuracy: {accuracy:.4f}')
print('Classification Report:')
print(report)