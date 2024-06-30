import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from torchvision.models import efficientnet_b1  # Import the EfficientNet model
import torch.nn as nn
from dataset import CatBreedDataset

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Define transformations with more aggressive data augmentation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = CatBreedDataset('train_labels.csv', '../data/images', transform)
test_dataset = CatBreedDataset('test_labels.csv', '../data/images', transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the model
class CustomModel(nn.Module):
    def __init__(self, num_classes):
        super(CustomModel, self).__init__()
        self.base_model = efficientnet_b1(pretrained=True)
        self.dropout = nn.Dropout(p=0.5)
        num_ftrs = self.base_model.classifier[1].in_features
        self.base_model.classifier[1] = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        x = self.base_model(x)
        return self.dropout(x)

num_classes = 37  # Ensure this matches the number of classes
model = CustomModel(num_classes=num_classes)
model.load_state_dict(torch.load('../saved_models/cat_breed_model.pth'))  # Load pre-trained weights
model.to(device)

# Freeze all layers except the last few layers
for param in model.parameters():
    param.requires_grad = False
for param in model.base_model.classifier.parameters():
    param.requires_grad = True
for param in model.base_model.features[7:].parameters():
    param.requires_grad = True

criterion = torch.nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

# Training loop with early stopping
num_epochs = 25
early_stopping = EarlyStopping(patience=10, verbose=True)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    print(f'Starting epoch {epoch+1}/{num_epochs}')
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 10 == 0:
            print(f'Batch {i}/{len(train_loader)}, Loss: {loss.item():.4f}')
    val_loss = running_loss / len(train_loader)
    scheduler.step(val_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
    early_stopping(val_loss, model)
    if early_stopping.early_stop:
        print("Early stopping")
        break

# Save the fine-tuned model
torch.save(model.state_dict(), '../saved_models/cat_breed_model_finetuned_advanced.pth')
print("Fine-tuning complete and model saved!")
