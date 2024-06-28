import torch
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam
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
train_dataset = CatBreedDataset('../data/processed/train_labels.csv', '../data/images', transform)
test_dataset = CatBreedDataset('../data/processed/test_labels.csv', '../data/images', transform)

# Use a smaller subset of data for quick debugging
train_subset = Subset(train_dataset, range(0, len(train_dataset)//10))  # Use 10% of data
test_subset = Subset(test_dataset, range(0, len(test_dataset)//10))

train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)

# Load model and move to GPU if available
model = get_model(num_classes=len(train_dataset.img_labels['breed_id'].unique()))  # Adjust num_classes
model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

# Training loop with reduced epochs
num_epochs = 5  # Reduced number of epochs for quick debugging
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    print(f'Starting epoch {epoch+1}/{num_epochs}')
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)  # Move data to GPU
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 10 == 0:  # Print every 10 batches
            print(f'Batch {i}/{len(train_loader)}, Loss: {loss.item():.4f}')
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

# Save the model
torch.save(model.state_dict(), '../saved_models/cat_breed_model.pth')
print("Training complete and model saved!")
