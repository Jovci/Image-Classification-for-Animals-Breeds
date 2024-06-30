import torch
from torch.utils.data import DataLoader
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

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load model and move to GPU if available
num_classes = 37  # Update with the correct number of classes
model = get_model(num_classes=num_classes)
model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 25  # Adjust as needed
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    print(f'Starting epoch {epoch+1}/{num_epochs}')
    for i, (images, labels) in enumerate(train_loader):
        # Debugging: print labels to check their range
        print(f'Batch {i} labels: {labels.tolist()}')
        
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
