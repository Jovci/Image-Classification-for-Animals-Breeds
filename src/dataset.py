import os
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms

class CatBreedDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        if not img_path.endswith(".jpg"):
            img_path += ".jpg"
        image = Image.open(img_path).convert("RGB")
        label = self.img_labels.iloc[idx, 1]  # Use breed_id as the label
        if self.transform:
            image = self.transform(image)
        return image, label

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Example usage
train_dataset = CatBreedDataset('../data/processed/train_labels.csv', '../data/images', transform)
test_dataset = CatBreedDataset('../data/processed/test_labels.csv', '../data/images', transform)
