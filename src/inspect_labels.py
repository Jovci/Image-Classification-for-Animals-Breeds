import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os

# Paths to CSV files
train_csv_path = '../data/processed/train_labels.csv'
test_csv_path = '../data/processed/test_labels.csv'
images_dir = '../data/images'

# Read CSV files
train_labels = pd.read_csv(train_csv_path)
test_labels = pd.read_csv(test_csv_path)

# Display the first few rows of the CSV files
print("Train Labels:")
print(train_labels.head())

print("\nTest Labels:")
print(test_labels.head())

# Function to visualize an image and its label
def show_image_with_label(image_file, label):
    image_path = os.path.join(images_dir, image_file)
    image = Image.open(image_path)
    plt.imshow(image)
    plt.title(f'Label: {label}')
    plt.axis('off')
    plt.show()

# Display a few examples from the training set
print("\nSample Training Images:")
for idx, row in train_labels.head(5).iterrows():
    show_image_with_label(row['file_name'], row['breed_id'])

# Display a few examples from the test set
print("\nSample Test Images:")
for idx, row in test_labels.head(5).iterrows():
    show_image_with_label(row['file_name'], row['breed_id'])
