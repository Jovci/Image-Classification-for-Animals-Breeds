import os
import pandas as pd

# Paths to dataset files
list_file_path = r'../data/annotations/list.txt'  
trainval_file_path = r'../data/annotations/trainval.txt' 
test_file_path = r'../data/annotations/test.txt'  

# Read list.txt to get the mapping
with open(list_file_path, 'r') as file:
    lines = file.readlines()

# Process the lines to extract the mapping, skipping the header lines
data = []
for line in lines[6:]:  # Skip the first 6 lines which contain the header information
    parts = line.strip().split(' ')
    if len(parts) == 4:
        file_name = parts[0] + '.jpg'
        class_id = int(parts[1]) - 1  # Adjust class_id to be 0-based
        species = int(parts[2])
        breed_id = int(parts[3])
        data.append([file_name, class_id, species, breed_id])

# Create a DataFrame
df = pd.DataFrame(data, columns=['file_name', 'class_id', 'species', 'breed_id'])

# Print the first few rows of the DataFrame for debugging
print("DataFrame created from list.txt:")
print(df.head())

# Read trainval.txt and test.txt to create train and test splits
with open(trainval_file_path, 'r') as file:
    trainval_files = [line.strip().split(' ')[0] + '.jpg' for line in file.readlines()]

with open(test_file_path, 'r') as file:
    test_files = [line.strip().split(' ')[0] + '.jpg' for line in file.readlines()]

# Print the first few entries of trainval_files and test_files for debugging
print("First few entries of trainval_files:")
print(trainval_files[:5])

print("First few entries of test_files:")
print(test_files[:5])

# Split the DataFrame into train and test sets
train_df = df[df['file_name'].isin(trainval_files)]
test_df = df[df['file_name'].isin(test_files)]

# Print the first few rows of the train and test DataFrames for debugging
print("Train DataFrame:")
print(train_df.head())

print("Test DataFrame:")
print(test_df.head())

# Save the train and test labels to CSV files
train_df.to_csv('train_labels.csv', index=False)
test_df.to_csv('test_labels.csv', index=False)

print("CSV files for train and test labels have been created successfully!")
