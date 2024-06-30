import pandas as pd

# Load the CSV files
train_labels_path = '../data/processed/train_labels.csv'
test_labels_path = '../data/processed/test_labels.csv'

train_labels = pd.read_csv(train_labels_path)
test_labels = pd.read_csv(test_labels_path)

# Verify label range
num_classes = 37  # Update with the correct number of classes

train_label_range = (train_labels['class_id'].min(), train_labels['class_id'].max())
test_label_range = (test_labels['class_id'].min(), test_labels['class_id'].max())

print(f'Train labels range: {train_label_range}')
print(f'Test labels range: {test_label_range}')

# Check if any label is out of range
train_out_of_range = train_labels[(train_labels['class_id'] < 0) | (train_labels['class_id'] >= num_classes)]
test_out_of_range = test_labels[(test_labels['class_id'] < 0) | (test_labels['class_id'] >= num_classes)]

if not train_out_of_range.empty:
    print("Train labels out of range:")
    print(train_out_of_range)

if not test_out_of_range.empty:
    print("Test labels out of range:")
    print(test_out_of_range)
