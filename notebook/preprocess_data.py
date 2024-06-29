import os
import pandas as pd
import xml.etree.ElementTree as ET

# Define directories
annotations_dir = '../data/annotations/xmls'  # Use raw strings or forward slashes
images_dir = '../data/images'
output_dir = '../data/processed'

# Function to extract data from XML file
def extract_data_from_xml(xml_file, breed_to_id):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    breed_name = root.find('object/name').text
    data = {
        'file_name': root.find('filename').text,
        'breed_id': breed_to_id[breed_name]
    }
    return data

# Get a list of all breeds
breed_names = set()
for xml_file in os.listdir(annotations_dir):
    if xml_file.endswith('.xml'):
        tree = ET.parse(os.path.join(annotations_dir, xml_file))
        root = tree.getroot()
        breed_name = root.find('object/name').text
        breed_names.add(breed_name)

# Create a mapping from breed name to unique integer ID
breed_to_id = {breed_name: idx for idx, breed_name in enumerate(sorted(breed_names))}

# Extract data from all XML files
data = []
for xml_file in os.listdir(annotations_dir):
    if xml_file.endswith('.xml'):
        data.append(extract_data_from_xml(os.path.join(annotations_dir, xml_file), breed_to_id))

# Convert to DataFrame
df = pd.DataFrame(data)

# Split into train and test sets (assuming an 80-20 split)
train_df = df.sample(frac=0.8, random_state=42)
test_df = df.drop(train_df.index)

# Save to CSV
os.makedirs(output_dir, exist_ok=True)
train_df.to_csv(os.path.join(output_dir, 'train_labels.csv'), index=False)
test_df.to_csv(os.path.join(output_dir, 'test_labels.csv'), index=False)

print("CSV files created successfully!")
