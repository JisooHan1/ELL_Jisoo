# import os
# import json

# def preprocess_tiny_imagenet(root='./datasets/tiny-imagenet-200'):
#     """Save the train/val datasets of Tiny ImageNet-200 as a JSON file"""
#     dataset_info = {'train': [], 'val': []}

#     # Process Train data
#     train_root = os.path.join(root, 'train')
#     cls_idx = {cls: idx for idx, cls in enumerate(sorted(os.listdir(train_root)))}

#     for cls, idx in cls_idx.items():
#         cls_path = os.path.join(train_root, cls, 'images')
#         for img in os.listdir(cls_path):
#             dataset_info['train'].append((os.path.join(cls_path, img), idx))

#     # Process Validation data
#     val_root = os.path.join(root, 'val')
#     val_images_path = os.path.join(val_root, 'images')
#     val_labels_path = os.path.join(val_root, 'val_annotations.txt')

#     with open(val_labels_path, 'r') as f:
#         for line in f:
#             image_name, label, *_ = line.strip().split('\t')
#             image_path = os.path.join(val_images_path, image_name)
#             dataset_info['val'].append((image_path, cls_idx[label]))

#     # Save as a JSON file
#     json_path = os.path.join(root, 'tiny_imagenet_preprocessed.json')
#     with open(json_path, 'w') as f:
#         json.dump(dataset_info, f)

#     print(f"Preprocessing complete! JSON file saved: {json_path}")

#     # Verify the number of images after processing
#     print(f"Train samples: {len(dataset_info['train'])}")
#     print(f"Validation samples: {len(dataset_info['val'])}")

#     # Verify the existence of actual files (example)
#     for path, _ in dataset_info['train'] + dataset_info['val']:
#         if not os.path.exists(path):
#             print(f"Warning: Missing image file - {path}")

# preprocess_tiny_imagenet()


import os
import json
import random

# Set the number of classes to select
num_classes = 50  # Modify as needed

# Load the existing JSON file
json_path = './datasets/tiny-imagenet-200/tiny_imagenet_preprocessed.json'
with open(json_path, 'r') as f:
    dataset_info = json.load(f)

# Get the list of existing classes
all_classes = set()
for img_path, label in dataset_info['train']:
    cls_name = os.path.basename(os.path.dirname(os.path.dirname(img_path)))  # Corrected class extraction
    all_classes.add(cls_name)

# Check the actual number of available classes
num_classes = min(num_classes, len(all_classes))  # Prevent errors
print(f"Using {num_classes} classes out of {len(all_classes)} available classes.")

# Select a random subset of classes
selected_classes = random.sample(list(all_classes), num_classes)

# Create a new JSON dataset
new_dataset_info = {'train': [], 'val': []}

for split in ['train', 'val']:
    for img_path, label in dataset_info[split]:
        cls_name = os.path.basename(os.path.dirname(os.path.dirname(img_path)))  # Corrected class extraction
        if cls_name in selected_classes:
            new_dataset_info[split].append((img_path, label))

# Save the new JSON file
new_json_path = './datasets/tiny-imagenet-200/tiny_imagenet_subset.json'
with open(new_json_path, 'w') as f:
    json.dump(new_dataset_info, f)

print(f"JSON file saved with {num_classes} selected classes at {new_json_path}")
