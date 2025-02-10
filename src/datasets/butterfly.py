import os
import pickle
from torch.utils.data import Dataset
from PIL import Image
from sklearn.model_selection import train_test_split


class ButterflyDataset(Dataset):
    def __init__(self, root, transform=None, split="train", train_ratio=0.8, random_seed=42):

        self.root_dir = root
        self.transform = transform

        # Check if the folder exists
        if not os.path.exists(root):
            raise FileNotFoundError(f"Dataset directory {root} not found.")

        # Read folders for each class
        classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

        # Cache file storage location (stored inside root_dir)
        cache_filename = f"butterfly200_split_{int(train_ratio*100)}_{int((1-train_ratio)*100)}.pkl"
        self.split_cache_file = os.path.join(root, cache_filename)

        # Load cache file if it exists (for speed improvement)
        if os.path.exists(self.split_cache_file):
            with open(self.split_cache_file, "rb") as f:
                split_data = pickle.load(f)
            train_images, test_images, train_labels, test_labels = split_data
        else:
            # Collect all image file paths
            all_images = []
            all_labels = []

            for cls_name in classes:
                cls_path = os.path.join(root, cls_name)
                for img_name in os.listdir(cls_path):
                    img_path = os.path.join(cls_path, img_name)
                    if os.path.isfile(img_path):  # Check if it is a file to remove unnecessary paths
                        all_images.append(img_path)
                        all_labels.append(self.class_to_idx[cls_name])

            # Train/Test Split
            train_images, test_images, train_labels, test_labels = train_test_split(
                all_images, all_labels, train_size=train_ratio, random_state=random_seed, stratify=all_labels
            )

            # Save the split data (cache)
            with open(self.split_cache_file, "wb") as f:
                pickle.dump((train_images, test_images, train_labels, test_labels), f)

        # Select dataset according to split
        if split == "train":
            self.image_paths = train_images
            self.labels = train_labels
        elif split == "test":
            self.image_paths = test_images
            self.labels = test_labels
        else:
            raise ValueError("split must be 'train' or 'test'")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image and apply transformations
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label
