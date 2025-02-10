import os
import pickle
from torch.utils.data import Dataset
from PIL import Image

class NABirds(Dataset):
    def __init__(self, root, split="train", transform=None):

        self.root_dir = root
        self.split = split
        self.transform = transform

        # Set pickle cache file
        cache_file = os.path.join(root, f"nabirds_split.pkl")

        # Load cache file if it exists
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                cached_data = pickle.load(f)
            self.data = cached_data["data"]
            self.label_to_index = cached_data["label_to_index"]
            self.index_to_label = cached_data["index_to_label"]
            print(f"Cached data loaded: {cache_file}")
        else:
            print("Cache file not found! Loading data...")

            # Set file paths
            images_file = os.path.join(root, "images.txt")  # UUID → Image path
            labels_file = os.path.join(root, "image_class_labels.txt")  # UUID → Label
            split_file = os.path.join(root, "train_test_split.txt")  # UUID → Train/Test

            # UUID → Image path mapping
            image_paths = {}
            with open(images_file, "r") as f:
                for line in f.readlines():
                    uuid, img_path = line.strip().split()
                    image_paths[uuid] = img_path  # {UUID: Image path}

            # UUID → Label mapping
            labels = {}
            with open(labels_file, "r") as f:
                for line in f.readlines():
                    uuid, label = line.strip().split()
                    labels[image_paths[uuid]] = int(label)  # {Image path: Label}

            # Train/Test split
            data = []
            with open(split_file, "r") as f:
                for line in f.readlines():
                    uuid, is_train = line.strip().split()
                    if (split == "train" and is_train == "1") or (split == "test" and is_train == "0"):
                        img_path = image_paths[uuid]
                        data.append((img_path, labels[img_path]))  # (Image path, Label)

            # Label indexing (e.g., "817" → 0, "860" → 1 ...)
            label_to_index = {label: idx for idx, label in enumerate(sorted(set(labels.values())))}
            index_to_label = {idx: label for label, idx in label_to_index.items()}

            # Save with transformed labels
            self.data = [(img, label_to_index[label]) for img, label in data]
            self.label_to_index = label_to_index
            self.index_to_label = index_to_label

            # Save with pickle (caching)
            with open(cache_file, "wb") as f:
                pickle.dump({
                    "data": self.data,
                    "label_to_index": self.label_to_index,
                    "index_to_label": self.index_to_label
                }, f)
            print(f"Data caching completed: {cache_file}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        img_full_path = os.path.join(self.root_dir, "images", img_path)
        image = Image.open(img_full_path).convert("RGB")

        # remove bottom 20px
        width, height = image.size
        image = image.crop((0, 0, width, height - 20))  # (left, upper, right, lower)

        if self.transform:
            image = self.transform(image)

        return image, label
