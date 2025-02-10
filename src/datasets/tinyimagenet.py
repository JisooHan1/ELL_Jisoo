import os
import pickle
from torch.utils.data import Dataset
from PIL import Image

class TinyImageNet200(Dataset):
    def __init__(self, root, split="train", transform=None, use_pickle=True):

        self.root = root
        self.split = split
        self.transform = transform
        self.use_pickle = use_pickle
        self.pickle_file = os.path.join(root, f"tiny_imagenet_{split}.pkl")

        if use_pickle and os.path.exists(self.pickle_file):
            print(f"Loading {split} dataset from pickle file: {self.pickle_file}")
            with open(self.pickle_file, "rb") as f:
                self.data = pickle.load(f)
        else:
            print(f"Processing {split} dataset from raw images...")
            self.data = self.process_data()
            if use_pickle:
                print(f"Saving {split} dataset as pickle: {self.pickle_file}")
                with open(self.pickle_file, "wb") as f:
                    pickle.dump(self.data, f)

    def process_data(self):
        dataset = []
        class_to_idx = {}

        if self.split == "train":
            train_dir = os.path.join(self.root, "train")
            classes = sorted(os.listdir(train_dir))
            class_to_idx = {cls: i for i, cls in enumerate(classes)}

            for cls in classes:
                class_path = os.path.join(train_dir, cls, "images")
                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)
                    dataset.append((img_path, class_to_idx[cls]))

        elif self.split == "val":
            val_dir = os.path.join(self.root, "val", "images")
            annotations_path = os.path.join(self.root, "val", "val_annotations.txt")

            train_dir = os.path.join(self.root, "train")
            classes = sorted(os.listdir(train_dir))
            class_to_idx = {cls: i for i, cls in enumerate(classes)}

            if os.path.exists(annotations_path):
                with open(annotations_path, "r") as f:
                    val_labels = {line.split("\t")[0]: line.split("\t")[1] for line in f.readlines()}
                for img_name in os.listdir(val_dir):
                    img_path = os.path.join(val_dir, img_name)
                    if img_name in val_labels:
                        cls = val_labels[img_name]
                        label = class_to_idx.get(cls, -1)
                        dataset.append((img_path, label))

        return dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
