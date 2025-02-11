import os
import pickle
from torch.utils.data import Dataset
from PIL import Image

class FGVC_Aircraft(Dataset):
    def __init__(self, root, split="train", transform=None, subset=0):
        """
        Args:
            root (str): Root directory of dataset.
            split (str): One of "train", "val", "test".
            transform (callable, optional): Optional transform to be applied on a sample.
            subset (int): Which subset to use (0 or 1).
            num_subsets (int): Total number of subsets.
        """
        self.root_dir = root
        self.split = split
        self.transform = transform
        self.subset = subset

        # Pickle file path
        self.pickle_file = os.path.join(root, f"data/dataset_{split}_subset{subset}.pkl")

        # Load pickle if it exists
        if os.path.exists(self.pickle_file):
            with open(self.pickle_file, "rb") as f:
                self.data, self.label_to_index, self.index_to_label = pickle.load(f)
            print(f"Loaded dataset from pickle: {self.pickle_file}")
        else:
            print("Processing dataset from scratch...")
            self._process_dataset()
            self._save_pickle()

    def _process_dataset(self):
        """Function to load and process the dataset"""
        image_list_file = os.path.join(self.root_dir, f"data/images_{self.split}.txt")
        label_file = os.path.join(self.root_dir, f"data/images_variant_{self.split}.txt")

        with open(image_list_file, "r") as f:
            self.image_files = [line.strip() + ".jpg" for line in f.readlines()]

        self.labels = {}
        with open(label_file, "r") as f:
            for line in f.readlines():
                parts = line.strip().split(" ", 1)
                self.labels[parts[0] + ".jpg"] = parts[1]

        # Create and index class list
        all_labels = sorted(set(self.labels.values()))
        self.label_to_index = {label: idx for idx, label in enumerate(all_labels)}
        self.index_to_label = {idx: label for label, idx in self.label_to_index.items()}

        # Split classes in half
        num_classes = len(all_labels)
        print(f"Number of classes: {num_classes}")
        class_split = num_classes // 2

        if self.subset == 0:
            selected_classes = set(all_labels[:class_split])
        else:
            selected_classes = set(all_labels[class_split:])

        # Create dataset containing only selected classes
        self.data = [
            (img, self.label_to_index[self.labels[img]])
            for img in self.image_files if img in self.labels and self.labels[img] in selected_classes
        ]

    def _save_pickle(self):
        """Save the processed dataset to a pickle"""
        with open(self.pickle_file, "wb") as f:
            pickle.dump((self.data, self.label_to_index, self.index_to_label), f)
        print(f"Saved dataset to pickle: {self.pickle_file}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, label = self.data[idx]
        img_path = os.path.join(self.root_dir, "data/images", img_name)
        image = Image.open(img_path).convert("RGB")

        # Remove bottom 20px
        width, height = image.size
        image = image.crop((0, 0, width, height - 20))  # (left, upper, right, lower)

        if self.transform:
            image = self.transform(image)

        return image, label



# import os
# from torch.utils.data import Dataset
# from PIL import Image

# class FGVC_Aircraft(Dataset):
#     def __init__(self, root, split="train", transform=None):
#         self.root_dir = root
#         self.split = split
#         self.transform = transform

#         image_list_file = os.path.join(root, f"data/images_{split}.txt")
#         label_file = os.path.join(root, f"data/images_variant_{split}.txt")

#         with open(image_list_file, "r") as f:
#             self.image_files = [line.strip() + ".jpg" for line in f.readlines()]

#         self.labels = {}
#         with open(label_file, "r") as f:
#             for line in f.readlines():
#                 parts = line.strip().split(" ", 1)
#                 self.labels[parts[0] + ".jpg"] = parts[1]

#         self.label_to_index = {label: idx for idx, label in enumerate(sorted(set(self.labels.values())))}
#         self.index_to_label = {idx: label for label, idx in self.label_to_index.items()}

#         self.data = [(img, self.label_to_index[self.labels[img]]) for img in self.image_files if img in self.labels]

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         img_name, label = self.data[idx]
#         img_path = os.path.join(self.root_dir, "data/images", img_name)
#         image = Image.open(img_path).convert("RGB")

#         # remove bottom 20px
#         width, height = image.size
#         image = image.crop((0, 0, width, height - 20))  # (left, upper, right, lower)

#         if self.transform:
#             image = self.transform(image)

#         return image, label