import os
from torch.utils.data import Dataset
from PIL import Image

class FGVC_Aircraft(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        image_list_file = os.path.join(root_dir, f"data/images_{split}.txt")
        label_file = os.path.join(root_dir, f"data/images_variant_{split}.txt")

        with open(image_list_file, "r") as f:
            self.image_files = [line.strip() + ".jpg" for line in f.readlines()]

        self.labels = {}
        with open(label_file, "r") as f:
            for line in f.readlines():
                parts = line.strip().split(" ", 1)
                self.labels[parts[0] + ".jpg"] = parts[1]

        self.label_to_index = {label: idx for idx, label in enumerate(sorted(set(self.labels.values())))}
        self.index_to_label = {idx: label for label, idx in self.label_to_index.items()}

        self.data = [(img, self.label_to_index[self.labels[img]]) for img in self.image_files if img in self.labels]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, label = self.data[idx]
        img_path = os.path.join(self.root_dir, "data/images", img_name)
        image = Image.open(img_path).convert("RGB")

        # remove bottom 20px
        width, height = image.size
        image = image.crop((0, 0, width, height - 20))  # (left, upper, right, lower)

        if self.transform:
            image = self.transform(image)

        return image, label