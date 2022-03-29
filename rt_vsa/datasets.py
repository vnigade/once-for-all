
import torch.utils.data as torch_data
import pandas as pd
from PIL import Image
import os


EXTENSION_LIST = ["jpg", "jpeg", "png"]


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


def make_dataset(root_dir, classidx_to_class, class_indices):
    # Read only class_indices data
    data_path = []
    label_indices = []

    # If class_indices are not mentioned than include every class
    if class_indices is None:
        class_indices = list(range(len(classidx_to_class)))

    # For all valid classes, read image file names
    for class_idx in class_indices:
        class_name = classidx_to_class[class_idx]
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        files = os.listdir(class_dir)
        for file_name in files:
            ext = file_name.split(".")[-1]
            if ext.lower() not in EXTENSION_LIST:
                continue
            file_path = os.path.join(class_dir, file_name)
            data_path.append(file_path)
            label_indices.append(class_idx)

    return data_path, label_indices


class SubsetDataset(torch_data.Dataset):
    def __init__(self, root_dir,
                 transform=None,
                 classes=None,
                 labels_file="labels.txt",
                 loader=default_loader,
                 data_type="train") -> None:

        # Read label file
        df_labels = pd.read_csv(os.path.join(root_dir, labels_file),
                                delimiter=",",
                                header=None,
                                names=["class_name", "class_desc"])

        labels = {idx: row["class_name"] for idx,
                  row in df_labels.iterrows()}

        # Create dataset for specified classes
        data_dir = os.path.join(root_dir, data_type)
        self.data, self.label_indices = make_dataset(
            data_dir, classidx_to_class=labels, class_indices=classes)
        self._transform = transform
        self._loader = loader

    def __getitem__(self, index: int):
        img = self._loader(self.data[index])
        if self._transform is not None:
            img = self._transform(img)
        label_idx = self.label_indices[index]
        return img, label_idx

    def __len__(self) -> int:
        return len(self.data)
