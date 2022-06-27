import torch
from torch.utils import data
import cv2


def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation = inter)

    return resized


class Dataset(data.Dataset):
    def __init__(self, df_info, image_size, transforms=None):
        self.df_info = df_info
        self.labels = {
            k:index for index, k in enumerate(set(self.df_info.class_name))
        }
        self.image_size = image_size
        self.transforms = transforms

    def __len__(self):
        return len(self.df_info)

    def __getitem__(self, index):
        class_name = self.df_info.iloc[index]["class_name"]
        image_path = self.df_info.iloc[index]["image_path"]
        image_path = f"{image_path}"

        image = cv2.imread(image_path)
        image = image_resize(image, width=self.image_size, height=self.image_size)

        if self.transforms:
            sample = {
                "image": image
            }
            sample = self.transforms(**sample)
            image = sample["image"]

        X = torch.Tensor(image).permute(2, 0, 1)
        y = self.labels[class_name]

        return X, y
