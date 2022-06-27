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
    def __init__(self, df_info, folder_data, image_size, transforms=None):
        self.df_info = df_info
        self.folder_data = folder_data
        self.image_size = image_size
        self.transforms = transforms

    def __len__(self):
        return len(self.df_info)

    def __getitem__(self, index):
        labels = self.df_info.iloc[index]["classes"]
        image_id = self.df_info.iloc[index]["imageid"]

        class_name = image_id.split("__")[0]
        image_name = image_id.split("__")[1]

        image_path = self.folder_data + "/" + image_id.split("__")[0] + f"/{image_name}"

        image = cv2.imread(image_path)
        image = image_resize(image, width=self.image_size, height=self.image_size)

        if self.transforms:
            sample = {
                "image": image
            }
            sample = self.transforms(**sample)
            image = sample["image"]

        X = torch.Tensor(image).permute(2, 0, 1)
        y = labels

        return X, y
