import torch
from torch.utils import data
import cv2
import numpy as np

import os
import sys
path = os.path.dirname(__file__)
root_folder = os.path.join(
    os.path.abspath(path).split("domain_calibration")[0],
    "domain_calibration"
)
sys.path.insert(0, root_folder)

from src.style_transfer.FDA import FDA_source_to_target_np


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
    def __init__(self, df_info, folder_data, image_size_ratio, 
                input_size, transforms=None, image_id_source=None,
                path_root_source=None, 
                L=0.001,
                export_main_domain=True):
        self.df_info = df_info
        self.folder_data = folder_data
        self.image_size_ratio = image_size_ratio
        self.input_size = input_size
        self.transforms = transforms
        self.image_id_source = image_id_source
        self.path_root_source = path_root_source
        self.L = L
        self.export_main_domain = export_main_domain

    def __len__(self):
        return len(self.df_info)

    def __getitem__(self, index):

        image_source_name = self.image_id_source.split("__")[1]
        image_source_path = self.path_root_source + "/" + self.image_id_source.split("__")[0] + f"/{image_source_name}"

        im_src_origin = cv2.imread(image_source_path)

        labels = self.df_info.iloc[index]["classes"]
        image_id = self.df_info.iloc[index]["imageid"]

        image_name = image_id.split("__")[1]

        image_path = self.folder_data + "/" + image_id.split("__")[0] + f"/{image_name}"

        image = cv2.imread(image_path)
        image = image_resize(image, width=self.image_size_ratio, height=self.image_size_ratio)
        if image.shape[0] < 224 and image.shape[0] < 224:
            image = cv2.resize(image, (self.input_size, self.input_size), interpolation = cv2.INTER_AREA)

        if self.export_main_domain == False:
            im_src = cv2.resize(im_src_origin, (224, 224), interpolation = cv2.INTER_AREA)
            im_trg = cv2.resize(image, (224, 224), interpolation = cv2.INTER_AREA)

            im_src = np.asarray(im_src, np.float32)
            im_trg = np.asarray(im_trg, np.float32)
            
            # im_src = im_src.transpose((2, 0, 1))
            # im_trg = im_trg.transpose((2, 0, 1))

            src_in_trg = FDA_source_to_target_np(im_trg, im_src, L=self.L)

            # src_in_trg = src_in_trg.reshape(im_src.shape).transpose((1,2,0))
            src_in_trg = src_in_trg.reshape(im_src.shape)
            
            image = np.array(np.clip(src_in_trg, 0.0, 255.0), np.uint8)

        if self.transforms:
            sample = {
                "image": image
            }
            sample = self.transforms(**sample)
            image = sample["image"]

        X = torch.Tensor(image).permute(2, 0, 1)
        y = labels

        return X, y, self.df_info.iloc[index]
