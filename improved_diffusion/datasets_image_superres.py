import os
from typing import List, Tuple

import blobfile as bf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def list_image_files_recursively(
        data_dir: str):
    """
    List image files in a data directory.
    """
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(list_image_files_recursively(full_path))
    return results


def load_data_superres(
        *,
        data_dir: str,
        batch_size: int,
        image_size_hr: int,
        image_size_lr: int,
        num_channels: int,
        class_cond: bool = False,
        num_class: int = None,
        deterministic: bool = False,
        crop: bool = False,
        droplast: bool = True
        ):
    """
    Creates a generator over (images, kwargs) pairs given a dataset.
    Reads image of size 'image_size_hr' in the dataset and returns it,
    along with a downscaled image of size 'image_size_lr'.

    Each image is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    Inputs:
    -------
        data_dir (str): a dataset directory.

        batch_size (int): the batch size of each returned pair.

        image_size_hr (int): the size to which images are resized.

        image_size_lr (int): the size of the low-resolution image

        num_channels (int): nbr of channels of the input image.

        class_cond (bool): if True, include a "y" key in returned dicts
            for class label. If classes are not available and this is
            true, an exception will be raised.

            Assume classes are the first part of the filename, before an
            underscore.

        num_class (int): number of data classes

        deterministic (bool): if True, yield results in a
            deterministic order.

        crop (bool): if True, randomly crops the image
            to the desired image_size
    """
    # Check inputs
    if not data_dir:
        raise ValueError("Unspecified data directory!")
    if not os.path.exists(data_dir):
        raise ValueError("The specified data directory does not exist!")

    all_files = list_image_files_recursively(data_dir)
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
        if num_class != len(sorted_classes):
            raise ValueError('Difference between the number of classes when reading the data and the input number of classes.')

    dataset = ImageDatasetSuperres(
        image_size_hr,
        image_size_lr,
        num_channels,
        all_files,
        classes=classes,
        crop=crop)

    return (
        DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=1,
            drop_last=droplast,
        )
        if deterministic
        else DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=1,
            drop_last=droplast,
         )
    )


class ImageDatasetSuperres(Dataset):

    def __init__(
            self,
            resolution: int,
            low_resolution: int,
            num_channels: int,
            image_paths: List[str],
            classes: List[str] = None,
            plot: bool = False,
            crop: bool = False):
        '''
        Inputs:
        -------
            resolution (int): desired size for the images, not
                necessarily the native resolution.

            low_resolution (int): size of the downsampled image

            num_channels (int): nbr of channels of the input
                image.

            images_paths (List[str]): list of all the urls
                associated with the images.

            classes (List[str]): if not None, it is the
                label associated to each image.

            crop (bool): if True, randomly crops the image
                to the desired image resolution.
        '''
        super().__init__()
        self.resolution = resolution
        self.low_resolution = low_resolution
        self.num_channels = num_channels
        self.local_images = image_paths
        self.local_classes = None if classes is None else classes
        self.plot = plot
        self.crop = crop

    def __len__(self):
        return len(self.local_images)

    def __getitem__(
            self,
            idx: int
            ) -> Tuple[np.ndarray, dict]:
        """
        Outputs the image in format (C, H, W) along with a dictionary
        containing the image label (if class_cond is True) and the
        low-resolution image.
        """
        path = self.local_images[idx]

        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()

        if self.crop:
            width, height = pil_image.size
            if width < self.resolution or height < self.resolution:
                raise ValueError('The data cannot be cropped to the desired resolution!')
            # Randomly crop a portion of the image
            left, bottom = np.random.randint(0, width - self.resolution),\
                np.random.randint(0, height - self.resolution)
            pil_image = pil_image.crop((left, bottom,
                                        left + self.resolution,
                                        bottom + self.resolution))

        else:
            # We are not on a new enough PIL to support the `reducing_gap`
            # argument, which uses BOX downsampling at powers of two first.
            # Thus, we do it by hand to improve downsample quality.
            while min(*pil_image.size) >= 2 * self.resolution:
                pil_image = pil_image.resize(
                    tuple(x // 2 for x in pil_image.size),
                    resample=Image.BOX
                )

            scale = self.resolution / min(*pil_image.size)
            pil_image = pil_image.resize(
                tuple(round(x * scale) for x in pil_image.size),
                resample=Image.BICUBIC
            )

        # This step converts the image to an array in [0, 255]
        if self.num_channels == 3:
            arr = np.array(pil_image.convert("RGB"))
        elif self.num_channels == 1:
            # Convert to grayscale
            arr = np.expand_dims(np.array(pil_image.convert('L')), axis=2)
        else:
            raise ValueError('We require either 1 or 3 channels.')

        if arr.shape != (self.resolution, self.resolution, self.num_channels):
            raise ValueError('The current image is not of the right size.')

        crop_y = (arr.shape[0] - self.resolution) // 2
        crop_x = (arr.shape[1] - self.resolution) // 2
        arr = arr[crop_y: crop_y + self.resolution,
                  crop_x: crop_x + self.resolution]

        # This step rescales the array to [-1, 1]
        arr = arr.astype(np.float32) / 127.5 - 1

        if self.plot:
            plt.imshow(arr)
            plt.colorbar()
            plt.show()

        # This step reorders the array dimensions
        arr = np.transpose(arr, [2, 0, 1])

        # Create dictionary with label/low_res data
        out_dict = {}

        # Add class data
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)

        # Add LR data
        arr = torch.tensor(arr)
        arr = arr.unsqueeze(0)
        out_dict["low_res"] = F.interpolate(arr, self.low_resolution, mode="area").squeeze(0)

        return arr.squeeze(0), out_dict
