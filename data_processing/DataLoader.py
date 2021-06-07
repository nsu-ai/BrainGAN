""" Module for the data loading pipeline for the model to train """

import os
import numpy as np
import torch as th
from torch.utils.data import Dataset
from torchvision import transforms

def augmentor(data):
  transform2 = transforms.RandomRotation(45)
  
  transform3 = transforms.RandomRotation(90)
  
  transform4 = transforms.RandomRotation(120)
  
  transform5 = transforms.RandomRotation(180)
  
  transform6 = transforms.RandomRotation(330)
  
  transform7 = transforms.RandomRotation(270)
  
  transform8 = transforms.RandomRotation(300)
    
  transformers = [transform2, transform3, transform4, transform5, transform6, transform7, transform8]
  # transformers = [transform1, transform3, transform5, transform7]
  
  new_imgs = []
  new_labels = []

  for X, y in data:
    new_imgs.append(X)
    new_labels.append(th.tensor(y))
    for trans in transformers:
      new_imgs.append(trans(X))
      new_labels.append(th.tensor(y))
  
  return th.stack(new_imgs), th.stack(new_labels) 

class FlatDirectoryImageDataset(Dataset):
    """ pyth Dataset wrapper for the generic flat directory images dataset """

    def __setup_files(self):
        """
        private helper for setting up the files_list
        :return: files => list of paths of files
        """
        file_names = os.listdir(self.data_dir)
        files = []  # initialize to empty list

        for file_name in file_names:
            possible_file = os.path.join(self.data_dir, file_name)
            if os.path.isfile(possible_file):
                files.append(possible_file)

        # return the files list
        return files

    def __init__(self, data_dir, transform=None):
        """
        constructor for the class
        :param data_dir: path to the directory containing the data
        :param transform: transforms to be applied to the images
        """
        # define the state of the object
        self.data_dir = data_dir
        self.transform = transform

        # setup the files for reading
        self.files = self.__setup_files()

    def __len__(self):
        """
        compute the length of the dataset
        :return: len => length of dataset
        """
        return len(self.files)

    def __getitem__(self, idx):
        """
        obtain the image (read and transform)
        :param idx: index of the file required
        :return: img => image array
        """
        from PIL import Image

        # read the image:
        img_name = self.files[idx]
        if img_name[-4:] == ".npy":
            img = np.load(img_name)
            img = Image.fromarray(img)
        else:
            img = Image.open(img_name)

        # apply the transforms on the image
        if self.transform is not None:
            img = self.transform(img)

        # return the image:
        return img


class FoldersDistributedDataset(Dataset):
    """ pyth Dataset wrapper for the MNIST dataset """

    def __setup_files(self):
        """
        private helper for setting up the files_list
        :return: files => list of paths of files
        """

        dir_names = os.listdir(self.data_dir)
        class_ids = {}
        for i, dir_name in enumerate(dir_names):
          class_ids[dir_name] = i

        files = []
        class_names = []  # initialize to empty list

        for dir_name in dir_names:
            file_path = os.path.join(self.data_dir, dir_name)
            file_names = os.listdir(file_path)
            for file_name in file_names:
                possible_file = os.path.join(file_path, file_name)
                if os.path.isfile(possible_file):
                    files.append(possible_file)
                    class_names.append(class_ids[dir_name])

        # return the files list
        return files, class_names

    def __init__(self, data_dir, transform=None):
        """
        constructor for the class
        :param data_dir: path to the directory containing the data
        :param transform: transforms to be applied to the images
        """
        # define the state of the object
        self.data_dir = data_dir
        self.transform = transform

        # setup the files for reading
        self.files, self.classes = self.__setup_files()

    def __len__(self):
        """
        compute the length of the dataset
        :return: len => length of dataset
        """
        return len(self.files)

    def __getitem__(self, idx):
        """
        obtain the image (read and transform)
        :param idx: index of the file required
        :return: img => image array
        """
        from PIL import Image

        # read the image:
        img_name = self.files[idx]
        if img_name[-4:] == ".npy":
            img = np.load(img_name)
            img = Image.fromarray(img)
        else:
            img = Image.open(img_name)

        # apply the transforms on the image
        if self.transform is not None:
            img = self.transform(img)

        if img.shape[0] == 4:
            # ignore the alpha channel
            # in the image if it exists
            img = img[:3, :, :]

        # return the image:
        return img, self.classes[idx]


def get_transform(new_size=None):
    """
    obtain the image transforms required for the input data
    :param new_size: size of the resized images
    :return: image_transform => transform object from thVision
    """
    from torchvision.transforms import ToTensor, Normalize, Compose, Resize

    if new_size is not None:
        image_transform = Compose([
            Resize(new_size),
            transforms.Grayscale(1),
                ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
        ])

    else:
        image_transform = Compose([
            transforms.Grayscale(1),
                ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
        ])
    return image_transform


def get_data_loader(dataset, batch_size, num_workers, angle_augmentation=True):
    """
    generate the data_loader from the given dataset
    :param dataset: F2T dataset
    :param batch_size: batch size of the data
    :param num_workers: num of parallel readers
    :return: dl => dataloader for the dataset
    """
    from torch.utils.data import DataLoader

    if angle_augmentation:
        dl = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=augmentor
        )
    else:
        dl = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )

    return dl
