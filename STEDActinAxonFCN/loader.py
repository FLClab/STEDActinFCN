
import numpy
import torch
import random

from tqdm import *
from skimage.external import tifffile

import tools

# Ensuring that we always start from the same random state (for reproductibility)
random.seed(42)
numpy.random.seed(42)


def convert2int(img):
    """
    Function to convert an uint image with range [0, 65536] in
    range [-32768, 32768].
    
    :param img: A ND numpy array of type uint
    
    :returns : A ND numpy array converted to int
    """
    if img.min() == 0:
        return img
    if img.dtype == "uint8":
        return img - 2**8 / 2
    elif img.dtype == "uint16":
        return img - 2**16 / 2
    elif img.dtype == "uint32":
        return img - 2**32 / 2
    else:
        return img


class CropsLoader:
    """
    Class to retreive the next crop from the image and label

    :param images_names: A list of file names of the images
    :param labels_names: A list of file names of the labels
    :param size: The size of the crops
    :param step: The step size between the crops
    """    
    def __init__(self, images_names, labels_names, size=128, step=112):
        self.images_names = images_names
        self.labels_names = labels_names
        self.size = size
        self.step = step

        # Sets the position pointer at the beginning of the list of images
        self.pos = 0

        # Sets the position of the crop
        self.i, self.j = 0, 0

        # Loads the image
        self.image = convert2int(tifffile.imread(self.images_names[self.pos]))
        self.label = tools.read_poly(self.labels_names[self.pos], self.image.shape)

    def next_crop(self):
        """
        Gets the next crop from the image. If all crops were used in the image
        the next image is loaded.
        
        :returns : The position of the pointer in the images list
                   A numpy array of the image crop 
                   A numpy array of the label crop
        """
        image_crop = self.image[self.j : self.j + self.size, self.i : self.i + self.size].astype(numpy.float32)
        label_crop = self.label[self.j : self.j + self.size, self.i : self.i + self.size].astype(numpy.float32)

        # Asserts the crops have the good shape
        if image_crop.size != self.size*self.size:
            image_crop = numpy.pad(image_crop, ((0, self.size - image_crop.shape[0]), (0, self.size - image_crop.shape[1])), "constant")
            label_crop = numpy.pad(label_crop, ((0, self.size - label_crop.shape[0]), (0, self.size - label_crop.shape[1])), "constant")

        # Update the position of the crop
        if (self.j + self.step >= self.image.shape[0]) and (self.i + self.step >= self.image.shape[1]):
            self.pos += 1
            if self.pos > len(self.images_names) - 1:
                return self.pos, numpy.array([]), numpy.array([])
            self.image = convert2int(tifffile.imread(self.images_names[self.pos]))
            self.label = tools.read_poly(self.labels_names[self.pos], self.image.shape)
            self.i, self.j = 0, 0
        elif self.i + self.step >= self.image.shape[1]:
            self.i = 0
            self.j += self.step
        else:
            self.i += self.step

        return self.pos, image_crop, label_crop

    def reset(self, images_names, labels_names):
        """
        Resets the loader with the new list of images and labels
        """
        self.images_names = images_names
        self.labels_names = labels_names
        self.pos = 0
        self.i, self.j = 0, 0
        self.image = convert2int(tifffile.imread(self.images_names[self.pos]))
        self.label = tools.read_poly(self.labels_names[self.pos], self.image.shape)


class DatasetLoader:
    """
    Class to create the next batch of images and labels 

    :param data: A python dict with keys {"image", "label"}
    :param batch_size: The batch size to output
    :param do_data_augmentation: The probability of doing data augmentation
    :param cuda: Wheter to use cuda
    :param size: Size of the crops
    :param step: The step between each crop (top left corner)
    :param image_maximum: The maximum from the image statistics
    :param image_minimum: The minimum from the image statistics
    """
    def __init__(self, data, batch_size, do_data_augmentation=0, cuda=False, size=128,
                    step=112, image_maximum=228.70, image_minimum=0):
        # Assign member variables
        self.data = data
        self.batch_size = batch_size
        self.data_aug = do_data_augmentation
        self.cuda = cuda
        self.size = size
        self.step = step
        self.image_maximum = image_maximum
        self.image_minimum = image_minimum

        # Find the sample files
        self.images_names = data["image"]
        self.labels_names = data["label"]

        # Set the position pointer at the beginning of the data
        self.pos = 0

        # Creates the crops loader
        self.CL = CropsLoader(self.images_names, self.labels_names,
                                size=self.size, step=self.step)


    def new_epoch(self):
        """
        Shuffles the dataset and resets the position of the pointer
        """
        temp = list(zip(self.images_names, self.labels_names))
        random.shuffle(temp)
        self.images_names, self.labels_names = zip(*temp)
        self.pos = 0
        self.CL.reset(self.images_names, self.labels_names)

    def __next__(self):

        # If there are no more images then stop iteration
        if self.pos >= len(self.images_names):
            raise StopIteration

        X, y = [], []
        for i in range(self.batch_size):

            # updates the position of the pointer and fetches an image and a label crop
            self.pos, image, label = self.CL.next_crop()
            if image.size == 0:
                # Reached the end of the dataset
                if len(X) == 0: raise StopIteration
                else: break

            # Rescale the image
            image -= self.image_minimum
            image /= (0.8 * (self.image_maximum - self.image_minimum))
            image = numpy.clip(image, 0, 1)

            # Data Augmentation
            if self.data_aug > 0:
                if random.random() < self.data_aug:
                    # left-right flip
                    image = numpy.fliplr(image).copy()
                    label = numpy.fliplr(label).copy()

                if random.random() < self.data_aug:
                    # up-down flip
                    image = numpy.flipud(image).copy()
                    label = numpy.flipud(label).copy()

                if random.random() < self.data_aug:
                    # intensity scale
                    intensityScale = numpy.clip(numpy.random.lognormal(0.01, numpy.sqrt(0.01)), 0, 1)
                    image = numpy.clip(image *  intensityScale, 0, 1)

                if random.random() < self.data_aug:
                    # gamma adaptation
                    gamma = numpy.clip(numpy.random.lognormal(0.005, numpy.sqrt(0.005)), 0, 1)
                    image = numpy.clip(image**gamma, 0, 1)

            X.append(image)
            y.append(label)

        X = numpy.array(X)
        y = numpy.array(y)

        Xtorch = torch.tensor(X)
        Xtorch = Xtorch.unsqueeze(1)
        ytorch = torch.tensor(y)
        if self.cuda:
            Xtorch = Xtorch.cuda()
            ytorch = ytorch.cuda()

        return Xtorch, ytorch

    def __iter__(self):
        return self

    def __len__(self):
        # The shape of the images are 224x224
        return int(numpy.ceil(((len(self.images_names) * (224 // self.step) ** 2) / self.batch_size)))


#########################################################
#########################################################
# BATCH
#########################################################
#########################################################

class BatchCropsLoader:
    """
    Class to load the next crop from the image

    :param size: The size of the crops
    :param step: The size of the step between each crops
    :param image_max: The maximum of the images in training dataset
    :param image_min: The minimum of the images in training dataset
    """
    def __init__(self, size, step, image_max, image_min):
        # Assigning member variables
        self.size = size
        self.step = step
        self.image_max = image_max
        self.image_min = image_min

        self.no_more_crops = False

    def set_current(self, image_name):
        """
        Method to load the image in memory

        :param image_name: Path of the image
        """
        # Sets the position of the crop
        self.j ,self.i = 0, 0

        # loads the image
        self.image = convert2int(tifffile.imread(image_name)).astype(numpy.float32)

        # Computes the number of crops in x and y
        self.ny = numpy.ceil(self.image.shape[0] / self.step)
        self.nx = numpy.ceil(self.image.shape[1] / self.step)

        # rescale the image
        self.image -= self.image_min
        self.image /= (0.8 * (self.image_max - self.image_min))
        self.image = numpy.clip(self.image, 0, 1)

    def next_crop(self):
        """
        Method to release the next crop from the image
        
        :returns : A numpy array of the image crop 
                   Wheter there are crops left 
        """
        image_crop = self.image[self.j : self.j + self.size, self.i : self.i + self.size]

        if image_crop.size != self.size*self.size:
            image_crop = numpy.pad(image_crop, ((0, self.size - image_crop.shape[0]), (0, self.size - image_crop.shape[1])), "constant")
        self.update()

        return image_crop, not self.no_more_crops

    def update(self):
        """
        Method to update the position of the crop in the image
        """
        if (self.j + self.step >= self.image.shape[0]) and (self.i + self.step >= self.image.shape[1]):
            self.no_more_crops = True
        elif self.i + self.step >= self.image.shape[1]:
            self.i = 0
            self.j += self.step
        else:
            self.i += self.step


class BatchDatasetLoader:
    """
    Class to load the next batch for inference

    :param batch_size: The size of the batch
    :param cuda: Wheter to use CUDA
    :param size: The size of the crops
    :param step: The size of the step between each crops
    :param stats: A dict containing keys {image_max, image_min} for the normalization
                  of the image from the statistics of the training images
    """
    def __init__(self, batch_size, cuda=False, size=128, step=112, stats={"image_max" : 228.70, "image_min" : 0.0}):
        # Assign member variables
        self.batch_size = batch_size
        self.cuda = cuda
        self.size = size
        self.step = step

        # Creates the batch crops loader
        self.BCL = BatchCropsLoader(self.size, self.step, stats["image_max"], stats["image_min"])

    def __next__(self):

        # Raises a stop iteration if there are no more crops
        if self.BCL.no_more_crops:
            self.BCL.no_more_crops = False
            raise StopIteration

        X = []
        for _ in range(self.batch_size):
            image, is_left = self.BCL.next_crop()
            X.append(image)
            if not is_left:
                break

        X = numpy.array(X)

        Xtorch = torch.Tensor(X)
        Xtorch = Xtorch.unsqueeze(1)
        if self.cuda:
            Xtorch = Xtorch.cuda()

        return Xtorch

    def __iter__(self):
        return self

    def __len__(self):
        return self.BCL.nx * self.BCL.ny
