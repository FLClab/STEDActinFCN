
import numpy
import torch
import random

from tqdm import *
from skimage.external import tifffile

import tools


# Ensuring that we always start from the same random state (for reproductibility)
random.seed(42)
numpy.random.seed(42)

# Sets default value for the channels of the images
ACTIN, AXON, DENDRITE = 0, 1, 2


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
    Class to load the next crop from the image and the label array.

    :param images_names: A list of file names of the images
    :param labels_names: A list of file names of the labels
    :param size: The size of the crops
    :param step: The step size between the crops
    """
    def __init__(self, images_names, labels_names, size=128, step=112):
        # Assign member variables
        self.images_names = images_names
        self.labels_names = labels_names
        self.size = size
        self.step = step

        # Sets the position pointer at the beginning of the list of images
        self.pos = 0

        # Sets the position of the crop
        self.i, self.j = 0, 0

        # Loads the image and the crop
        self.image = convert2int(tifffile.imread(self.images_names[self.pos])[ACTIN])
        self.label = tools.read_poly(self.labels_names[self.pos], self.image.shape)

    def next_crop(self):
        """
        Gets the next crop from the image. If all crops were used in the image
        the next image is loaded. If a label crop doesn't contain at least 1% of
        its total area labeled the crop is simply skipped.
        
        :returns : The position in the list of images  
                   A numpy array of the image crop 
                   A numpy array of the label crop
        """
        image_crop = self.image[self.j : self.j + self.size, self.i : self.i + self.size].astype(numpy.float32)
        label_crop = self.label[:, self.j : self.j + self.size, self.i : self.i + self.size].astype(numpy.float32)

        # Only keep crops that have 1% label and update the position of the crop window
        if label_crop.sum() > 0.01 * self.size * self.size:
            update = self.update_position()
            if update == False:
                return self.pos, numpy.array([]), numpy.array([])
        else:
            update = self.update_position()
            if update == False:
                return self.pos, numpy.array([]), numpy.array([])
            while not (label_crop.sum() > 0.01 * self.size * self.size):
                image_crop = self.image[self.j : self.j + self.size, self.i : self.i + self.size].astype(numpy.float32)
                label_crop = self.label[:, self.j : self.j + self.size, self.i : self.i + self.size].astype(numpy.float32)
                update = self.update_position()
                if update == False:
                    return self.pos, numpy.array([]), numpy.array([])

        # Asserts the crops have the good shape
        if image_crop.size != self.size*self.size:
            image_crop = numpy.pad(image_crop, ((0, self.size - image_crop.shape[0]), (0, self.size - image_crop.shape[1])), "constant")
            label_crop = numpy.pad(label_crop, ((0, 0), (0, self.size - label_crop.shape[1]), (0, self.size - label_crop.shape[2])), "constant")

        return self.pos, image_crop, label_crop

    def update_position(self):
        """
        Updates the position of the crop
        """
        # Updates the position of the crops
        if self.i + self.step > self.image.shape[1]:
            self.i = 0
            self.j += self.step
        else:
            self.i += self.step

        # Update image and label if all crops were used
        if self.j + self.step > self.image.shape[0]:
            self.pos += 1
            if self.pos > len(self.images_names) - 1:
                return False
            self.image = convert2int(tifffile.imread(self.images_names[self.pos])[ACTIN])
            self.label = tools.read_poly(self.labels_names[self.pos], self.image.shape)

            self.i, self.j = 0, 0

    def reset(self, images_names, labels_names):
        """
        Resets the loader with the new list of images and labels
        """
        self.images_names = images_names
        self.labels_names = labels_names
        self.pos = 0
        self.i, self.j = 0, 0
        self.image = convert2int(tifffile.imread(self.images_names[self.pos])[ACTIN])
        self.label = tools.read_poly(self.labels_names[self.pos], self.image.shape)


class DatasetLoader:
    """
    Class to create the next batch to be infered by the network 

    :param data: A python dict with keys {"images", "labels"} containing list of filenames
    :param batch_size: The batch size to output
    :param do_data_augmentation: The probability of doing data augmentation
    :param cuda: Wheter to use cuda
    :param size: Size of the crops
    :param step: The step between each crop (top left corner)
    :param image_maximum: The maximum from the image statistics
    :param image_minimum: The minimum from the image statistics
    """
    def __init__(self, data, batch_size, do_data_augmentation=0, cuda=False, size=128,
                    step=112, image_maximum=540.15, image_minimum=0):
        # Assign member variables
        self.data = data
        self.batch_size = batch_size
        self.data_aug = do_data_augmentation
        self.cuda = cuda
        self.size = size
        self.step = step
        self.image_maximum = image_maximum
        self.image_minimum = image_minimum

        # Loads the names of the images and the labels
        self.images_names = data["images"]
        self.labels_names = data["labels"]

        # Set the position pointer at the beginning of the data
        self.pos = 0

        # Creates the crops loader
        self.CL = CropsLoader(self.images_names, self.labels_names,
                                size=self.size, step=self.step)

        # Defines the length of the loader, is updated in outside loop
        self.len = 0

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
                    label = numpy.array([numpy.fliplr(l).copy() for l in label])

                if random.random() < self.data_aug:
                    # up-down flip
                    image = numpy.flipud(image).copy()
                    label = numpy.array([numpy.flipud(l).copy() for l in label])

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
        return self.len


##################################################################################################################
##################################################################################################################
# BATCH
##################################################################################################################
##################################################################################################################

class BatchCropsLoader:
    """
    Class to load the next crop from the input image 

    :param images_min: Minimum of images in training dataset
    :param images_max: Maximum of images in training dataset
    :param size: The size of the crops
    :param step: The size of the step between each crops
    """    
    def __init__(self, image_min, image_max, size, step):
        # Assign member variables
        self.image_min = image_min
        self.image_max = image_max
        self.size = size
        self.step = step

        self.no_more_crops = False

    def set_current(self, image_name):
        """
        Method to load the image in memory

        :param image_name: Path of the image
        """
        # Sets the position of the crops
        self.j, self.i = 0, 0

        # Loads the image
        self.image = convert2int(tifffile.imread(image_name)).astype(numpy.float)

        # Computes the number of crops in x and y
        self.ny = numpy.ceil(self.image.shape[0] / self.step)
        self.nx = numpy.ceil(self.image.shape[1] / self.step)

        # Rescale the image
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
    Class to load the next batch to be infered

    :param batch_size: The size of the batch
    :param cuda: Wheter to use cuda
    :param size: The size of the crops
    :param step: The size of the step between each crops
    :param stats: A python dict with keys {"image_max", "image_min"} for the normalization
                  of the image in the training loop
    """
    def __init__(self, batch_size, cuda=False, size=128, step=112, stats={"image_max" : 574.37, "image_min" : 0.0}):
        # Assign member variables
        self.batch_size = batch_size
        self.cuda = cuda
        self.size = size
        self.step = step

        # Creates the batch crops loader
        self.BCL = BatchCropsLoader(stats["image_min"], stats["image_max"],
                                    self.size, self.step)

    def __next__(self):

        # Raises a stop iterations if there are no more crops
        if self.BCL.no_more_crops:
            self.BCL.no_more_crops = False
            raise StopIteration

        X, indices = [], []
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
