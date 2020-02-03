
import numpy
import torch
import pickle
import os
import time
import random

from matplotlib import pyplot
from collections import defaultdict
from tqdm import *

import UNet
import loader
import tools

# Ensuring that we always start from the same random state (for reproductibility)
random.seed(42)
numpy.random.seed(42)


class Trainer():
    """
    Creates the Trainer for the network architecture.

    :param output_folder: The path where to save the model with the name of the model
    :param in_channels: The number of channels in the input image
    :param out_channels: The number of output channels in the output segmentation
    :param depth: The depth of the network
    :param epochs: The number of epochs to train the network
    :param batch_size: The number of crops per batch
    :param lr: The learning rate of the optimizer
    :param do_data_augmentation: The probability of performing data augmentation
    :param cuda: Wheter to use CUDA
    :param number_filter: The number of filters in the first layer (2 ** number_filter)
    :param size: The size of the crops
    :param step: The step between each crops
    """
    def __init__(self, output_folder, in_channels=1, out_channels=2, depth=4, epochs=250, batch_size=72,
                    lr=0.001, do_data_augmentation=0.5, cuda=False, number_filter=4, size=128, step=112):
        self.output_folder = output_folder
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.data_aug = do_data_augmentation
        self.cuda = cuda
        self.number_filter = number_filter
        self.size = size
        self.step = step

        # Creates the output folder in the specified path
        try :
            os.makedirs(self.output_folder, exist_ok=True)
            pickle.dump(vars(self), open(os.path.join(self.output_folder, "params_trainer.pkl"), "wb"))
        except OSError as err:
            print("The name of the folder already exist! Try changing the name of the folder.")
            print("OSError", err)
            exit()

        # Creation of the network
        self.network = UNet.UNet(in_channels=self.in_channels, out_channels=self.out_channels,
                            number_filter=self.number_filter, depth=self.depth, size=self.size)
        if self.cuda:
            self.network = self.network.cuda()

        # Creation of the optimizer
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)

        # To keep track of the statistics
        self.stats = defaultdict(list)

        # Creation of the criterion
        self.criterion = torch.nn.MSELoss()
        if self.cuda:
            self.criterion = self.criterion.cuda()

        # To keep track of the network generalizing the most
        self.min_val_loss = numpy.inf

    def train(self, dataset_path=None, stats={"image_max" : 574.37, "image_min" : 0.0}):
        """
        Training method of the Trainer class.

        :param dataset_path: The path of the dataset
        :param stats: A dict containing keys {image_max, image_min} for the normalization
                      of the image in the training loop
        """
        # Load training and validation files
        if dataset_path == None:
            dataset_path = os.path.join(os.getcwd(), "dataset")
        training = pickle.load(open(os.path.join(dataset_path, "training.pkl"), "rb"))
        validation = pickle.load(open(os.path.join(dataset_path, "validation.pkl"), "rb"))

        pickle.dump(training, open(os.path.join(self.output_folder, "training.pkl"), "wb"), protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(validation, open(os.path.join(self.output_folder, "validation.pkl"), "wb"), protocol=pickle.HIGHEST_PROTOCOL)

        # Creating the dataloader for train and validation
        dataTrain = loader.DatasetLoader(training, batch_size=self.batch_size, do_data_augmentation=self.data_aug,
                        cuda=self.cuda, size=self.size, step=self.step, image_maximum=stats["image_max"],
                        image_minimum=stats["image_min"])
        dataTest = loader.DatasetLoader(validation, batch_size=self.batch_size, do_data_augmentation=0,
                        cuda=self.cuda, size=self.size, step=self.step, image_maximum=stats["image_max"],
                        image_minimum=stats["image_min"])

        for epoch in range(self.epochs):
            start = time.time()
            print("Starting epoch {}/{}".format(epoch + 1, self.epochs))

            # Shuffles the data
            dataTrain.new_epoch()
            dataTest.new_epoch()

            # Keep track of the loss of train and test
            statLossTrain, statLossTest = [], []

            # Puts the network in training mode
            self.network.train()
            iterations = 0
            for (X, y) in tqdm(dataTrain):

                # New batch we reset the optimizer
                self.optimizer.zero_grad()

                # Prediction and loss computation
                pred = self.network.forward(X)
                loss = self.criterion(pred, y)

                # Keeping track of statistics
                statLossTrain.append(loss.cpu().data.numpy())

                # Back-propagation and optimizer step
                loss.backward()
                self.optimizer.step()

                # To avoid memory leak
                del X, y, pred, loss
                iterations += 1
            dataTrain.len = iterations

            # Puts the network in evaluation mode
            self.network.eval()
            iterations = 0
            for (X, y) in tqdm(dataTest):
                # Prediction and computation loss
                pred = self.network.forward(X)
                loss = self.criterion(pred, y)

                # Keeping track of statistics
                statLossTest.append(loss.cpu().data.numpy())

                # To avoid memory leaks
                del X, y, pred, loss
                iterations += 1
            dataTest.len = iterations

            # Aggregate stats
            for key, func in zip(("trainMean", "trainMed", "trainMin", "trainStd"),
                                 (numpy.mean, numpy.median, numpy.min, numpy.std)):
                self.stats[key].append(numpy.sqrt(func(statLossTrain)))
            for key, func in zip(("testMean", "testMed", "testMin", "testStd"),
                                 (numpy.mean, numpy.median, numpy.min, numpy.std)):
                self.stats[key].append(numpy.sqrt(func(statLossTest)))

            # Loss curves
            pyplot.figure(figsize=(10, 7))
            pyplot.plot(self.stats["trainMean"], linewidth=2, color="#2678B2", label="Train")
            pyplot.fill_between(numpy.arange(len(self.stats["trainMean"])),
                                numpy.array(self.stats["trainMean"]) - numpy.array(self.stats["trainStd"]),
                                numpy.array(self.stats["trainMean"]) + numpy.array(self.stats["trainStd"]),
                                color="#AFC8E7", alpha=0.7)
            pyplot.plot(self.stats["testMean"], linewidth=2, color="#FD7F28", label="Validation")
            pyplot.fill_between(numpy.arange(len(self.stats["testMean"])),
                                numpy.array(self.stats["testMean"]) - numpy.array(self.stats["trainStd"]),
                                numpy.array(self.stats["testMean"]) + numpy.array(self.stats["trainStd"]),
                                color="#FDBA7D", alpha=0.7)
            pyplot.legend()
            pyplot.xlabel("Epoch")
            pyplot.ylabel("RMSE over the predicted scores")
            pyplot.ylim(0, 1)
            pyplot.savefig(os.path.join(self.output_folder, "epoch_{}.pdf".format(epoch)), bbox_inches="tight")
            pyplot.close("all")

            isBest = False
            if self.min_val_loss > self.stats["testMean"][-1]:
                print("New beat network ({} RMSE is better than the previous {})".format(self.stats["testMean"][-1], self.min_val_loss))
                self.min_val_loss = self.stats["testMean"][-1]
                self.save()
                isBest = True

            print("Epoch {} done!\n\tAvg loss train/validation : {} / {}\n\tTook {} seconds".format(epoch + 1, self.stats["trainMean"][-1], self.stats["testMean"][-1], time.time() - start))
            pickle.dump(self.stats, open(os.path.join(self.output_folder, "stats.pkl"), "wb"), protocol=pickle.HIGHEST_PROTOCOL)

    def save(self):
        """
        Saves the network parameters and optimizer state in the given folder
        """
        torch.save(self.network.state_dict(), os.path.join(self.output_folder, "params.net"))
        torch.save(self.optimizer.state_dict(), os.path.join(self.output_folder, "optimizer.data"))
        pickle.dump(self.stats, open(os.path.join(self.output_folder, "statsCkpt.pkl"), "wb"), protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":

    trainer = Trainer("Model0")
    trainer.train()
