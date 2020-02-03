
import numpy
import torch
import pickle
import os
import glob
import argparse

from tqdm import *
from matplotlib import pyplot

import UNet
import loader


def load(output_folder, cuda):
    """
    Loads a previous network model from the given folder.

    :param output_folder: The path of the folder containing the network
    :param cuda: Wheter to use CUDA

    :returns : The parameters of the network
    """
    net_params = torch.load(os.path.join(output_folder, "params.net"),
                            map_location=None if cuda else "cpu")
    return net_params


class PredictionBuilder():
    """
    This class is used to create the final prediction from the predictions
    that are output by the network. This class stores the predictions in an output
    array to avoid memory overflow with the method `add_predictions` and then
    computes the mean prediction with the `return_prediction` method.

    :param _shape: The shape of the image
    :param _size: The size of the crops
    :param _step: The size of the step between each crops
    """
    def __init__(self, _shape, _size, _step):
        # Assgin member variables
        self._shape = _shape
        self._size = _size
        self._step = _step

        # Creates the output array
        self.ny, self.nx = int(numpy.ceil(_shape[0] / _step)), int(numpy.ceil(_shape[1] / _step))
        self.pred = numpy.zeros((self.ny*_step + (_size-_step), self.nx*_step + (_size-_step)))
        self.pixels = numpy.zeros((self.ny*_step + (_size-_step), self.nx*_step + (_size-_step)))

        # Position of the crops in the array
        self.j, self.i = 0, 0

    def add_predictions(self, predictions):
        """
        Method to store the prediction to the output array

        :param predictions: A numpy array of the predictions with size (batch_size, features, H, W)
        """
        for pred in predictions:
            # Stores the predictions in output array
            self.pred[self.j * self._step : self.j * self._step + self._size,
                      self.i * self._step : self.i * self._step + self._size] += pred[1]
            self.pixels[self.j * self._step : self.j * self._step + self._size,
                        self.i * self._step : self.i * self._step + self._size] += 1

            # Updates the position of the crops
            self.i += 1
            if self.j >= self.ny:
                self.i, self.j = 0, 0
            if self.i >= self.nx:
                self.i = 0
                self.j += 1

    def return_prediction(self):
        """
        Method to return the final prediction
        
        :returns : The average prediction of the overlapping predictions
        """
        self.pixels[self.pixels == 0] += 1
        return (self.pred / self.pixels)[:self._shape[0], :self._shape[1]]


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Setting the parameters of the prediction.")
    parser.add_argument("--network_path", type=str, default="pre-trained",
                        help="The path of the trained network. Defaults to the pre-trained model.")
    parser.add_argument("--images_folder", type=str, default="images_to_predict",
                        help="Folder containing the images to predict.")
    parser.add_argument("--cuda", action="store_true", default=False,
                        help="Wheter or not to use CUDA.")
    parser.add_argument("--size", type=int, default=128,
                        help="The size of the crops used for training the network.")
    parser.add_argument("--step", type=int, default=128,
                        help="The step between each crops.")
    parser.add_argument("--show", action="store_true", default=False,
                        help="Wheter or not to show each prediction.")

    print("Parsing the arguments ...")
    args = parser.parse_args()

    print("Loading the previous network ...")
    net_params = load(args.network_path, args.cuda)
    trainer_params = pickle.load(open(os.path.join(args.network_path, "params_trainer.pkl"), "rb"))
    network = UNet.UNet(in_channels=trainer_params["in_channels"], out_channels=trainer_params["out_channels"],
                        number_filter=trainer_params["number_filter"], depth=trainer_params["depth"],
                        size=trainer_params["size"])
    network.load_state_dict(net_params)
    if args.cuda:
        network = network.cuda()

    print("Initializing loaders ...")
    images_names = glob.glob(os.path.join(args.images_folder, "*.tif"))
    dataTest = loader.BatchDatasetLoader(batch_size=96, cuda=args.cuda, size=args.size, step=args.step)

    print("Predicting the probability from the input images ...")
    network.eval()

    for i_name in tqdm(images_names, total=len(images_names)):

        # Sets the current image to predict
        dataTest.BCL.set_current(i_name)

        # Creates a prediction builder instance to avoid memory overflow
        pb = PredictionBuilder(dataTest.BCL.image.shape, trainer_params["size"], args.step)

        for X in dataTest:
            # Prediction from the network
            pred = network.forward(X)

            # Convert to numpy array
            if args.cuda:
                prednumpy = pred.cpu().data.numpy()
            else:
                prednumpy = pred.data.numpy()

            # Adds the predictions to the prediction builder
            pb.add_predictions(prednumpy)

            # avoid memory leaks
            del X

        # Returns the prediction from the prediction builder
        pred = pb.return_prediction()

        # Fetches the image loaded in memory
        X = dataTest.BCL.image

        if args.show:
            fig, axes = pyplot.subplots(1, 2, figsize=(15,5), tight_layout=True, sharex=True, sharey=True)
            axes[0].imshow(X, cmap="gray", vmax=0.45*X.max())
            axes[1].imshow(pred, vmin=0, vmax=1)
            for ax, title in zip(axes.ravel(), ["Original", "Rings"]):
                ax.set_axis_off()
                ax.set_title(title)
            pyplot.show()
