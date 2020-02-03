# FCN-Dendrite

This repository contains the network architecture of the FCN-dendrite. This is the network architecture presented in [Deep learning-based image analysis reveals activity-dependent reorganization of the periodical actin lattice in dendrites](http://www.google.com). The scripts to train a model from scratch and to use a pre-trained model for the user to predict images of his/her own are provided.

## Dependencies 

* install [PyTorch](https://pytorch.org/)
* install the following python librairies
```
pip install numpy scikit-image matplotlib 
```

## Train a network from scratch using provided example images

To train a network from scratch, the user should use the provided script `train.py`. 
```
python train.py
```

Using defaults value will create a folder named `Model0` in the working directory. The network should then start training with the 4 images provided (2 training image and 2 validation image) in the folder `dataset/data`.

Following training, the `Model0` folder will contain 
* `params.net` : Parameters of the network with best generalization 
* `params_trainer.pkl` : Parameters used for training
* `optimizer.data` : Parameters of the optimizers of the network with best generalization 
* `statsCkpt.pkl` : Statistics of the model with best generalization and previous
* `stats.pkl` : Statistics for all epochs
* `epoch_X.pdf` : Learning curves following each epoch

To change the default parameters the user should specify the parameters when creating the `Trainer()` object in the `train.py` script. See information provided with the `Trainer()` class for the description of the parameters. 

## Predict images using a pre-trained model with provided example images

We provide the pre-trained network used to analyse the data presented in the paper. The user should use the provided script `batch.py` and run it using the `--show` argument to see the predictions.
```
python batch.py --show
```

Using defaults will load the network on CPU and display the predictions after inference in a `matplotlib` figure. The images to predict are in the `images_folder` of the repository. The parameters can be accessed with the provided parser.
```
usage: batch.py [-h] [--network_path NETWORK_PATH]
                [--images_folder IMAGES_FOLDER] [--cuda] [--size SIZE]
                [--step STEP] [--show]

Setting the parameters of the prediction.

optional arguments:
  -h, --help            show this help message and exit
  --network_path NETWORK_PATH
                        The path of the trained network. Defaults to the pre-
                        trained model.
  --images_folder IMAGES_FOLDER
                        Folder containing the images to predict.
  --cuda                Wheter or not to use CUDA.
  --size SIZE           The size of the crops used for training the network.
  --step STEP           The step between each crops.
  --show                Wheter or not to show each prediction.
```

## Data structure

The images used for training contain 3 different channels with type `uint16` and `.tif` format. The 3 channels contain information about different labeled structures of the neurons.
1. actin (phalloidin-STAR635)
2. axonal marker (SMI31-STAR580)
3. dendritic marker (MAP2-STAR488)

The labels used for training are in a `.txt` file. Some examples are provided in the [data folder](dataset/data). Each line of the file contains information about a polygonal region that was labeled by the expert. 
```
id_label (x0, y0), (x1, y1), (x2, y2), (x3, y3)
id_label (x0, y0), (x1, y1), (x2, y2), (x3, y3)
...
```
The `id_label` is the id associated with the label. In our case, 0 are clear rings, 1 are unclear rings, 2 is for the dendrite label and 3 are fibers. 

To train a network from scratch with his/her own images, the user should provide the path to the `training.pkl` and `validation.pkl` files. Those files contain the filepaths of images and labels in a python dictionnary. The structure of both files is the same. 
```json
{
  "images" : ["path/to/image1.tif", "path/to/image2.tif"], 
  "labels" : ["path/to/label1.txt", "path/to/label2.txt"]
}
```

To predict images of his/her own using the pre-trained model, the user requires single channel `.tif` images of type `uint16`. The user should be careful of the `image_min` and `image_max` used to normalize the images since FCN can be sensitive to the distribution of value of pixels. We recommend the user to extract the statistics from his/her images. 
