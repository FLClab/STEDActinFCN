
import numpy

from skimage import draw


def load_poly(path):
    """
    Loads the poly.txt file

    :param path: The path of the file to open

    :returns : A list of every line in the file
    """
    with open(path, "r") as f:
        data = f.readlines()
    return [x.strip() for x in data]


def read_poly(path, shape):
    """
    Reads and creates the labels from the poly.txt files

    :param path: The path to the poly.txt file
    :param shape: The shape of the image

    :returns : A numpy array of the labels

    NOTE. 0 is uncertain rings and 1 is clear rings
    """
    label = numpy.zeros(shape)
    data = load_poly(path)
    for row in data:
        l = int(row[0:1])
        if l in [0, 1]:
            coordinates = eval(row[2:])
            r, c = [], []
            for coord in coordinates:
                r.append(int(coord[1]))
                c.append(int(coord[0]))
            rr, cc = draw.polygon(r, c, shape=shape)
            label[rr, cc] = 1
    return label
