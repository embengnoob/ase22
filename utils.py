import sys
import os
import shutil
import time
import datetime
from datetime import timedelta
from pathlib import Path
import csv
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split

# Make sure that we are using QT5
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5 import QtWidgets, QtCore
import matplotlib.image as mpimg
import cv2

from colorama import Fore, Back
from colorama import init
init(autoreset=True)

import colored_traceback
colored_traceback.add_hook(always=True)

from tqdm import tqdm

from config import Config

######################################################################################
############################## HEATMAP UTIL IMPORTS ##################################
######################################################################################
import math
np.bool = np.bool_
import collections
from PIL import Image

RESIZED_IMAGE_HEIGHT, RESIZED_IMAGE_WIDTH = 80, 160
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 160, 320, 3
INPUT_SHAPE = (RESIZED_IMAGE_HEIGHT, RESIZED_IMAGE_WIDTH, IMAGE_CHANNELS)

csv_fieldnames_original_simulator = ["center", "left", "right", "steering", "throttle", "brake", "speed"]
csv_fieldnames_improved_simulator = ["frameId", "model", "anomaly_detector", "threshold", "sim_name",
                                     "lap", "waypoint", "loss",
                                     "uncertainty",  # newly added
                                     "cte", "steering_angle", "throttle", "speed", "brake", "crashed",
                                     "distance", "time", "ang_diff",  # newly added
                                     "center", "tot_OBEs", "tot_crashes", "car_position"]
csv_fieldnames_in_manual_mode = ["center", "left", "right", "steeringAngle", "throttle", "brake", "speed", "lap", "sector", "cte", "car_position"]

class ScrollableWindow(QtWidgets.QMainWindow):
    def __init__(self, fig):
        # self.qapp = QtWidgets.QApplication([])
        plt.close("all")
        if not QtWidgets.QApplication.instance():
            self.app = QtWidgets.QApplication(sys.argv)
        else:
            self.app = QtWidgets.QApplication.instance()

        QtWidgets.QMainWindow.__init__(self)
        self.widget = QtWidgets.QWidget()
        self.setCentralWidget(self.widget)
        self.widget.setLayout(QtWidgets.QVBoxLayout())
        self.widget.layout().setContentsMargins(0,0,0,0)
        self.widget.layout().setSpacing(0)

        self.fig = fig
        self.canvas = FigureCanvas(self.fig)
        self.canvas.draw()
        self.scroll = QtWidgets.QScrollArea(self.widget)
        self.scroll.setWidget(self.canvas)

        self.nav = NavigationToolbar(self.canvas, self.widget)
        self.widget.layout().addWidget(self.nav)
        self.widget.layout().addWidget(self.scroll)

        self.showMaximized()
        exit(self.app.exec_())

class ScrollableGraph(QtWidgets.QMainWindow):
    def __init__(self, fig, ax, step=0.1):
        plt.close("all")
        if not QtWidgets.QApplication.instance():
            self.app = QtWidgets.QApplication(sys.argv)
        else:
            self.app = QtWidgets.QApplication.instance() 

        QtWidgets.QMainWindow.__init__(self)
        self.widget = QtWidgets.QWidget()
        self.setCentralWidget(self.widget)
        self.widget.setLayout(QtWidgets.QVBoxLayout())
        self.widget.layout().setContentsMargins(0,0,0,0)
        self.widget.layout().setSpacing(0)

        self.fig = fig
        self.ax = ax
        self.canvas = FigureCanvas(self.fig)
        self.canvas.draw()
        self.scroll = QtWidgets.QScrollBar(QtCore.Qt.Horizontal)
        self.step = step
        self.setupSlider()
        self.nav = NavigationToolbar(self.canvas, self.widget)
        self.widget.layout().addWidget(self.nav)
        self.widget.layout().addWidget(self.canvas)
        self.widget.layout().addWidget(self.scroll)

        self.canvas.draw()
        self.show()
        self.app.exec_()

    def setupSlider(self):
        self.lims = np.array(self.ax.get_xlim())
        self.scroll.setPageStep(self.step*100)
        self.scroll.actionTriggered.connect(self.update)
        self.update()

    def update(self, evt=None):
        r = self.scroll.value()/((1+self.step)*100)
        l1 = self.lims[0]+r*np.diff(self.lims)
        l2 = l1 +  np.diff(self.lims)*self.step
        self.ax.set_xlim(l1,l2)
        print(self.scroll.value(), l1,l2)
        self.fig.canvas.draw_idle()

def load_image(data_dir, image_file):
    """
    Load RGB images from a file
    """
    image_dir = data_dir
    local_path = "/".join(image_file.split("/")[-4:-1]) + "/" + image_file.split("/")[-1]
    img_path = "{0}/{1}".format(image_dir, local_path)
    try:
        return mpimg.imread(img_path)
    except FileNotFoundError:
        print(image_file + " not found")
        # exit(1)


def crop(image):
    """
    Crop the image (removing the sky at the top and the car front at the bottom)
    """
    return image[60:-25, :, :]  # remove the sky and the car front


def resize(image):
    """
    Resize the image to the input_image shape used by the network model (1/4 of the simulator image size)
    """
    return cv2.resize(image, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT), cv2.INTER_AREA)


def resize_original_size(image):
    """
    Resize the image to the input_image shape used by the network model (the original size of the simulator)
    """
    return cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)


def rgb2yuv(image):
    """
    Convert the image from RGB to YUV (This is what the NVIDIA model does)
    """
    return cv2.cvtColor(image.astype('uint8') * 255, cv2.COLOR_RGB2YUV)


def preprocess(image, old_model):
    """
    Combine all preprocess functions into one
    """
    image = crop(image)
    if old_model:
        image = resize_original_size(image)
    else:
        image = resize(image)
    image = rgb2yuv(image)
    return image


def choose_image(data_dir, center, left, right, steering_angle):
    """
    Randomly choose an image from the center, left or right, and adjust
    the steering angle.
    """
    choice = np.random.choice(3)
    if choice == 0:
        return load_image(data_dir, left), steering_angle + 0.2
    elif choice == 1:
        return load_image(data_dir, right), steering_angle - 0.2
    return load_image(data_dir, center), steering_angle


def random_flip(image, steering_angle):
    """
    Randomly flip the image left <-> right, and adjust the steering angle.
    """
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle
    return image, steering_angle


def random_translate(image, steering_angle, range_x, range_y):
    """
    Randomly shift the image vertically and horizontally (translation).
    """
    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    steering_angle += trans_x * 0.002
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))
    return image, steering_angle


def random_shadow(image):
    """
    Generates and adds random shadow
    """
    # (x1, y1) and (x2, y2) forms a line
    # xm, ym gives all the locations of the image
    x1, y1 = IMAGE_WIDTH * np.random.rand(), 0
    x2, y2 = IMAGE_WIDTH * np.random.rand(), IMAGE_HEIGHT
    xm, ym = np.mgrid[0:IMAGE_HEIGHT, 0:IMAGE_WIDTH]

    # mathematically speaking, we want to set 1 below the line and zero otherwise
    # Our coordinate is up side down.  So, the above the line: 
    # (ym-y1)/(xm-x1) > (y2-y1)/(x2-x1)
    # as x2 == x1 causes zero-division problem, we'll write it in the below form:
    # (ym-y1)*(x2-x1) - (y2-y1)*(xm-x1) > 0
    mask = np.zeros_like(image[:, :, 1])
    mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = 1

    # choose which side should have shadow and adjust saturation
    cond = mask == np.random.randint(2)
    s_ratio = np.random.uniform(low=0.2, high=0.5)

    # adjust Saturation in HLS(Hue, Light, Saturation)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
    return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)


def random_brightness(image):
    """
    Randomly adjust brightness of the image.
    """
    # HSV (Hue, Saturation, Value) is also called HSB ('B' for Brightness).
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    hsv[:, :, 2] = hsv[:, :, 2] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def augment(data_dir, center, left, right, steering_angle, range_x=100, range_y=10):
    """
    Generate an augmented image and adjust steering angle.
    (The steering angle is associated with the center image)
    """
    image, steering_angle = choose_image(data_dir, center, left, right, steering_angle)
    # TODO: flip should be applied to left/right only and w/ no probability
    image, steering_angle = random_flip(image, steering_angle)
    image, steering_angle = random_translate(image, steering_angle, range_x, range_y)
    image = random_shadow(image)
    image = random_brightness(image)
    return image, steering_angle


def rmse(y_true, y_pred):
    """
    Calculates RMSE
    """
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def write_csv_line(filename, row):
    if filename is not None:
        filename += "/driving_log.csv"
        with open(filename, mode='a') as result_file:
            writer = csv.writer(result_file,
                                delimiter=',',
                                quotechar='"',
                                quoting=csv.QUOTE_MINIMAL,
                                lineterminator='\n')
            writer.writerow(row)
            result_file.flush()
            result_file.close()
    else:
        create_csv_results_file_header(filename)


def create_csv_results_file_header(file_name, fieldnames):
    """
    Creates the folder to store the driving simulation data from the Udacity simulator
    """
    if file_name is not None:
        file_name += "/driving_log.csv"
        with open(file_name, mode='w', newline='') as result_file:
            csv.writer(result_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
            writer = csv.DictWriter(result_file, fieldnames=fieldnames)
            writer.writeheader()
            result_file.flush()
            result_file.close()

    return None


def create_output_dir(cfg, fieldnames):
    """
    Creates the folder to store the driving simulation data from the Udacity simulator
    """
    path = os.path.join(cfg.TESTING_DATA_DIR, cfg.SIMULATION_NAME, "IMG")
    csv_path = os.path.join(cfg.TESTING_DATA_DIR, cfg.SIMULATION_NAME)

    if os.path.exists(path):
        print("Deleting folder at {}".format(csv_path))
        shutil.rmtree(csv_path)

    print("Creating image folder at {}".format(path))
    os.makedirs(path)
    create_csv_results_file_header(csv_path, fieldnames)


def load_driving_data_log(cfg: object) -> object:
    """
    Retrieves the driving data log from cfg.SIMULATION_NAME
    """
    path = None
    data_df = None
    try:
        data_df = pd.read_csv(os.path.join(cfg.TESTING_DATA_DIR, cfg.SIMULATION_NAME, 'driving_log.csv'),
                              keep_default_na=False)
    except FileNotFoundError:
        print("Unable to read file %s" % path)
        exit()

    return data_df


def get_driving_styles(cfg):
    """
    Retrieves the driving styles to compose the training set
    """
    if cfg.TRACK == "track1":
        return cfg.TRACK1_DRIVING_STYLES
    elif cfg.TRACK == "track2":
        return cfg.TRACK2_DRIVING_STYLES
    elif cfg.TRACK == "track3":
        return cfg.TRACK3_DRIVING_STYLES
    else:
        print("Invalid TRACK option within the config file")
        exit(1)


def load_improvement_set(cfg, ids):
    """
    Load the paths to the images in the cfg.SIMULATION_NAME directory.
    Filters those having a frame id in the set ids.
    """
    start = time.time()

    x = None
    path = None

    try:
        path = os.path.join(cfg.TESTING_DATA_DIR,
                            cfg.SIMULATION_NAME,
                            'driving_log.csv')
        data_df = pd.read_csv(path)

        print("Filtering only false positives")
        data_df = data_df[data_df['frameId'].isin(ids)]

        if x is None:
            x = data_df[['center']].values
        else:
            x = np.concatenate((x, data_df[['center']].values), axis=0)

    except FileNotFoundError:
        print("Unable to read file %s" % path)

    if x is None:
        print("No driving data_nominal were provided for training. Provide correct paths to the driving_log.csv files")
        exit()

    duration_train = time.time() - start
    print("Loading improvement data_nominal set completed in %s." % str(
        datetime.timedelta(seconds=round(duration_train))))

    print("False positive data_nominal set: " + str(len(x)) + " elements")

    return x


# copy of load_all_images for loading the heatmaps
def load_all_heatmaps(cfg):
    """
    Load the actual heatmaps (not the paths!)
    """
    path = os.path.join(cfg.TESTING_DATA_DIR,
                        cfg.SIMULATION_NAME,
                        'heatmaps-smoothgrad',
                        'driving_log.csv')
    data_df = pd.read_csv(path)

    x = data_df["center"]
    print("read %d images from directory %s" % (len(x), path))

    start = time.time()

    # load the images
    images = np.empty([len(x), RESIZED_IMAGE_HEIGHT, RESIZED_IMAGE_WIDTH, IMAGE_CHANNELS])

    for i, path in enumerate(x):
        try:
            image = mpimg.imread(path)  # load center images
        except FileNotFoundError:
            path = path.replace('\\', '/')
            image = mpimg.imread(path)

        # visualize the input_image image
        # import matplotlib.pyplot as plt
        # plt.imshow(image)
        # plt.show()

        images[i] = image

    duration_train = time.time() - start
    print("Loading data_nominal set completed in %s." % str(datetime.timedelta(seconds=round(duration_train))))

    print("Data set: " + str(len(images)) + " elements")

    return images


def load_all_images(cfg):
    """
    Load the actual images (not the paths!) in the cfg.SIMULATION_NAME directory.
    """
    path = os.path.join(cfg.TESTING_DATA_DIR,
                        cfg.SIMULATION_NAME,
                        'driving_log.csv')
    data_df = pd.read_csv(path)

    x = data_df["center"]
    print("read %d images from directory %s" % (len(x), path))

    start = time.time()

    # load the images
    images = np.empty([len(x), IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])

    for i, path in enumerate(x):
        path = path.replace("\\", "/")

        image = mpimg.imread(path)  # load center images

        # visualize the input_image image
        # import matplotlib.pyplot as plt
        # plt.imshow(image)
        # plt.show()

        images[i] = image

    duration_train = time.time() - start
    print("Loading data_nominal set completed in %s." % str(datetime.timedelta(seconds=round(duration_train))))

    print("Data set: " + str(len(images)) + " elements")

    return images


def plot_reconstruction_losses(losses, new_losses, name, threshold, new_threshold, data_df):
    """
    Plots the reconstruction errors for one or two sets of losses, along with given thresholds.
    Crashes are visualized in red.
    """
    plt.figure(figsize=(20, 4))
    x_losses = np.arange(len(losses))

    x_threshold = np.arange(len(x_losses))
    y_threshold = [threshold] * len(x_threshold)
    plt.plot(x_threshold, y_threshold, '--', color='black', alpha=0.4, label='threshold')

    # visualize crashes
    try:
        crashes = data_df[data_df["crashed"] == 1]
        is_crash = (crashes.crashed - 1) + threshold
        plt.plot(is_crash, 'x:r', markersize=4)
    except KeyError:
        print("crashed column not present in the csv")

    if new_threshold is not None:
        plt.plot(x_threshold, [new_threshold] * len(x_threshold), color='red', alpha=0.4, label='new threshold')

    plt.plot(x_losses, losses, '-.', color='blue', alpha=0.7, label='original')
    if new_losses is not None:
        plt.plot(x_losses, new_losses, color='green', alpha=0.7, label='retrained')

    plt.legend()
    plt.ylabel('Loss')
    plt.xlabel('Number of Instances')
    plt.title("Reconstruction error for " + name)

    plt.savefig('plots/reconstruction-plot-' + name + '.png')

    plt.show()


def laplacian_variance(images):
    """
    Computes the Laplacian variance for the given list of images
    """
    return [cv2.Laplacian(image, cv2.CV_32F).var() for image in images]


def load_autoencoder_from_disk():
    cfg = Config()
    cfg.from_pyfile("config_my.py")

    encoder = tensorflow.keras.models.load_model(
        cfg.SAO_MODELS_DIR + os.path.sep + 'encoder-' + cfg.ANOMALY_DETECTOR_NAME)
    decoder = tensorflow.keras.models.load_model(
        cfg.SAO_MODELS_DIR + os.path.sep + 'decoder-' + cfg.ANOMALY_DETECTOR_NAME)

    # TODO: manage the case in which the files do not exist
    return encoder, decoder

# Colors: BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE, RESET.
#         LIGHTBLACK_EX, LIGHTRED_EX, LIGHTGREEN_EX, LIGHTYELLOW_EX, LIGHTBLUE_EX,
#         LIGHTMAGENTA_EX, LIGHTCYAN_EX, LIGHTWHITE_EX
def cprintf(fstring, color):
    """
    Colorful foreground print in terminal.
    """
    if color == 'black':
        print(Fore.BLACK + fstring)
    elif color == 'red':
        print(Fore.RED + fstring)
    elif color == 'green':
        print(Fore.GREEN + fstring)
    elif color == 'yellow':
        print(Fore.YELLOW + fstring)
    elif color == 'blue':
        print(Fore.BLUE + fstring)
    elif color == 'magenta':
        print(Fore.MAGENTA + fstring)
    elif color == 'cyan':
        print(Fore.CYAN + fstring)
    elif color == 'white':
        print(Fore.WHITE + fstring)
    elif color == 'l_black':
        print(Fore.LIGHTBLACK_EX + fstring)
    elif color == 'l_red':
        print(Fore.LIGHTRED_EX + fstring)
    elif color == 'l_green':
        print(Fore.LIGHTGREEN_EX + fstring)
    elif color == 'l_yellow':
        print(Fore.LIGHTYELLOW_EX + fstring)
    elif color == 'l_blue':
        print(Fore.LIGHTBLUE_EX + fstring)
    elif color == 'l_magenta':
        print(Fore.LIGHTMAGENTA_EX + fstring)
    elif color == 'l_cyan':
        print(Fore.LIGHTCYAN_EX + fstring)
    elif color == 'l_white':
        print(Fore.LIGHTWHITE_EX + fstring)

def cprintb(fstring, color):
    """
    Colorful background print in terminal.
    """
    if color == 'black':
        print(Back.BLACK + fstring)
    elif color == 'red':
        print(Back.RED + fstring)
    elif color == 'green':
        print(Back.GREEN + fstring)
    elif color == 'yellow':
        print(Back.YELLOW + fstring)
    elif color == 'blue':
        print(Back.BLUE + fstring)
    elif color == 'magenta':
        print(Back.MAGENTA + fstring)
    elif color == 'cyan':
        print(Back.CYAN + fstring)
    elif color == 'white':
        print(Back.WHITE + fstring)
    elif color == 'l_black':
        print(Back.LIGHTBLACK_EX + fstring)
    elif color == 'l_red':
        print(Back.LIGHTRED_EX + fstring)
    elif color == 'l_green':
        print(Back.LIGHTGREEN_EX + fstring)
    elif color == 'l_yellow':
        print(Back.LIGHTYELLOW_EX + fstring)
    elif color == 'l_blue':
        print(Back.LIGHTBLUE_EX + fstring)
    elif color == 'l_magenta':
        print(Back.LIGHTMAGENTA_EX + fstring)
    elif color == 'l_cyan':
        print(Back.LIGHTCYAN_EX + fstring)
    elif color == 'l_white':
        print(Back.LIGHTWHITE_EX + fstring)

def load_data(cfg):
    """
    Load training data_nominal and split it into training and validation set
    """
    start = time.time()

    x = None
    y = None
    path = None
    x_train = None
    y_train = None
    x_test = None
    y_test = None

    drive = get_driving_styles(cfg)
    print("Loading training set " + cfg.TRACK + str(drive))
    for drive_style in drive:
        try:
            path = os.path.join(cfg.TRAINING_DATA_DIR,
                                cfg.TRAINING_SET_DIR,
                                cfg.TRACK,
                                drive_style,
                                'driving_log.csv')
            data_df = pd.read_csv(path)
            if x is None:
                x = data_df[['center', 'left', 'right']].values
                y = data_df['steering'].values
            else:
                x = np.concatenate((x, data_df[['center', 'left', 'right']].values), axis=0)
                y = np.concatenate((y, data_df['steering'].values), axis=0)
        except FileNotFoundError:
            print("Unable to read file %s" % path)
            continue

    if x is None:
        print("No driving data_nominal were provided for training. Provide correct paths to the driving_log.csv files")
        exit()

    try:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=cfg.TEST_SIZE, random_state=0)
    except TypeError:
        print("Missing header to csv files")
        exit()

    duration_train = time.time() - start
    print("Loading training set completed in %s." % str(datetime.timedelta(seconds=round(duration_train))))

    print("Data set: " + str(len(x)) + " elements")
    print("Training set: " + str(len(x_train)) + " elements")
    print("Test set: " + str(len(x_test)) + " elements")
    return x_train, x_test, y_train, y_test



######################################################################################
################################# HEATMAP UTILITIES ##################################
######################################################################################

def batch_run(function, images, batch_size=5000):
    '''
    function   : lambda function taking images with shape [N,H,W,C] as input
    images     : tensor of shape [N,H,W,C]
    batch_size : batch size
    '''
    
    res = []
    
    for i in range(math.ceil(len(images) / batch_size)):
        
        res.append(function(images[i*batch_size:(i+1)*batch_size]))
    
    return np.concatenate(res, axis=0)


def preprocess(attributions, q1, q2, use_abs=False):
    
    if use_abs:
        attributions = np.abs(attributions)
    
    if tf.is_tensor(attributions):
        attributions = attributions.numpy()
        # attributions.eval(session=tf.compat.v1.Session())
    attributions = np.sum(attributions, axis=-1)
    if attributions.ndim == 2:
        attributions = attributions[np.newaxis is None,:,:]
    # cprintf(f'{attributions.shape}', 'l_blue')
    a_min = np.percentile(attributions, q1, axis=(1,2), keepdims=True)
    a_max = np.percentile(attributions, q2, axis=(1,2), keepdims=True)
    
    pos = np.tile(a_min > 0, [1,attributions.shape[1],attributions.shape[2]])
    ind = np.where(attributions < a_min)
    
    attributions = np.clip(attributions, a_min, a_max)
    attributions[ind] = (1 - pos[ind]) * attributions[ind]
    
    return attributions


def pixel_range(img):
    vmin, vmax = np.min(img), np.max(img)

    if vmin * vmax >= 0:
        
        v = np.maximum(np.abs(vmin), np.abs(vmax))
        
        return [-v, v], 'bwr'
    
    else:

        if -vmin > vmax:
            vmax = -vmin
        else:
            vmin = -vmax

        return [vmin, vmax], 'bwr'


def scale(x):
    
    return x / 127.5 - 1.0

def tensor(arg):
  arg = tf.convert_to_tensor(arg, dtype=tf.float32)
  return arg

def file_name(x):
    return x.split('.')[0]