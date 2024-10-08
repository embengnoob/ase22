import sys
import os
import shutil
import glob
import time
from pathlib import Path
import ntpath
from datetime import datetime, timedelta

import csv
import numpy as np
import pandas as pd
from collections import defaultdict
from itertools import product

import tensorflow as tf
# from tensorflow.keras import backend as K
from keras import backend as K
from sklearn.model_selection import train_test_split

# Make sure that we are using QT5
import matplotlib
# matplotlib.use('Qt5Agg')
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colormaps
import matplotlib.transforms as transforms
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5 import QtWidgets, QtCore
import matplotlib.image as mpimg
import cv2
from skimage.color import rgb2gray

from colorama import Fore, Back
from colorama import init
init(autoreset=True)

import colored_traceback
colored_traceback.add_hook(always=True)

import gc
from tqdm import tqdm
import warnings
import re
from config import Config


######################################################################################
############################## EVAL UTIL IMPORTS ##################################
######################################################################################
from scipy.stats import gamma, wasserstein_distance, pearsonr, spearmanr, kendalltau, entropy
from scipy.special import kl_div, rel_entr
from sklearn import preprocessing
from sklearn.metrics import pairwise_distances_argmin_min, pairwise_distances, pairwise, mutual_info_score
from sklearn.decomposition import PCA
from libpysal.weights import lat2W
from esda.moran import Moran
from splot.esda import moran_scatterplot
######################################################################################
############################## HEATMAP UTIL IMPORTS ##################################
######################################################################################
import math
np.bool = np.bool_
import collections
from PIL import Image, ImageFont, ImageDraw


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

    encoder = tf.keras.models.load_model(
        cfg.SAO_MODELS_DIR + os.path.sep + 'encoder-' + cfg.ANOMALY_DETECTOR_NAME)
    decoder = tf.keras.models.load_model(
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

def highlight(var, color='white'):
    try:
        for name in globals():
            if eval(name) == var:
                cprintb(f'{name}:', color)
    except:
        pass
    cprintb(f'{var}', color)

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


def preprocess_atm(attributions, q1, q2, use_abs=False):
    
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


######################################################################################
################################ EVAL FUNCTIONS ######################################
######################################################################################

def string_to_np_array(vector_string, frame_num):
    if '[' in vector_string:
        # autonomous mode
        vector_string = ' '.join(vector_string.split())
        vector_string = vector_string.strip("[]").strip().replace(' ', ',')
        vector = np.fromstring(vector_string, dtype=float, sep=',')
    elif '(' in vector_string:
        # manual training mode
        vector_string = vector_string.strip("()").replace('  ', ' ')
        vector = np.fromstring(vector_string, dtype=float, sep=' ')
    if vector.shape != (3,):
        cprintf(str(vector.shape), 'l_red')
        print(vector_string)
        print(vector)
        raise ValueError(f"Car position format of frame number {frame_num} can't be interpreted.")
    return vector

def correct_windows_path(address):
    if "\\" in address:
        address = address.replace("\\", "/")
    elif "\\\\" in address:
        address = address.replace("\\\\", "/")
    return address

def extract_time_from_str(first_img_path, last_img_path):
    first_img_name = ntpath.basename(first_img_path)
    last_img_name = ntpath.basename(last_img_path)
    start_time = []
    end_time = []
    if 'FID' in first_img_name:
        start_idx = 6
        end_idx = 2
    else:
        start_idx = 4
        end_idx = 0
    for i in range(start_idx, end_idx, -1):
        start_time.append(first_img_name.split('_')[-i].split('.')[0])
        end_time.append(last_img_name.split('_')[-i].split('.')[0])
    return start_time, end_time

def get_threshold_thirdeye(losses, conf_level=0.95):
    print("Fitting reconstruction error distribution using Gamma distribution")

    # removing zeros
    losses = np.array(losses)
    losses_copy = losses[losses != 0]
    shape, loc, scale = gamma.fit(losses_copy, floc=0)

    print("Creating threshold using the confidence intervals: %s" % conf_level)
    t = gamma.ppf(conf_level, shape, loc=loc, scale=scale)
    print('threshold: ' + str(t))
    return t

def get_nom_window_scores(heatmap_type, summary_types, aggregation_methods, nom_csv_npy_paths):

    # extract necessary paths
    NOMINAL_HEATMAP_CSV_PATH, NOM_NPY_SCORES_FOLDER_PATH = nom_csv_npy_paths

    # Create a defaultdict of dictionaries
    window_scores = defaultdict(dict)
    thresholds = defaultdict(dict)
    loss_scores = defaultdict(dict)

    for summary_type in summary_types:
        for aggregation_method in aggregation_methods: 
            cprintf(f'\n########### NOM sim using agg_method:{aggregation_method}, summary type:{summary_type} ###########', 'yellow')

            nom_npy_losses_path = os.path.join(NOM_NPY_SCORES_FOLDER_PATH, 'htm-' + heatmap_type + '-scores' + summary_type + '.npy')
            nominal_losses = np.load(nom_npy_losses_path)
            
            # load heatmap scores in nominal conditions
            data_df_nominal_orig = pd.read_csv(NOMINAL_HEATMAP_CSV_PATH)
            data_df_nominal = data_df_nominal_orig.copy()
            data_df_nominal['loss'] = nominal_losses
            number_frames_nominal = pd.Series.max(data_df_nominal['frameId'])
            simulation_time_nominal = pd.Series.max(data_df_nominal['time'])
            fps_nominal = number_frames_nominal // simulation_time_nominal
            num_windows_nominal = int(len(data_df_nominal) // fps_nominal)


            if len(data_df_nominal) % fps_nominal != 0:
                num_to_delete = len(data_df_nominal) - (num_windows_nominal * fps_nominal) - 1
                data_df_nominal = data_df_nominal[:-num_to_delete]
            
            loss_scores[summary_type] = data_df_nominal['loss']
            losses = pd.Series(data_df_nominal['loss'])
            sma_nominal = losses.rolling(fps_nominal, min_periods=1).mean()

            list_aggregated = []

            for idx, loss in enumerate(sma_nominal):
                if idx > 0 and idx % fps_nominal == 0:
                    aggregated_score = None
                    if aggregation_method == "mean":
                        aggregated_score = pd.Series(sma_nominal.iloc[idx - fps_nominal:idx]).mean()
                    elif aggregation_method == "max":
                        aggregated_score = pd.Series(sma_nominal.iloc[idx - fps_nominal:idx]).max()

                    list_aggregated.append(aggregated_score)

                elif idx == len(sma_nominal) - 1:
                    aggregated_score = None
                    if aggregation_method == "mean":
                        aggregated_score = pd.Series(sma_nominal.iloc[idx - fps_nominal:idx]).mean()
                    elif aggregation_method == "max":
                        aggregated_score = pd.Series(sma_nominal.iloc[idx - fps_nominal:idx]).max()

                    list_aggregated.append(aggregated_score)

            assert len(list_aggregated) == num_windows_nominal
            threshold = get_threshold_thirdeye(list_aggregated, conf_level=0.95)
            window_scores[summary_type][aggregation_method] = list_aggregated
            thresholds[summary_type][aggregation_method] = threshold

    return window_scores, thresholds, loss_scores, data_df_nominal



def get_ano_window_scores(heatmap_type, summary_types, aggregation_methods, ano_csv_npy_paths):

    # extract necessary paths
    ANOMALOUS_HEATMAP_CSV_PATH, ANO_NPY_SCORES_FOLDER_PATH, ANOMALOUS_MAIN_CSV_PATH = ano_csv_npy_paths

    # Create a defaultdict of dictionaries
    window_scores = defaultdict(dict)
    window_indices = defaultdict(dict)
    loss_scores = defaultdict(dict)

    for summary_type in summary_types:
        for aggregation_method in aggregation_methods: 
            # cprintf(f'\n########### ANO sim using agg_method:{aggregation_method}, summary type:{summary_type} ###########', 'yellow')

            # # # creates the ground truth
            # # all_first_frame_position_OOT_sequences, OOT_anomalous_in_anomalous_conditions = get_OOT_frames(data_df_anomalous, number_frames_anomalous)
            # # print("identified %d OOT(s)" % len(all_first_frame_position_OOT_sequences))
            # # print(all_first_frame_position_OOT_sequences)



            # frames_to_reassign = fps_anomalous * seconds_to_anticipate  # start of the sequence / length of time window (seconds to anticipate) in terms of number of frames

            # # first frame n seconds before the failure / length of time window one second shorter
            # # than seconds_to_anticipate in terms of number of frame
            # frames_to_reassign_2 = fps_anomalous * (seconds_to_anticipate - 1)
            # # frames_to_reassign_2 = 1  # first frame before the failure

            # reaction_window = pd.Series()
            # (reaction_window_x_min, reaction_window_x_max) = (0,0)
            ####################################################################################################################################################################################################################
            #load heatmap scores in anomalous conditions
            data_df_anomalous_orig = pd.read_csv(ANOMALOUS_HEATMAP_CSV_PATH)
            data_df_anomalous = data_df_anomalous_orig.copy()
            number_frames_anomalous = len(data_df_anomalous['frameId'])
            # calculate simulation time window and cut extra samples
            number_frames_anomalous = pd.Series.max(data_df_anomalous['frameId'])
            # to cover both manual and auto versions of csv
            try:
                simulation_time_anomalous = pd.Series.max(data_df_anomalous['time'])
                fps_anomalous = number_frames_anomalous // simulation_time_anomalous
            except:
                # get total simulation time
                main_csv_anomalous = pd.read_csv(ANOMALOUS_MAIN_CSV_PATH)
                main_csv = pd.DataFrame(main_csv_anomalous['center'].copy(), columns=['center'])

                first_img_path = main_csv['center'].iloc[0]
                last_img_path = main_csv['center'].iloc[-1]

                start_time, end_time = extract_time_from_str(first_img_path, last_img_path)
                start_time = f"{start_time[0]}:{start_time[1]}:{start_time[2]}.{start_time[3]}"
                end_time = f"{end_time[0]}:{end_time[1]}:{end_time[2]}.{end_time[3]}"


                # convert time string to datetime
                t1 = datetime.strptime(start_time, "%H:%M:%S.%f")
                t2 = datetime.strptime(end_time, "%H:%M:%S.%f")
                # get difference
                simulation_time_anomalous = t2 - t1
                number_frames_anomalous = len(data_df_anomalous['center'])
                # ano FPS
                fps_anomalous = int(number_frames_anomalous // simulation_time_anomalous.total_seconds())

            num_windows_anomalous = len(data_df_anomalous) // fps_anomalous
            ano_npy_losses_path = os.path.join(ANO_NPY_SCORES_FOLDER_PATH, 'htm-' + heatmap_type + '-scores' + summary_type + '.npy')
            anomalous_losses = np.load(ano_npy_losses_path)
            data_df_anomalous['loss'] = anomalous_losses

            list_aggregated = []
            list_aggregated_indices = []

            num_to_delete = 0

            if len(data_df_anomalous) % fps_anomalous != 0:
                num_to_delete = int(len(data_df_anomalous) - (num_windows_anomalous * fps_anomalous) - 1)

            if num_to_delete != 0:
                data_df_anomalous = data_df_anomalous[:-num_to_delete]
            
            loss_scores[summary_type] = data_df_anomalous['loss']
            sma_anomalous = pd.Series(data_df_anomalous['loss'])
            
            # calculate mean or max of windows the length of fps_anomalous starting from idx-fps_anomalous to idx
            for idx, loss in enumerate(sma_anomalous):
                if idx > 0 and idx % fps_anomalous == 0:
                    aggregated_score = None
                    if aggregation_method == "mean":
                        aggregated_score = pd.Series(sma_anomalous.iloc[idx - fps_anomalous:idx]).mean()
                    elif aggregation_method == "max":
                        aggregated_score = pd.Series(sma_anomalous.iloc[idx - fps_anomalous:idx]).max()

                    list_aggregated_indices.append(idx)
                    list_aggregated.append(aggregated_score)

                elif idx == len(sma_anomalous) - 1:
                    aggregated_score = None
                    if aggregation_method == "mean":
                        aggregated_score = pd.Series(sma_anomalous.iloc[idx - fps_anomalous:idx]).mean()
                    elif aggregation_method == "max":
                        aggregated_score = pd.Series(sma_anomalous.iloc[idx - fps_anomalous:idx]).max()

                    list_aggregated_indices.append(idx)
                    list_aggregated.append(aggregated_score)

            assert len(list_aggregated) == num_windows_anomalous
            window_scores[summary_type][aggregation_method] = list_aggregated
            window_indices[summary_type][aggregation_method] = list_aggregated_indices

    return window_scores, window_indices, loss_scores, data_df_anomalous, simulation_time_anomalous, fps_anomalous


def get_threshold(score_file_path, distance_type= None, conf_level=0.95, text_file=True, min_log=True):
    if not min_log:
        if distance_type != None:
            print(f"Fitting \"{distance_type}\" scores using Gamma distribution")
        else:
            print(f"Fitting \"thirdeye\" scores using Gamma distribution")
    if text_file:
        scores = np.loadtxt(score_file_path, dtype='float')
    else:
        scores = score_file_path
    # removing zeros
    scores = np.array(scores)
    scores_copy = scores[scores != 0]
    shape, loc, scale = gamma.fit(scores_copy, floc=0)
    if not min_log:
        print("Creating threshold using the confidence intervals: %s" % conf_level)
    t = gamma.ppf(conf_level, shape, loc=loc, scale=scale)
    if not min_log:
        print('threshold: ' + str(t))
    return t

def get_OOT_frames(data_df_anomalous, number_frames_anomalous):
    OOT_anomalous = data_df_anomalous['crashed']
    OOT_anomalous.is_copy = None
    OOT_anomalous_in_anomalous_conditions = OOT_anomalous.copy()

    # creates the ground truth
    all_first_frame_position_OOT_sequences = []
    for idx, item in enumerate(OOT_anomalous_in_anomalous_conditions):
        if idx == number_frames_anomalous:  # we have reached the end of the file
            continue

        if OOT_anomalous_in_anomalous_conditions[idx] == 0 and OOT_anomalous_in_anomalous_conditions[idx + 1] == 1: # if next frame is an OOT
            first_OOT_index = idx + 1
            all_first_frame_position_OOT_sequences.append(first_OOT_index) # makes a list of all frames where OOT first happened
            # print("first_OOT_index: %d" % first_OOT_index) 
    return all_first_frame_position_OOT_sequences, OOT_anomalous_in_anomalous_conditions

def get_ranges(boolean_cte_array):
    list_of_ranges = []
    rng_min = -1
    rng_max = -1
    counting_range = False
    for idx, condition in enumerate(boolean_cte_array):
        if condition == True:
            if not counting_range:
                rng_min = idx
                counting_range = True
            else:
                rng_max = idx
                counting_range = True
        else:
            if counting_range:
                if rng_max == -1:
                    list_of_ranges.append(rng_min)
                else:
                    list_of_ranges.append([rng_min,rng_max])
                counting_range = False
                rng_min = -1
                rng_max = -1 
    return list_of_ranges

def get_all_ranges(boolean_cte_array):
    list_of_ranges = []
    true_rng_min = -1
    true_rng_max = -1
    false_rng_min = -1
    false_rng_max = -1
    counting_range_true = False
    counting_range_false = False
    for idx, condition in enumerate(boolean_cte_array):
        if condition == True:
            if not counting_range_true:
                true_rng_min = idx
                counting_range_true = True
            else:
                true_rng_max = idx
                counting_range_true = True
            
            if counting_range_false:
                if false_rng_max == -1:
                    list_of_ranges.append(false_rng_min)
                else:
                    list_of_ranges.append([false_rng_min,false_rng_max])
                counting_range_false = False
                false_rng_min = -1
                false_rng_max = -1

            if idx == len(boolean_cte_array)-1:
                if true_rng_max == -1:
                    list_of_ranges.append(true_rng_min)
                else:
                    list_of_ranges.append([true_rng_min,true_rng_max])
        else:
            if not counting_range_false:
                false_rng_min = idx
                counting_range_false = True
            else:
                false_rng_max = idx
                counting_range_false = True

            if counting_range_true:
                if true_rng_max == -1:
                    list_of_ranges.append(true_rng_min)
                else:
                    list_of_ranges.append([true_rng_min,true_rng_max])
                counting_range_true = False
                true_rng_min = -1
                true_rng_max = -1

            if idx == len(boolean_cte_array)-1:
                if false_rng_max == -1:
                    list_of_ranges.append(false_rng_min)
                else:
                    list_of_ranges.append([false_rng_min,false_rng_max])
    return list_of_ranges

def merge_ranges(all_ranges, boolean_array):
    if len(all_ranges) != len(boolean_array):
        raise ValueError(Fore.RED + f"Mismatch range array and boolean array length {idx}: {len(all_ranges)} != {len(boolean_array)} " + Fore.RESET) 
    dict_of_ranges = {True:[],False:[]}
    true_rng_min = -1
    true_rng_max = -1
    false_rng_min = -1
    false_rng_max = -1
    counting_range_true = False
    counting_range_false = False
    for idx, condition in enumerate(boolean_array):
        if isinstance(all_ranges[idx], list):
            current_range_min = all_ranges[idx][0]
            current_range_max = all_ranges[idx][-1]
        else:
            current_range_min = all_ranges[idx]
            current_range_max = all_ranges[idx]

        if condition == True:
            if not counting_range_true:
                true_rng_min = current_range_min
                true_rng_max = current_range_max
                counting_range_true = True
            else:
                true_rng_max = current_range_max
                counting_range_true = True
            
            if counting_range_false:
                if (false_rng_max == -1) or (false_rng_min == false_rng_max):
                    dict_of_ranges[False].append(false_rng_min)
                else:
                    dict_of_ranges[False].append([false_rng_min,false_rng_max])
                counting_range_false = False
                false_rng_min = -1
                false_rng_max = -1

            if idx == len(boolean_array)-1:
                if (true_rng_max == -1) or (true_rng_min == true_rng_max):
                    dict_of_ranges[True].append(true_rng_min)
                else:
                    dict_of_ranges[True].append([true_rng_min,true_rng_max])
        else:
            if not counting_range_false:
                false_rng_min = current_range_min
                false_rng_max = current_range_max
                counting_range_false = True
            else:
                false_rng_max = current_range_max
                counting_range_false = True

            if counting_range_true:
                if (true_rng_max == -1) or (true_rng_min == true_rng_max):
                    dict_of_ranges[True].append(true_rng_min)
                else:
                    dict_of_ranges[True].append([true_rng_min,true_rng_max])
                counting_range_true = False
                true_rng_min = -1
                true_rng_max = -1

            if idx == len(boolean_array)-1:
                if (false_rng_max == -1) or (false_rng_min == false_rng_max):
                    dict_of_ranges[False].append(false_rng_min)
                else:
                    dict_of_ranges[False].append([false_rng_min,false_rng_max])
    return dict_of_ranges

def get_alarm_frames_thirdeye(window_scores, window_indices, threshold, num_anomalous_frames):
    # cprintf(f'{window_scores}', 'l_green')
    # print(window_indices)
    window_scores_for_all_frames = np.zeros(num_anomalous_frames)
    current_window_idx = 0
    for idx, _ in enumerate(window_scores_for_all_frames):
        if idx <= window_indices[current_window_idx]:
            window_scores_for_all_frames[idx] = window_scores[current_window_idx]
        else:
            current_window_idx += 1
            window_scores_for_all_frames[idx] = window_scores[current_window_idx]
    cprintf(f'window_scores_for_all_frames', 'white')
    highlight(window_scores_for_all_frames)
    alarm_condition = (window_scores_for_all_frames>=threshold)
    no_alarm_condition = (window_scores_for_all_frames<threshold)
    highlight(no_alarm_condition, 'green')
    highlight(len(no_alarm_condition), 'green')
    highlight(np.sum(no_alarm_condition == True), 'green')
    alarm_ranges = get_ranges(alarm_condition)
    no_alarm_ranges = get_ranges(no_alarm_condition)
    all_ranges = get_all_ranges(alarm_condition)
    return alarm_ranges, no_alarm_ranges, all_ranges

def get_alarm_frames(distance_vector_avg, threshold):
    alarm_condition = (distance_vector_avg>=threshold)
    no_alarm_condition = (distance_vector_avg<threshold)
    alarm_ranges = get_ranges(alarm_condition)
    no_alarm_ranges = get_ranges(no_alarm_condition)
    all_ranges = get_all_ranges(alarm_condition)
    return alarm_ranges, no_alarm_ranges, all_ranges

def colored_ranges(speed_anomalous, cte_anomalous, cte_diff, alpha=0.2, YELLOW_BORDER = 3.6, ORANGE_BORDER = 5.0, RED_BORDER = 7.0):
    # plot cross track error values: 
    # yellow_condition: reaching the borders of the track: yellow
    # orange_condition: on the borders of the track (partial crossing): orange
    # red_condition: out of track (full crossing): red

    if cte_diff is not None:
        yellow_condition = (
            ((abs(cte_diff)>YELLOW_BORDER)&(abs(cte_diff)<ORANGE_BORDER)) |
            ((abs(cte_anomalous) > YELLOW_BORDER) & (abs(cte_anomalous) < ORANGE_BORDER)))
        orange_condition = (
            ((abs(cte_anomalous) > ORANGE_BORDER) & (abs(cte_anomalous) < RED_BORDER)) |
            ((abs(cte_diff) > ORANGE_BORDER) & (abs(cte_diff) < RED_BORDER))
        )
        red_condition = (abs(cte_anomalous)>RED_BORDER) | (abs(cte_diff)>RED_BORDER)
    else:
        yellow_condition = ((abs(cte_anomalous) > YELLOW_BORDER) & (abs(cte_anomalous) < ORANGE_BORDER))
        orange_condition = ((abs(cte_anomalous) > ORANGE_BORDER) & (abs(cte_anomalous) < RED_BORDER))
        red_condition = (abs(cte_anomalous)>RED_BORDER)

    yellow_ranges = get_ranges(yellow_condition)
    orange_ranges = get_ranges(orange_condition)
    red_ranges = get_ranges(red_condition)

    yellow_frames = []
    for yellow_rng in yellow_ranges:
        if isinstance(yellow_rng, list):
            yellow_frame = yellow_rng[0]
        else:
            yellow_frame = yellow_rng
        yellow_frames.append(yellow_frame)

    orange_frames = []
    for orange_rng in orange_ranges:
        if isinstance(orange_rng, list):
            orange_frame = orange_rng[0]
        else:
            orange_frame = orange_rng
        orange_frames.append(orange_frame)

    red_frames = []
    for red_rng in red_ranges:
        if isinstance(red_rng, list):
            red_frame = red_rng[0]
        else:
            red_frame = red_rng
        red_frames.append(red_frame)

    # plot crash instances: speed < 1.0 
    crash_condition = (abs(speed_anomalous)<1.0)
    # remove the first 10 frames: starting out so speed is less than 1 
    crash_condition[:10] = False
    crash_ranges = get_ranges(crash_condition)
    # plot_ranges(crash_ranges, ax, color='blue', alpha=0.2)
    NUM_OF_FRAMES_TO_CHECK = 20
    is_crash_instance = False
    collision_frames = []
    for rng in crash_ranges:
        # check 20 frames before first frame with speed < 1.0. if not bigger than 15 it's not
        # a crash instance it's reset instance
        if isinstance(rng, list):
            crash_frame = rng[0]
        else:
            crash_frame = rng
        for speed in speed_anomalous[crash_frame-NUM_OF_FRAMES_TO_CHECK:crash_frame]:
            if speed > 15.0:
                is_crash_instance = True
        if is_crash_instance == True:
            is_crash_instance = False
            collision_frames.append(crash_frame)
            continue
    return red_frames, orange_frames, yellow_frames, collision_frames

def plot_ranges(ax, cte_anomalous, cte_diff, alpha=0.2, YELLOW_BORDER = 3.6,ORANGE_BORDER = 5.0, RED_BORDER = 7.0):
    # plot cross track error values: 
    # yellow_condition: reaching the borders of the track: yellow
    # orange_condition: on the borders of the track (partial crossing): orange
    # red_condition: out of track (full crossing): red

    if cte_diff is not None:
        yellow_condition = (
            ((abs(cte_diff)>YELLOW_BORDER)&(abs(cte_diff)<ORANGE_BORDER)) |
            ((abs(cte_anomalous) > YELLOW_BORDER) & (abs(cte_anomalous) < ORANGE_BORDER)))
        orange_condition = (
            ((abs(cte_anomalous) > ORANGE_BORDER) & (abs(cte_anomalous) < RED_BORDER)) |
            ((abs(cte_diff) > ORANGE_BORDER) & (abs(cte_diff) < RED_BORDER))
        )
        red_condition = (abs(cte_anomalous)>RED_BORDER) | (abs(cte_diff)>RED_BORDER)
    else:
        yellow_condition = ((abs(cte_anomalous) > YELLOW_BORDER) & (abs(cte_anomalous) < ORANGE_BORDER))
        orange_condition = ((abs(cte_anomalous) > ORANGE_BORDER) & (abs(cte_anomalous) < RED_BORDER))
        red_condition = (abs(cte_anomalous)>RED_BORDER)

    yellow_ranges = get_ranges(yellow_condition)
    orange_ranges = get_ranges(orange_condition)
    red_ranges = get_ranges(red_condition)

    all_ranges = [yellow_ranges, orange_ranges, red_ranges]
    colors = ['yellow', 'orange', 'red']
    for idx, list_of_ranges in enumerate(all_ranges):
        for rng in list_of_ranges:
            if isinstance(rng, list):
                ax.axvspan(rng[0], rng[1], color=colors[idx], alpha=alpha)
            else:
                ax.axvspan(rng, rng+1, color=colors[idx], alpha=alpha)

def plot_crash_ranges(ax, speed_anomalous, return_frames=False):
    # plot crash instances: speed < 1.0 
    crash_condition = (abs(speed_anomalous)<1.0)
    # remove the first 10 frames: starting out so speed is less than 1 
    crash_condition[:10] = False
    crash_ranges = get_ranges(crash_condition)
    # plot_ranges(crash_ranges, ax, color='blue', alpha=0.2)
    NUM_OF_FRAMES_TO_CHECK = 20
    is_crash_instance = False
    collision_frames = []
    for rng in crash_ranges:
        # check 20 frames before first frame with speed < 1.0. if not bigger than 15 it's not
        # a crash instance it's reset instance
        if isinstance(rng, list):
            crash_frame = rng[0]
        else:
            crash_frame = rng
        for speed in speed_anomalous[crash_frame-NUM_OF_FRAMES_TO_CHECK:crash_frame]:
            if speed > 15.0:
                is_crash_instance = True
        if is_crash_instance == True:
            is_crash_instance = False
            reset_frame = crash_frame
            ax.axvline(x = reset_frame, color = 'blue', linestyle = '--')
            collision_frames.append(crash_frame)
            continue
        # plot crash ranges (speed < 1.0)
        if isinstance(rng, list):
            ax.axvspan(rng[0], rng[1], color='teal', alpha=0.2)
        else:
            ax.axvspan(rng, rng+1, color='teal', alpha=0.2)
    if return_frames:
        return collision_frames
    

def get_heatmaps(anomalous_frame, anomalous, nominal, pos_mappings, return_size=False, return_IMAGE=False, return_cte=False):
    # load the addresses of centeral camera heatmap of this anomalous frame and the closest nominal frame in terms of position
    ano_hm_address = anomalous['center'].iloc[anomalous_frame]
    closest_nom_hm_address = nominal['center'].iloc[int(pos_mappings[anomalous_frame])]
    closest_nom_cte = nominal['cte'].iloc[int(pos_mappings[anomalous_frame])]
    # correct windows path, if necessary
    ano_hm_address = correct_windows_path(ano_hm_address)
    closest_nom_hm_address = correct_windows_path(closest_nom_hm_address)
    # load corresponding heatmaps
    if not return_IMAGE:
        ano_hm = mpimg.imread(ano_hm_address)
        closest_nom_hm = mpimg.imread(closest_nom_hm_address)
        if ano_hm.shape != closest_nom_hm.shape:
            raise ValueError(Fore.RED + f"Different heatmap sizes for nominal and anomalous conditions!" + Fore.RESET)
    else:
        ano_hm = Image.open(ano_hm_address)
        closest_nom_hm = Image.open(closest_nom_hm_address)
    if return_size:
        return ano_hm.shape[0], ano_hm.shape[1]
    else:
        if return_cte:
            return ano_hm, closest_nom_hm, closest_nom_cte
        else:
            return ano_hm, closest_nom_hm

def get_images(cfg, anomalous_frame, pos_mappings):
    # load the image file paths from main csv
    ano_csv_path = os.path.join(cfg.TESTING_DATA_DIR,
                                cfg.SIMULATION_NAME,
                                'driving_log.csv')
    nom_csv_path = os.path.join(cfg.TESTING_DATA_DIR,
                                cfg.SIMULATION_NAME_NOMINAL,
                                'driving_log.csv')
    ano_data = pd.read_csv(ano_csv_path)
    ano_img_address = ano_data["center"].iloc[anomalous_frame]
    nom_data = pd.read_csv(nom_csv_path)
    closest_nom_img_address = nom_data['center'].iloc[int(pos_mappings[anomalous_frame])]
    ano_img_address = correct_windows_path(ano_img_address)
    closest_nom_img_address = correct_windows_path(closest_nom_img_address)
    ano_img = Image.open(ano_img_address)
    closest_nom_img = Image.open(closest_nom_img_address)   
    return ano_img, closest_nom_img

def save_ax_nosave(ax, **kwargs):
    import io
    ax.axis("on")
    ax.figure.canvas.draw()
    trans = ax.figure.dpi_scale_trans.inverted() 
    bbox = ax.bbox.transformed(trans)
    buff = io.BytesIO()
    plt.savefig(buff, format="png", dpi=ax.figure.dpi, bbox_inches=bbox,  **kwargs)
    # ax.axis("on")
    buff.seek(0)
    # im = plt.imread(buff)
    im = Image.open(buff)
    return im

# Video creation functions

def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)

def make_avi(image_folder, video_folder_path, name):
    video_name = f'{name}.avi'
    video_path = os.path.join(video_folder_path, video_name)
    if not os.path.exists(video_path):
        cprintf('Creating video ...', 'l_cyan')
        # path to video folder
        if not os.path.exists(video_folder_path):
            os.makedirs(video_folder_path)
        images = [img for img in os.listdir(image_folder)]
        sort_nicely(images)
        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape
        fps = 10
        video = cv2.VideoWriter(video_path, 0, fps, (width,height))
        for image in tqdm(images):
            video.write(cv2.imread(os.path.join(image_folder, image)))
        cv2.destroyAllWindows()
        video.release()
    else:
        cprintf('Video already exists. Skipping video creation ...', 'l_green')

def make_gif(frame_folder, name):
    frames = [Image.open(image) for image in glob.glob(f"{frame_folder}/*.PNG")]
    frame_one = frames[0]
    frame_one.save(f"{name}.gif", format="GIF", append_images=frames,
               save_all=True, duration=100, loop=0)

def average_filter_1D(data_array, kernel=np.ones((5), dtype=float)):

    if not isinstance(kernel, np.ndarray):
        raise ValueError(Fore.RED + f"The provided kernel '{kernel}' is not a numpy array." + Fore.RESET)
    elif not isinstance(data_array, np.ndarray):
        raise ValueError(Fore.RED + f"The provided data is not a numpy array." + Fore.RESET)
    elif not ((kernel.ndim == 1) and (data_array.ndim == 1)):
        raise ValueError(Fore.RED + f"The provided numpy arrays must have 1 dimension." + Fore.RESET)
    
    filtered_array = np.zeros((len(data_array)), dtype=float)
    kernel_length = len(kernel)
    kernel_range = math.floor(len(kernel)/2)

    for dp_index, data_point in enumerate(data_array):
        data_window = np.zeros((kernel_length), dtype=float)
        data_window[kernel_range] = data_point
        for kernel_index in range(1, kernel_range+1):
            if not ((dp_index - kernel_index) < 0):
                data_window[kernel_range-kernel_index] = data_array[dp_index-kernel_index]

            if not ((dp_index + kernel_index) > len(data_array)-1):
                data_window[kernel_range+kernel_index] = data_array[dp_index+kernel_index]
        filtered_array[dp_index] = np.average(np.multiply(kernel, data_window))
    return filtered_array

def window_slope(data_array, window_length=5):

    if not isinstance(data_array, np.ndarray):
        raise ValueError(Fore.RED + f"The provided data is not a numpy array." + Fore.RESET)
    
    slope_array = np.zeros((len(data_array)), dtype=float)

    for dp_index, data_point in enumerate(data_array):
        if dp_index < window_length-2:
            continue
        else:
            win_first_element = data_array[dp_index-window_length-1]
            win_last_element = data_point
            slope_array[dp_index] = win_last_element - win_first_element

    return slope_array


def Morans_I(data, plot=False):
    """transforming RGB data to grayscale"""
    data_gray = np.dot(data[...,:3], [0.2989, 0.5870, 0.1140])
    col,row = data_gray.shape[:2]
    WeightMatrix= lat2W(row,col)
    WeightMatrix = lat2W(data_gray.shape[0],data_gray.shape[1])
    MoranM= Moran(data_gray,WeightMatrix)
    if plot:
        fig, ax = moran_scatterplot(MoranM, aspect_equal=True)
        print("Raster Dimensions:\t" + str(data_gray.shape))
        print("Moran's I Value:\t" +str(round(MoranM.I,4)))
        plt.show()
    return round(MoranM.I,4)


def h_minus_1_sobolev_norm(A, B):
    # Compute the Fourier transform of the difference of the heatmaps
    fft_diff = np.fft.fft2(A - B)
    # Create a meshgrid of frequencies (wavevectors)
    nx, ny = A.shape
    kx = np.fft.fftfreq(nx)
    ky = np.fft.fftfreq(ny)
    kx, ky = np.meshgrid(kx, ky, indexing='ij')  # Use 'ij' indexing to match array shapes
    # Calculate the wavenumber |k|
    wavenumber = np.sqrt(kx**2 + ky**2)
    # Discount Fourier coefficients by the wavenumber and sum them up
    norm_squared = np.sum(np.abs(fft_diff)**2 / (1 + (2 * np.pi * wavenumber)**2))
    return np.sqrt(norm_squared)

def p2p_lineplot(ax, distance_vector, distance_vector_avg, distance_type, heatmap_type, color, color_avg,
             eval_vars=None, eval_method=None, spine_color='black', alpha=0.4, avg_filter_length=5,
             pca_dimension=None, pca_plot=False, replace_initial_and_ending_values=True):
    # Plot distance scores
    # ax.set_xlabel('Frame ID', color=color)
    ax.set_ylabel(f'{distance_type} scores', color=color)

    ax_spines = ['top', 'right', 'left', 'bottom']
    for spine in ax_spines:
        ax.spines[spine].set_color(spine_color)

    ax.plot(distance_vector, label=distance_type, linewidth= 0.5, linestyle = '-', color=color, alpha=alpha)

    if replace_initial_and_ending_values:
        for rng in range(math.floor(avg_filter_length/2)):
                idx = rng
                distance_vector_avg[idx] = distance_vector[idx]
                idx = -(rng+1)
                distance_vector_avg[idx] = distance_vector[idx]

    ax.plot(distance_vector_avg, label=f'avg_filter({avg_filter_length})', linewidth= 0.8, linestyle = 'dashed', color=color_avg)
    
    # persistent_slope = []
    if eval_method == 'threshold':
        if len(eval_vars) > 1:
            MULTI_VAR = True
        else:
            MULTI_VAR = False
        threshold = eval_vars[0]
        if MULTI_VAR:
            ano_threshold = eval_vars[1]
            ax.axhline(y = ano_threshold, color = 'blue', linestyle = '--')
        ax.axhline(y = threshold, color = 'red', linestyle = '--')
        trans = transforms.blended_transform_factory(ax.get_yticklabels()[0].get_transform(), ax.transData)
        ax.text(0, threshold, "{:.0f}".format(threshold), color="red", transform=trans, ha="right", va="center")
        if MULTI_VAR:
            ax.text(0, ano_threshold, "{:.0f}".format(threshold), color="blue", transform=trans, ha="right", va="center")
        # avg_slope = window_slope(distance_vector_avg)
        for frame, distance_score in enumerate(distance_vector_avg):
                
            # if (distance_score < threshold) and (avg_slope[frame] > 100):
            #     persistent_slope.append(True)
            # else:
            #     persistent_slope = []
            
            # if len(persistent_slope) != 0 and test_print:
            #     print(len(persistent_slope), frame)
            if distance_score > threshold:
                color = 'red'
            # elif len(persistent_slope) > 3:
            #     color = 'red'
            else:
                color = 'lime'
            ax.hlines(y=threshold, xmin=frame, xmax=frame+1, color=color, linewidth=3)
        if MULTI_VAR:
            for frame, distance_score in enumerate(distance_vector_avg):
                if distance_score > ano_threshold:
                    color = 'red'
                else:
                    color = 'lime'
                ax.hlines(y=ano_threshold, xmin=frame, xmax=frame+1, color=color, linewidth=3)

    if pca_plot:
        bolded_part = f": {distance_type} - PCA {pca_dimension}d"
        title = heatmap_type + r"$\bf{" + bolded_part + "}$"
    else:
        bolded_part = f": {distance_type}"
        title = heatmap_type + r"$\bf{" + bolded_part + "}$"
    # plt.title("This is title number: " + r"$\bf{" + str(number) + "}$")
    ax.set_title(title, color=spine_color)
    ax.legend(loc='upper left')

    # set tick and ticklabel color
    ax.tick_params(axis='x', colors=spine_color)    #setting up X-axis tick color to red
    ax.tick_params(axis='y', colors=spine_color)  #setting up Y-axis tick color to black
    ax.set_xticks(np.arange(0, len(distance_vector), 50.0), minor=False)


def score_when_decrease(output):
    return -1.0 * output[:, 0]

def splitall(path):
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path: # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts

def fix_escape_sequences(img_addr):
    if "\\\\" in img_addr:
        img_addr = img_addr.replace("\\\\", "/")
    elif "\\" in img_addr:
        img_addr = img_addr.replace("\\", "/")
    elif "\\a" in img_addr:
        img_addr = img_addr.replace("\\a", "/a")
    elif "\\t" in img_addr:
        img_addr = img_addr.replace("\\t", "/t")
    elif "\\n" in img_addr:
        img_addr = img_addr.replace("\\n", "")
    elif "\\b" in img_addr:
        img_addr = img_addr.replace("\\b", "")
    return img_addr


######################
# EVALUATE ALL UTILS #
######################

def get_paths(cfg, run_id, sim_name, attention_type, sim_type, method):
    if type(run_id) != str:
        run_id = str(run_id)
    SIM_PATH = os.path.join(cfg.TESTING_DATA_DIR, cfg.TRACK, sim_type, sim_name)
    RUN_RESULTS_PATH = os.path.join(SIM_PATH, 'results', run_id)
    if cfg.SPARSE_ATTRIBUTION:
        RUN_FIGS_PATH = os.path.join(RUN_RESULTS_PATH, 'SPARSE_FIGS', method)
    else:
        RUN_FIGS_PATH = os.path.join(RUN_RESULTS_PATH, 'FIGS', method)
    if sim_type == 'nominal':
        nominal = True
        threshold = False
    elif sim_type == 'anomalous':
        nominal = False
        threshold = False
    elif sim_type == 'threshold':
        nominal = False
        threshold = True

    if not (nominal or threshold):
        MAIN_CSV_PATH = os.path.join(SIM_PATH, "src", run_id, "driving_log.csv")
    else:
        MAIN_CSV_PATH = os.path.join(SIM_PATH, "src", "driving_log.csv")

    HEATMAP_PARENT_FOLDER_PATH = os.path.join(SIM_PATH, "heatmaps", "heatmaps-" + attention_type.lower())

    if not nominal:
        if cfg.SPARSE_ATTRIBUTION:
            HEATMAP_FOLDER_PATH = os.path.join(HEATMAP_PARENT_FOLDER_PATH, f'{run_id}_SPARSE')
            HEATMAP_CSV_PATH = os.path.join(HEATMAP_PARENT_FOLDER_PATH, f'{run_id}_SPARSE', "driving_log.csv")
            HEATMAP_IMG_PATH = os.path.join(HEATMAP_PARENT_FOLDER_PATH, f'{run_id}_SPARSE', "IMG")
            HEATMAP_IMG_GRADIENT_PATH = os.path.join(HEATMAP_PARENT_FOLDER_PATH, f'{run_id}_SPARSE', "IMG_GRADIENT")
        else:
            HEATMAP_FOLDER_PATH = os.path.join(HEATMAP_PARENT_FOLDER_PATH, run_id)
            HEATMAP_CSV_PATH = os.path.join(HEATMAP_PARENT_FOLDER_PATH, run_id, "driving_log.csv")
            HEATMAP_IMG_PATH = os.path.join(HEATMAP_PARENT_FOLDER_PATH, run_id, "IMG")
            HEATMAP_IMG_GRADIENT_PATH = os.path.join(HEATMAP_PARENT_FOLDER_PATH, run_id, "IMG_GRADIENT")
    else:
        HEATMAP_FOLDER_PATH = HEATMAP_PARENT_FOLDER_PATH
        if cfg.SPARSE_ATTRIBUTION:
            HEATMAP_CSV_PATH = os.path.join(HEATMAP_PARENT_FOLDER_PATH, 'SPARSE', "driving_log.csv")
            HEATMAP_IMG_PATH = os.path.join(HEATMAP_PARENT_FOLDER_PATH, 'SPARSE', "IMG")
            HEATMAP_IMG_GRADIENT_PATH = os.path.join(HEATMAP_PARENT_FOLDER_PATH, 'SPARSE', "IMG_GRADIENT")
        else:
            HEATMAP_CSV_PATH = os.path.join(HEATMAP_PARENT_FOLDER_PATH, "driving_log.csv")
            HEATMAP_IMG_PATH = os.path.join(HEATMAP_PARENT_FOLDER_PATH, "IMG")
            HEATMAP_IMG_GRADIENT_PATH = os.path.join(HEATMAP_PARENT_FOLDER_PATH, "IMG_GRADIENT")

    NUM_OF_FRAMES = get_num_frames(run_id, SIM_PATH, MAIN_CSV_PATH)
    # check if img paths in the csv file need correcting (possible directory change of simulation data)
    correct_img_paths_in_csv_files(MAIN_CSV_PATH)
    NPY_SCORES_FOLDER_PATH = os.path.join(HEATMAP_PARENT_FOLDER_PATH, "NPY_SCORES")
    PATHS = [SIM_PATH, MAIN_CSV_PATH, HEATMAP_PARENT_FOLDER_PATH, HEATMAP_FOLDER_PATH, HEATMAP_CSV_PATH, HEATMAP_IMG_PATH, HEATMAP_IMG_GRADIENT_PATH, RUN_RESULTS_PATH, RUN_FIGS_PATH, NPY_SCORES_FOLDER_PATH]
    result = {
    'paths': PATHS,
    'sim_type': [nominal, threshold],
    "num_of_frames": NUM_OF_FRAMES   
    }
    return result


def heatmap_calculation_modus(cfg, run_id, sim_name, attention_type, sim_type, method):
    if type(run_id) != str:
        run_id = str(run_id)
    get_paths_result = get_paths(cfg, run_id, sim_name, attention_type, sim_type, method)
    PATHS = get_paths_result['paths']
    SIM_PATH, MAIN_CSV_PATH, HEATMAP_PARENT_FOLDER_PATH, HEATMAP_FOLDER_PATH, HEATMAP_CSV_PATH, HEATMAP_IMG_PATH, HEATMAP_IMG_GRADIENT_PATH, RUN_RESULTS_PATH, RUN_FIGS_PATH, NPY_SCORES_FOLDER_PATH = PATHS
    nominal, threshold = get_paths_result['sim_type']
    NUM_OF_FRAMES = get_paths_result['num_of_frames']
        
    if not os.path.exists(SIM_PATH):
        raise ValueError(Fore.RED + f"The provided simulation path does not exist: {SIM_PATH}" + Fore.RESET)
    else:
        # 1- IMG & Gradient folders don't exist 
        if (not os.path.exists(HEATMAP_IMG_PATH)) and (not os.path.exists(HEATMAP_IMG_GRADIENT_PATH)):
            validation_warnings(False, sim_name, attention_type, run_id, nominal, threshold)
            cprintf(f"Neither heatmap IMG folder nor gradient folder exist. Creating image folder at {HEATMAP_IMG_PATH} and {HEATMAP_IMG_GRADIENT_PATH}", 'l_blue')
            os.makedirs(HEATMAP_IMG_PATH)
            os.makedirs(HEATMAP_IMG_GRADIENT_PATH)
            MODE = 'new_calc'

        # 2- Heatmap IMG folder exists but gradient folder doesn't
        elif (os.path.exists(HEATMAP_IMG_PATH)) and (not os.path.exists(HEATMAP_IMG_GRADIENT_PATH)):
            validation_warnings(False, sim_name, attention_type, run_id, nominal, threshold)
            if (len(os.listdir(HEATMAP_IMG_PATH)) < NUM_OF_FRAMES):
                cprintf(f"Heatmap folder exists, but there are less heatmaps than there should be.", 'yellow')
                cprintf(f"Deleting folder at {HEATMAP_IMG_GRADIENT_PATH}", 'yellow')
                shutil.rmtree(HEATMAP_IMG_PATH)
                cprintf(f"Switching to new_calc mode ...", 'yellow')
                MODE = 'new_calc'
            else:
                cprintf(f"Heatmap IMG folder exists and is complete, but gradient folder doesn't. Creating image folder at {HEATMAP_IMG_GRADIENT_PATH}", 'l_blue')
                os.makedirs(HEATMAP_IMG_GRADIENT_PATH)
                MODE = 'gradient_calc'

        # 3- Heatmap IMG folder doesn't exist but gradient folder does
        elif (not os.path.exists(HEATMAP_IMG_PATH)) and (os.path.exists(HEATMAP_IMG_GRADIENT_PATH)):
            validation_warnings(False, sim_name, attention_type, run_id, nominal, threshold)
            if (len(os.listdir(HEATMAP_IMG_GRADIENT_PATH)) < NUM_OF_FRAMES-1):
                cprintf(f"Gradient folder exists, but there are less gradients than there should be.", 'yellow')
                cprintf(f"Deleting folder at {HEATMAP_IMG_GRADIENT_PATH}", 'yellow')
                shutil.rmtree(HEATMAP_IMG_GRADIENT_PATH)
                cprintf(f"Switching to new_calc mode ...", 'yellow')
                MODE = 'new_calc'
            else:
                cprintf(f"Heatmap IMG folder doesn't exist but gradient folder exits and is complete. Creating image folder at {HEATMAP_IMG_PATH}", 'l_blue')
                os.makedirs(HEATMAP_IMG_PATH)
                MODE = 'heatmap_calc'

        # 4- Both folders exist, but there are less heatmaps/gradients than there should be
        elif (len(os.listdir(HEATMAP_IMG_PATH)) < NUM_OF_FRAMES) and (len(os.listdir(HEATMAP_IMG_GRADIENT_PATH)) < NUM_OF_FRAMES-1):
            validation_warnings(False, sim_name, attention_type, run_id, nominal, threshold)
            cprintf(f"Both folders exist, but there are less heatmaps/gradients than there should be.", 'yellow')
            cprintf(f"Deleting folder at {HEATMAP_IMG_PATH} and {HEATMAP_IMG_GRADIENT_PATH}", 'yellow')
            shutil.rmtree(HEATMAP_IMG_PATH)
            shutil.rmtree(HEATMAP_IMG_GRADIENT_PATH)
            MODE = 'new_calc'

        # 5- both folders exist, but there are less heatmaps than there should be (gradients are the correct number)
        elif (len(os.listdir(HEATMAP_IMG_PATH)) < NUM_OF_FRAMES) and (len(os.listdir(HEATMAP_IMG_GRADIENT_PATH)) == NUM_OF_FRAMES-1):
            validation_warnings(False, sim_name, attention_type, run_id, nominal, threshold)
            cprintf(f"Both folders exist, but there are less heatmaps than there should be (gradients are the correct number).", 'yellow')
            cprintf(f"Deleting folder at {HEATMAP_IMG_PATH}", 'yellow')
            shutil.rmtree(HEATMAP_IMG_PATH)
            MODE = 'heatmap_calc'

        # 6- gradient folder exists, but there are less gradients than there should be (heatmaps are the correct number)
        elif (len(os.listdir(HEATMAP_IMG_PATH)) == NUM_OF_FRAMES) and (len(os.listdir(HEATMAP_IMG_GRADIENT_PATH)) < NUM_OF_FRAMES-1):
            validation_warnings(False, sim_name, attention_type, run_id, nominal, threshold)
            cprintf(f"Both folders exist, but there are less gradients than there should be (heatmaps are the correct number).", 'yellow')
            cprintf(f"Deleting folder at {HEATMAP_IMG_GRADIENT_PATH}", 'yellow')
            shutil.rmtree(HEATMAP_IMG_GRADIENT_PATH)
            MODE = 'gradient_calc'

        # 7- heatmap folder exists and correct number of heatmaps, but no csv file was generated.
        elif not 'driving_log.csv' in os.listdir(HEATMAP_FOLDER_PATH):
            validation_warnings(False, sim_name, attention_type, run_id, nominal, threshold)
            cprintf(f"Correct number of heatmaps exist. CSV File doesn't.", 'yellow')
            MODE = 'csv_missing'
        
        # 8- everything exists except for the .npy score files
        elif (not os.path.exists(NPY_SCORES_FOLDER_PATH)) or (len(os.listdir(NPY_SCORES_FOLDER_PATH)) < 4):
            cprintf(f"Incorrect number of .npy score files.", 'yellow')
            MODE = 'score_calc'
        else:
            MODE = None
            validation_warnings(True, sim_name, attention_type, run_id, nominal, threshold)

        if (MODE != 'csv_missing') and (MODE != 'heatmap_calc') and (MODE != 'new_calc'):
            # check if img paths in the csv file need correcting (possible directory change of simulation data)
            correct_img_paths_in_csv_files(HEATMAP_CSV_PATH)
    return MODE


def validation_warnings(valid, sim_name, attention_type, run_id, nominal, threshold):
    if type(run_id) != str:
        run_id = str(run_id)
    if valid:
        if nominal:
            cprintf(f"Heatmaps for nominal \u2713 \"{sim_name}\" of attention type \"{attention_type}\" exist.", 'l_green')
        elif threshold:
            cprintf(f"Heatmaps for threshold sim \u2713 \"{sim_name}\" of attention type \"{attention_type}\" and run ID \"{run_id}\" exist.", 'l_green')
        else:
            cprintf(f"Heatmaps for anomalous sim \u2713 \"{sim_name}\" of attention type \"{attention_type}\" and run ID \"{run_id}\" exist.", 'l_green')
    else:
        if nominal:
            cprintf(f"WARNING: Nominal \u2717 sim \"{sim_name}\" of attention type \"{attention_type}\" heatmap folder has deficiencies. Looking for a solution... ", "red")
        elif threshold:
            cprintf(f"WARNING: Threshold \u2717 sim \"{sim_name}\" of attention type \"{attention_type}\" and run ID \"{run_id}\" heatmap folder has deficiencies. Looking for a solution... ", "red")
        else:
            cprintf(f"WARNING: Anomalous \u2717 sim \"{sim_name}\" of attention type \"{attention_type}\" and run ID \"{run_id}\" heatmap folder has deficiencies. Looking for a solution... ", "red")


######## In case the data was copied into another directory the image paths inside csv files must be corrected ########

def correct_img_address(img_addr, csv_dir):
    img_name = Path(img_addr).stem
    corrected_path = os.path.join(csv_dir, 'IMG', img_name + '.jpg')
    return corrected_path

def check_addresses(center_img_addresses, csv_dir):
    first_center_img_address = fix_escape_sequences(center_img_addresses[0])
    if not os.path.exists(first_center_img_address):
        corrected_path = correct_img_address(first_center_img_address, csv_dir)
        if not os.path.exists(corrected_path):
            raise ValueError(Fore.RED + f"The provided img path in the csv file is not in the same dir or does not exist: {corrected_path}" + Fore.RESET)
        return False
    else:
        return True
    
def correct_img_paths_in_csv_files(CSV_PATH):

    csv_dir = os.path.dirname(CSV_PATH)
    csv_file = pd.read_csv(CSV_PATH)
    center_img_addresses = csv_file["center"]

    if 'left' in csv_file.columns:
        left_img_addresses = csv_file["left"]
    
    if 'right' in csv_file.columns:
        right_img_addresses = csv_file["right"]

    # if the img exists in the correct path but the path in the csv file is wrong:
    if not check_addresses(center_img_addresses, csv_dir):
        # center images
        for img_addr in center_img_addresses:
            fixed_img_addr = fix_escape_sequences(img_addr)
            corrected_path = correct_img_address(fixed_img_addr, csv_dir)
            csv_file.replace(to_replace=img_addr, value=corrected_path, inplace=True)
        if 'left' in csv_file.columns:
            # left images
            for img_addr in left_img_addresses:
                fixed_img_addr = fix_escape_sequences(img_addr)
                corrected_path = correct_img_address(fixed_img_addr, csv_dir)
                csv_file.replace(to_replace=img_addr, value=corrected_path, inplace=True)
        if 'right' in csv_file.columns:
            # right images
            for img_addr in right_img_addresses:
                fixed_img_addr = fix_escape_sequences(img_addr)
                corrected_path = correct_img_address(fixed_img_addr, csv_dir)
                csv_file.replace(to_replace=img_addr, value=corrected_path, inplace=True)
        
        csv_file.to_csv(CSV_PATH, index=False)


######## In case of manual training data the simulator doesn't save the field names (column names) ########

def correct_csv_field_names(SIM_PATH, MAIN_CSV_PATH, run_id):
    if type(run_id) != str:
        run_id = str(run_id)
    UPDATED_CSV_PATH = os.path.join(SIM_PATH, "src", run_id, "driving_log_updated.csv")
    try:
        # autonomous mode simulation data (is going to fail if manual, since manual has address string in first column)
        data_df = pd.read_csv(MAIN_CSV_PATH)
        NUM_OF_FRAMES = pd.Series.max(data_df["frameId"]) + 1
    except:
        # manual mode simulation data
        with open(MAIN_CSV_PATH,'r') as f:
            lines = f.readlines()[1:]
            with open(UPDATED_CSV_PATH,'w') as f1:
                 for line in lines:
                    f1.write(line)
        df = pd.read_csv(UPDATED_CSV_PATH, header=None)
        df.rename(columns={ 0: csv_fieldnames_in_manual_mode[0], 1: csv_fieldnames_in_manual_mode[1],
                            2: csv_fieldnames_in_manual_mode[2], 3: csv_fieldnames_in_manual_mode[3],
                            4: csv_fieldnames_in_manual_mode[4], 5: csv_fieldnames_in_manual_mode[5],
                            6: csv_fieldnames_in_manual_mode[6], 7: csv_fieldnames_in_manual_mode[7],
                            8: csv_fieldnames_in_manual_mode[8], 9: csv_fieldnames_in_manual_mode[9],
                            10: csv_fieldnames_in_manual_mode[10]}, inplace=True)
        df['frameId'] = df.index
        df.to_csv(MAIN_CSV_PATH, index=False) # save to new csv file
        os.remove(UPDATED_CSV_PATH)
        data_df = pd.read_csv(MAIN_CSV_PATH)
        NUM_OF_FRAMES = pd.Series.max(data_df["frameId"]) + 1
    return data_df


def get_num_frames(run_id, SIM_PATH, MAIN_CSV_PATH):
    # add field names to main csv file if it's not there (manual training recording)
    csv_df = correct_csv_field_names(SIM_PATH, MAIN_CSV_PATH, run_id)
    # get number of frames
    NUM_OF_FRAMES = pd.Series.max(csv_df['frameId']) + 1
    return NUM_OF_FRAMES


def copy_run_figs(cfg, SIM_PATH, run_id, run_figs, RUN_FIGS_FOLDER_PATH):
    if type(run_id) != str:
        run_id = str(run_id)

    if not os.path.exists(RUN_FIGS_FOLDER_PATH):
        cprintf(f'Run figure folder does not exist. Creating folder ...' ,'l_blue')
        os.makedirs(RUN_FIGS_FOLDER_PATH)
    
    cprintf(f"Copying run figures of all assigned heatmap types to: {RUN_FIGS_FOLDER_PATH}", 'l_cyan')
    for run_fig_address in tqdm(run_figs):
        shutil.copy(run_fig_address, RUN_FIGS_FOLDER_PATH)


def delete_contents_except(directory, directory_to_keep):
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            # Skip the directory you want to keep
            if item_path != directory_to_keep:
                shutil.rmtree(item_path)
        else:
            # Delete files
            os.remove(item_path)

#################################
# PROCESSING EVAL RESULTS UTILS #
#################################

def f_beta_score(precision, recall, beta=3):
    numerator = (1 + beta ** 2) * (precision * recall)
    denominator = (beta ** 2 * precision) + recall
    f_beta_score = numerator / denominator
    return f_beta_score

def heatmap_or_distance_type_scores(total_scores_paths, HEATMAP_OR_DISTANCE_TYPES, sim_idx, sim_name, number_of_runs,
                        seconds_to_anticipate, scores_df, last_index, type = 'heatmap_type', print_results=False):
    if scores_df is None:
        raise ValueError("Received None as dataframe")
    
    # add simulation name row
    sim_name_row = ['Simulation']
    for col_idx in range((len(seconds_to_anticipate)+1)*4):
        if col_idx == 0:
            sim_name_row.append(sim_name)
        else:
            sim_name_row.append('-')
    scores_df.iloc[last_index] = sim_name_row

    for heatmap_or_distance_type in HEATMAP_OR_DISTANCE_TYPES:
        last_index += 1
        if print_results:
            cprintf(f'-------------------------------------------------------------------------------------------------------', 'l_cyan')
            cprintf(f'{heatmap_or_distance_type}', 'l_cyan')
        # replace 0th column with heatmap or distance names
        scores_df.at[last_index, 'Window Size'] = heatmap_or_distance_type
        # reset column number to 1 for after each heatmap or distance type
        col_ctr = 1

        # reset sum arrays including sum vars for each sta for after each heatmap or distance type
        avg_prec_sum = np.zeros((len(seconds_to_anticipate)), dtype=float)
        avg_re_sum = np.zeros((len(seconds_to_anticipate)), dtype=float)
        f3_scores_sum = np.zeros((len(seconds_to_anticipate)), dtype=float)
        avg_acc_sum = np.zeros((len(seconds_to_anticipate)), dtype=float)
        
        avg_prec_all_sta_sum = 0.0
        avg_rec_all_sta_sum = 0.0
        f3_score_all_sta_sum = 0.0
        avg_acc_all_sta_sum = 0.0

        for run_idx, run_id in enumerate(number_of_runs[sim_idx]):
            for sta_idx, sta in enumerate(seconds_to_anticipate):
                if print_results:
                    cprintf(f'------------------------------------ STA: {str(sta)} --------- RUN_NUMBER: {str(run_id)} ------------------------------------', 'l_cyan')
                    cprintf(f'SIM_NAME: {sim_name}', 'l_cyan')
                    # cprintf(f'rows: {len(total_scores_paths)}, cols: {len(total_scores_paths[0])}', 'l_yellow')
                    cprintf(f'SIM_IDX: {sim_idx}, RUN_IDX: {run_idx}', 'l_magenta')
                    cprintf(f'{total_scores_paths[sim_idx][run_idx]}', 'l_green')

                # read heatmap or distance or  results csv file for this run_id and filter by heatmap or distance type
                results_df = pd.read_csv(total_scores_paths[sim_idx][run_idx])
                filter_by_sta = results_df[(results_df['sta'] == sta) & (results_df[type] == heatmap_or_distance_type)]
                # precision
                precision = filter_by_sta['precision'].values
                avg_prec_sum[sta_idx] += precision
                if print_results:
                    print(f'avg_prec_sum[sta_idx:{sta_idx}]: {avg_prec_sum}')
                # recall
                recall = filter_by_sta['recall'].values
                avg_re_sum[sta_idx] += recall
                if print_results:
                    print(f'avg_re_sum[sta_idx:{sta_idx}]: {avg_re_sum}')
                # f3
                f3_score = f_beta_score(precision, recall, beta=3)
                f3_scores_sum[sta_idx] += f3_score
                if print_results:
                    print(f'f3_scores_sum[sta_idx:{sta_idx}]: {f3_scores_sum}')
                # accuracy
                accuracy = filter_by_sta['accuracy'].values
                avg_acc_sum[sta_idx] += accuracy
                if print_results:
                    print(f'avg_acc_sum[sta_idx:{sta_idx}]: {avg_acc_sum}')

                if print_results:
                    cprintf(f'sta: avg_precision: {sta}: {precision*100}', 'l_green')
                    cprintf(f'sta: avg_recall: {sta}: {recall*100}', 'l_yellow')
                    cprintf(f'sta: f3_score: {sta}: {f3_score*100}', 'l_red')    
                    cprintf(f'sta: avg_accuracy: {sta}: {accuracy*100}', 'l_blue')
            
                # if last run then save average scores between multiple runs to dataframe
                if run_id == number_of_runs[sim_idx][-1]:
                    avg_prec = avg_prec_sum[sta_idx]/len(number_of_runs[sim_idx])
                    avg_re = avg_re_sum[sta_idx]/len(number_of_runs[sim_idx])
                    avg_f3 = f3_scores_sum[sta_idx]/len(number_of_runs[sim_idx])
                    avg_acc = avg_acc_sum[sta_idx]/len(number_of_runs[sim_idx])

                    if print_results:
                        print(f'avg_prec: {avg_prec}')
                        print(f'avg_re: {avg_re}')
                        print(f'avg_f3: {avg_f3}')
                        print(f'avg_acc: {avg_acc}')
                        # print(last_index, col_ctr)
                    scores_df.iat[last_index, col_ctr] = avg_prec
                    scores_df.iat[last_index, col_ctr+1] = avg_re
                    scores_df.iat[last_index, col_ctr+2] = avg_f3
                    scores_df.iat[last_index, col_ctr+3] = avg_acc
                    col_ctr += 4
                    if print_results:
                        # print(last_index, col_ctr)
                        print(f'ALL RUNS FOR STA {str(sta)} DONE ######################################################')
                    if sta == 3:
                        all_sta_all_runs_avg_prec = np.average(avg_prec_sum)/len(number_of_runs[sim_idx])
                        all_sta_all_runs_avg_re = np.average(avg_re_sum)/len(number_of_runs[sim_idx])
                        all_sta_all_runs_avg_f3 = np.average(f3_scores_sum)/len(number_of_runs[sim_idx])
                        all_sta_all_runs_avg_acc = np.average(avg_acc_sum)/len(number_of_runs[sim_idx])
                        if print_results:
                            print(f'all_sta_all_runs_avg_prec: {all_sta_all_runs_avg_prec}')
                            print(f'all_sta_all_runs_avg_re: {all_sta_all_runs_avg_re}')
                            print(f'all_sta_all_runs_avg_f3: {all_sta_all_runs_avg_f3}')
                            print(f'all_sta_all_runs_avg_acc: {all_sta_all_runs_avg_acc}')
                            # print(last_index, col_ctr)
                        scores_df.iat[last_index, col_ctr] = all_sta_all_runs_avg_prec
                        scores_df.iat[last_index, col_ctr+1] = all_sta_all_runs_avg_re
                        scores_df.iat[last_index, col_ctr+2] = all_sta_all_runs_avg_f3
                        scores_df.iat[last_index, col_ctr+3] = all_sta_all_runs_avg_acc
                        col_ctr += 4
                        if print_results:
                            print(f'ALL RUNS FOR STA all DONE ######################################################')                    
    last_index += 1
    return scores_df, last_index

def create_result_df(total_scores_paths, DISTANCE_OR_HEATMAP_TYPES, ANO_SIMULATIONS, sim_idx, number_of_runs, seconds_to_anticipate_str):
    # test if the number of paths and runs are the same:
    if not (len(total_scores_paths[sim_idx]) == len(number_of_runs[sim_idx])):
            raise ValueError(Fore.RED + f"Mismatch in number of runs per simulation and number of heatmap result score csv paths: {len(total_scores_paths[sim_idx])} != {len(number_of_runs[sim_idx])}" + Fore.RESET)
    # build the top rows of column names
    scores_df = pd.DataFrame(np.zeros(((len(DISTANCE_OR_HEATMAP_TYPES)+1)*len(ANO_SIMULATIONS), 17)))
    col_names_row_1 = ['Window Size']
    col_names_row_2 = ['Criteria']
    for sta_str in seconds_to_anticipate_str:
        for col_idx in range(4):
            col_names_row_1.append(sta_str)
        col_names_row_2.append('Precision')
        col_names_row_2.append('Recall')
        col_names_row_2.append('F3')
        col_names_row_2.append('Accuracy')
    scores_df.columns = pd.MultiIndex.from_arrays([col_names_row_1, col_names_row_2])
    return scores_df

heatmap_type_colors = {
    'gradcam++' : ('deepskyblue', 'navy'),
    'rectgrad' : ('indianred', 'brown'),
    'smoothgrad' : ('mediumslateblue', 'darkslateblue'),
    'rectgrad_prr' : ('mediumseagreen', 'darkgreen'),
    'saliency' :('lightslategrey', 'darkslategrey'),
    'guided_bp' : ('wheat','orange'),
    'gradient-input' : ('darkseagreen', 'darkolivegreen'),
    'integgrad' : ('goldenrod', 'darkgoldenrod'),
    'epsilon_lrp': ('salmon', 'maroon')}

distance_type_colors = {
    'euclidean' : ('deepskyblue', 'navy'),
    'manhattan' : ('indianred', 'brown'),
    'cosine' : ('mediumslateblue', 'darkslateblue'),
    'sobolev-norm' : ('mediumseagreen', 'darkgreen'),
    'EMD' :('lightslategrey', 'darkslategrey'),
    'pearson' : ('wheat','orange'),
    'spearman' : ('darkseagreen', 'darkolivegreen'),
    'moran' : ('goldenrod', 'darkgoldenrod'),
    'kl-divergence': ('salmon', 'maroon'),
    'mutual-info': ('salmon', 'maroon')}

summary_type_colors = {
    '-avg' : ('deepskyblue', 'navy'),
    '-avg-grad' : ('indianred', 'brown')}

def ht_or_dt_comparison_plot(ANO_SIMULATIONS, seconds_to_anticipate_str, STA_PLOT, HEATMAP_OR_DISTANCE_TYPES, criterion_vals, PLOTTING_CRITERION, FIG_SAVE_ADDRESS, type):
    # Plotting
    plt.figure(figsize=(10, 7))
    x = ANO_SIMULATIONS
    # for sim_idx, sim_name in enumerate(ANO_SIMULATIONS):
    y = []
    line_color_type = 0 # 0 or 1 for each heatmap type
    for sta_idx, sta in enumerate(seconds_to_anticipate_str):
        if not STA_PLOT[sta_idx]:
            continue
        for ht_or_dt_idx, ht_or_dt in enumerate(HEATMAP_OR_DISTANCE_TYPES):
            for sim_idx, _ in enumerate(ANO_SIMULATIONS):
                row_idx_for_current_sta_and_hm= sim_idx*len(HEATMAP_OR_DISTANCE_TYPES) + ht_or_dt_idx
                y.append(criterion_vals[row_idx_for_current_sta_and_hm][sta_idx])
            if type == 'Heatmap':
                color = heatmap_type_colors[ht_or_dt][line_color_type]
            elif type == 'Distance':
                color = distance_type_colors[ht_or_dt][line_color_type]
            else:
                raise ValueError(f"Unknown eval results comparison plot type: {type}")
            plt.scatter(x, y, s=200, marker=r"$ {} $".format(sta), color=color)
            plt.plot(x, y, label=ht_or_dt, color=color)
            for x,y in zip(x,y):
                if not abs(y-1.00)<=0.0001:
                    label = "{:.4f}".format(y)

                    plt.annotate(label, # this is the text
                                (x,y), # these are the coordinates to position the label
                                textcoords="offset points", # how to position the text
                                xytext=(0,10), # distance from text to points (x,y)
                                ha='center',
                                color=color) # horizontal alignment can be left, right or center
            y = []
            x = ANO_SIMULATIONS

    # Customize plot
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Lighting/Weather Conditions')
    plt.ylabel(PLOTTING_CRITERION)
    plt.title(f'Performance Comparison of Different {type} Types')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.grid(True)

    # Show plot
    plt.tight_layout()
    plt.savefig(FIG_SAVE_ADDRESS)