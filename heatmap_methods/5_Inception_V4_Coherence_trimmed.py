import sys
sys.path.append("..")
import utils
from utils import *
import heatmap_utils
from heatmap_utils import *
from utils_models import *
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import os
from pathlib import Path
import collections
from deepexplain.tensorflow import DeepExplain
from heatmap_utils import preprocess_atm, pixel_range
# import tf_slim as slim

try:
    from config import load_config
except:
    from config import Config

os.chdir(os.getcwd().replace('heatmap_methods', ''))
try:
    cfg = Config("config_my.py")
except:
    cfg = load_config("config_my.py")

c_dir = os.getcwd()

def tensor(arg):
  arg = tf.convert_to_tensor(arg, dtype=tf.float32)
  return arg

def load_images(sim_name, img_size=(IMAGE_WIDTH, IMAGE_HEIGHT)):

    # load the image file paths from main csv
    SIM_PATH = os.path.join(cfg.TESTING_DATA_DIR, sim_name)
    MAIN_CSV_PATH = os.path.join(SIM_PATH, "driving_log.csv")
    # IMG_FOLDER_PATH = os.path.join(cfg.TESTING_DATA_DIR, 'track1-sunny-positioned-nominal', 'IMG')
    main_data = pd.read_csv(MAIN_CSV_PATH)
    data = main_data["center"]

    def file_name(x):
        return x.split('.')[0]

    # filenames = sorted(os.listdir(img_dir), key=file_number)
    filenames_valid = []
    images = []
    tensors = []
    for idx, img_adr in enumerate(tqdm(data)):
        # convert Windows path, if needed
        if "\\\\" in img_adr:
            img_adr = img_adr.replace("\\\\", "/")
        elif "\\" in img_adr:
            img_adr = img_adr.replace("\\", "/")

        # load image
        # image = PIL.Image.open(img_adr)
        image = mpimg.imread(img_adr)

        # preprocess image
        image = utils.resize(image).astype('float32')
        images.append(image)
        image = image[np.newaxis is None,:,:,:]
        tensor_img = tensor(image)
        tensors.append(tensor_img)
        filenames_valid.append(file_name(img_adr))
    # # for file in filenames:
    #     try: 
    #         image = PIL.Image.open(os.path.join(img_dir, file))
    #         image = np.array(image.resize(img_size, PIL.Image.ANTIALIAS))
            
    #         if image.ndim < 3 or (image.ndim == 3 and image.shape[-1] != 3):
                
    #             raise Exception('Invalid Image')
            
    #         images.append(image)
    #         filenames_valid.append(file)
        
    #     except:
    #          cprintf(f'exception', 'l_red')
    #          continue
    
    images = [image for image in images if len(image.shape) == 3]
    images = np.array(images)
    images = images / 127.5 - 1.0
    
    return filenames_valid, images, tensors

# LABELS_LOC='InceptionModel/imagenet_comp_graph_label_strings.txt'
# label_map = np.array(open(LABELS_LOC).read().split('\n'))

# Define your preprocess and load_images functions here

# Assuming you have defined your custom model
self_driving_car_model = keras.models.load_model(Path(os.path.join(cfg.SDC_MODELS_DIR, cfg.SDC_MODEL_NAME)))
print(INPUT_SHAPE)

# # # Image loading
# filenames, xs, xs_tensor = load_images('track1-sunny-positioned-nominal')
# cprintf(f'{type(xs)}', 'l_green')
# cprintf(f'{xs.shape}', 'l_green')

# load the image file paths from main csv
SIM_PATH = os.path.join(cfg.TESTING_DATA_DIR, 'track1-sunny-positioned-nominal')
MAIN_CSV_PATH = os.path.join(SIM_PATH, "driving_log.csv")
# IMG_FOLDER_PATH = os.path.join(cfg.TESTING_DATA_DIR, 'track1-sunny-positioned-nominal', 'IMG')
main_data = pd.read_csv(MAIN_CSV_PATH)
data = main_data["center"]

def file_name(x):
    return x.split('.')[0]

# filenames = sorted(os.listdir(img_dir), key=file_number)
filenames_valid = []
images = []
for idx, img_adr in enumerate(tqdm(data)):
    if idx != 0:
        break
    # convert Windows path, if needed
    if "\\\\" in img_adr:
        img_adr = img_adr.replace("\\\\", "/")
    elif "\\" in img_adr:
        img_adr = img_adr.replace("\\", "/")

    # load image
    # image = PIL.Image.open(img_adr)
    xs = mpimg.imread(img_adr)

    # preprocess image
    xs = utils.resize(xs).astype('float32')
    xs = xs[np.newaxis is None,:,:,:]
    cprintf(f'{xs.shape}', 'l_green')
    xs_tensor = tensor(xs)
    cprintf(f'{xs_tensor.shape}', 'l_green')

    # Placeholder for the model input
    # X = tf.keras.backend.placeholder(tf.float32, shape=(None,) + INPUT_SHAPE)
    # X = tf.compat.v1.placeholder(tf.float32, shape=(None,) + INPUT_SHAPE)

    # T = tf.reduce_max(input_tensor=self_driving_car_model(xs_tensor) , axis=1)
    T = self_driving_car_model(xs_tensor)
    # cprintf(f'{T}', 'l_blue')
    # cprintf(f'{T.shape}', 'l_blue')
    # # (ys=self.T, xs=self.X)
    # g = tf.gradients(ys=T, xs=X)
    # cprintf(f'{g}', 'l_magenta')
    # def session_run(session, T, xs):
    #     feed_dict = {}
    #     feed_dict[X] = xs
    #     return session.run(T, feed_dict)

    # cprintf(f'GRADIENTS:', 'l_blue')
    # cprintf(f'{sess.run(g)}', 'l_blue')
    # sess.run(feed_dict={X: xs})
    # raise ValueError('EOC')

    # x = tf.constant(3.0)
    # with tf.GradientTape() as g:
    #     g.watch(x)
    #     y = x * x
    # dy_dx = g.gradient(y, x)
    # print(dy_dx)

    # Use tf.GradientTape to compute gradients
    with tf.GradientTape(persistent=True) as tape:
        # Forward pass
        tape.watch(xs_tensor)
        # y_pred = self_driving_car_model(xs)
        y_pred = self_driving_car_model(xs_tensor)
        # cprintf(f'{(tape.gradient(y_pred, self_driving_car_model.trainable_variables))}', 'l_red')
    gradients = tape.gradient(y_pred, xs_tensor)

    # attribution_methods = [
    #     ('RectGrad', 'rectgrad'),
    #     ('RectGrad PRR', 'rectgradprr'),
    #     ('Saliency Map', 'saliency'),
    #     ('Guided BP', 'guidedbp'),
    #     ('SmoothGrad', 'smoothgrad'),
    #     ('Gradient * Input', 'grad*input'),
    #     ('IntegGrad', 'intgrad'),
    #     ('Epsilon-LRP', 'elrp'),
    #     ('DeepLIFT', 'deeplift')
    # ]

    attribution_methods = [
        ('RectGrad', 'rectgrad'),
        ('RectGrad PRR', 'rectgradprr'),
        ('Saliency Map', 'saliency'),
        ('Guided BP', 'guidedbp'),
        ('SmoothGrad', 'smoothgrad'),
        ('Gradient * Input', 'grad*input'),
        ('IntegGrad', 'intgrad'),
        ('Epsilon-LRP', 'elrp')
    ]

    attribution_methods = collections.OrderedDict(attribution_methods)

    attributions_orig = collections.OrderedDict()
    attributions_sparse = collections.OrderedDict()

    # DeepExplain initialization
    with DeepExplain() as de:
        for k, v in attribution_methods.items():
            cprintf(f'Running {k} explanation method', 'l_cyan')
            
            # Explanation using DeepExplain
            attribution = de.explain(v, T, xs, xs_tensor, y_pred, tape, self_driving_car_model)
            
            # Preprocessing based on the method used
            if 'RectGrad' in k:
                attributions_orig[k] = preprocess_atm(attribution, 0.5, 99.5)
                attributions_sparse[k] = preprocess_atm(attribution, 0.5, 99.5)
            else:
                attributions_orig[k] = preprocess_atm(attribution, 0.5, 99.5)
                attributions_sparse[k] = preprocess_atm(attribution, 95, 99.5)

    print('Done!')

    # print(type(attributions_orig))
    # print(type(attributions_sparse))

    ### Attribution Map Comparison W/O Baseline Final Thresholding

for i, xi in enumerate(xs):
    if i != 0:
        break
    plt.figure(figsize=(13,6))
    
    xi = (xi - np.min(xi))
    xi /= np.max(xi)
    
    plt.subplot(2, 5, 1)
    plt.imshow(xi)
    plt.xticks([])
    plt.yticks([])
    # plt.title(label_map[labels[i] - 1].split(',')[0].capitalize(), fontsize=20, pad=10)
    # 2023_11_07_12_39_56_257_FID_0.jpg
    plt.title('nominal_FID_0.jpg')
    
    for j, a in enumerate(attribution_methods):
        
        plt.subplot(2, 5, j + 2)
        v, cmap = pixel_range(attributions_orig[a][i])
        plt.imshow(attributions_orig[a][i], vmin=v[0], vmax=v[1], cmap=cmap)
        plt.xticks([])
        plt.yticks([])
        plt.title(a, fontsize=20, pad=10)
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=0.2)

    png_folder = os.path.join('png_results', '5')
    png_file = os.path.join(png_folder, f'5_Attribution_Map_NO_Baseline_final_thresholding{i}.png')
    if not os.path.exists(png_folder):
        os.makedirs(png_folder)  
    plt.savefig(png_file, bbox_inches='tight', dpi=300)

    plt.show()    
    
    plt.close()

### Attribution Map Comparison with Baseline Final Thresholding

for i, xi in enumerate(xs):
    if i != 0:
        break    
    plt.figure(figsize=(13,6))
    
    xi = (xi - np.min(xi))
    xi /= np.max(xi)
    
    plt.subplot(2, 5, 1)
    plt.imshow(xi)
    plt.xticks([])
    plt.yticks([])
    # plt.title(label_map[labels[i] - 1].split(',')[0].capitalize(), fontsize=20, pad=10)
    # 2023_11_07_12_39_56_257_FID_0.jpg
    plt.title('nominal_FID_0.jpg')

    for j, a in enumerate(attribution_methods):
        
        plt.subplot(2, 5, j + 2)
        v, cmap = pixel_range(attributions_sparse[a][i])
        plt.imshow(attributions_sparse[a][i], vmin=v[0], vmax=v[1], cmap=cmap)
        plt.xticks([])
        plt.yticks([])
        plt.title(a, fontsize=20, pad=10)
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=0.2)

    results_folder = os.path.join(c_dir, 'png_results')
    png_folder = os.path.join(results_folder, '5')
    png_file = os.path.join(png_folder, f'5_Attribution_Map_WITH_Baseline_final_thresholding{i}.png')
    if not os.path.exists(results_folder):
        os.mkdir(results_folder)
    if not os.path.exists(png_folder):
        os.mkdir(png_folder)  
    plt.savefig(png_file, bbox_inches='tight', dpi=300)

    plt.show()    
    
    plt.close()
