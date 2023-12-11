import utils
from utils import *
import heatmap_utils
from heatmap_utils import *
from utils_models import *
try:
    from config import load_config
except:
    from config import Config
try:
    cfg = Config("config_my.py")
except:
    cfg = load_config("config_my.py")

c_dir = os.getcwd()

from deepexplain.tensorflow import DeepExplain
from utils import preprocess, pixel_range
import tf_slim as slim

checkpoint = 'InceptionModel/inception_v4.ckpt'
def load_images(img_dir, img_size=(IMAGE_WIDTH, IMAGE_HEIGHT)):
    
    def file_number(x):
        return int(x.split('.')[0])

    filenames = sorted(os.listdir(img_dir), key=file_number)
    filenames_valid = []
    images = []

    for file in filenames:
        
        try: 
            image = PIL.Image.open(os.path.join(img_dir, file))
            image = np.array(image.resize(img_size, PIL.Image.ANTIALIAS))
            
            if image.ndim < 3 or (image.ndim == 3 and image.shape[-1] != 3):
                
                raise Exception('Invalid Image')
            
            images.append(image)
            filenames_valid.append(file)
        
        except:
             cprintf(f'exception', 'l_red')
             continue
    
    images = [image for image in images if len(image.shape) == 3]
    images = np.array(images)
    images = images / 127.5 - 1.0
    
    return filenames_valid, images

LABELS_LOC='InceptionModel/imagenet_comp_graph_label_strings.txt'
label_map = np.array(open(LABELS_LOC).read().split('\n'))


### Attribution Map Generation
tf.compat.v1.reset_default_graph()
sess = tf.compat.v1.Session()

with DeepExplain(session=sess, graph=sess.graph) as de:
    PH = tf.compat.v1.placeholder(tf.float32, shape=(None, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS))

    with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
        _, end_points = inception_v4.inception_v4(X, num_classes=1001, is_training=False)

    logits = end_points['Logits']
    yv = tf.reduce_max(input_tensor=logits, axis=1)
    yi = tf.argmax(input=logits, axis=1)

    saver = tf.compat.v1.train.Saver(slim.get_model_variables())
    saver.restore(sess, checkpoint)
    
    IMG_FOLDER_PATH = os.path.join(cfg.TESTING_DATA_DIR, 'track1-sunny-positioned-nominal', 'IMG')
    filenames, imgs = load_images(IMG_FOLDER_PATH)
    labels = sess.run(yi, feed_dict={PH: imgs})

    attribution_methods = [
                           ('RectGrad'         , 'rectgrad'),
                           ('RectGrad PRR'     , 'rectgradprr'),
                           ('Saliency Map'     , 'saliency'),
                           ('Guided BP'        , 'guidedbp'),
                           ('SmoothGrad'       , 'smoothgrad'),
                           ('Gradient * Input' , 'grad*input'),
                           ('IntegGrad'        , 'intgrad'),
                           ('Epsilon-LRP'      , 'elrp'),
                           ('DeepLIFT'         , 'deeplift')
                          ]

    attribution_methods = collections.OrderedDict(attribution_methods)
    
    attributions_orig   = collections.OrderedDict()
    attributions_sparse = collections.OrderedDict()
    
    for k, v in attribution_methods.items():
        
        print('Running {} explanation method'.format(k))
        
        attribution = de.explain(v, yv, X, xs)
        
        if 'RectGrad' in k:
            attributions_orig[k]   = preprocess(attribution, 0.5, 99.5)
            attributions_sparse[k] = preprocess(attribution, 0.5, 99.5)
        else:
            attributions_orig[k]   = preprocess(attribution, 0.5, 99.5)
            attributions_sparse[k] = preprocess(attribution, 95, 99.5)
    
    print('Done!')

    ### Attribution Map Comparison W/O Baseline Final Thresholding

for i, xi in enumerate(xs):
    
    plt.figure(figsize=(13,6))
    
    xi = (xi - np.min(xi))
    xi /= np.max(xi)
    
    plt.subplot(2, 5, 1)
    plt.imshow(xi)
    plt.xticks([])
    plt.yticks([])
    plt.title(label_map[labels[i] - 1].split(',')[0].capitalize(), fontsize=20, pad=10)
    
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
    
    plt.figure(figsize=(13,6))
    
    xi = (xi - np.min(xi))
    xi /= np.max(xi)
    
    plt.subplot(2, 5, 1)
    plt.imshow(xi)
    plt.xticks([])
    plt.yticks([])
    plt.title(label_map[labels[i] - 1].split(',')[0].capitalize(), fontsize=20, pad=10)
    
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
