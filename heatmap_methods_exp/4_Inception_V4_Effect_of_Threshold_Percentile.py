
import utils
from utils import *

c_dir = os.getcwd()
if not os.path.exists('CIFAR10_data'):
    os.mkdir('CIFAR10_data')  
    tar_file_path = os.path.join(c_dir, 'cifar-10-python.tar.gz')
    if not os.path.exists(tar_file_path):
        url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
        wget.download(url)
    extract_to = os.path.join(c_dir, 'CIFAR10_data')
    extract_all_files(tar_file_path, extract_to)

if not os.path.exists('ILSVRC_data'):
    os.mkdir('ILSVRC_data')
    tar_file_path = os.path.join(c_dir, 'ILSVRC_data.tar')
    extract_to = os.path.join(c_dir, 'ILSVRC_data')
    extract_all_files(tar_file_path, extract_to)

if not os.path.exists('models'):
    git.Git(c_dir).clone("https://github.com/tensorflow/models.git")

if not os.path.exists('InceptionModel/inception_v4.ckpt'):
    tar_file_path = os.path.join(c_dir, 'inception_v4_2016_09_09.tar.gz')
    if not os.path.exists(tar_file_path):
        url = 'http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz'
        wget.download(url)
    tar_file_path = os.path.join(c_dir, 'inception_v4_2016_09_09.tar.gz')
    extract_to = os.path.join(c_dir, 'InceptionModel')
    extract_all_files(tar_file_path, extract_to)

sys.path.append(os.path.abspath('models/research/slim'))

from nets import inception_v4

from deepexplain.tensorflow import DeepExplain
from utils import preprocess, pixel_range

import tf_slim as slim

checkpoint = 'InceptionModel/inception_v4.ckpt'


def load_images(img_dir, img_size=(299,299)):
    
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
    
    X = tf.compat.v1.placeholder(tf.float32, shape=(None, 299, 299, 3))

    with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
        _, end_points = inception_v4.inception_v4(X, num_classes=1001, is_training=False)

    logits = end_points['Logits']
    yv = tf.reduce_max(input_tensor=logits, axis=1)
    yi = tf.argmax(input=logits, axis=1)

    saver = tf.compat.v1.train.Saver(slim.get_model_variables())
    saver.restore(sess, checkpoint)
    
    filenames, xs = load_images(os.path.join('ILSVRC_data', 'ILSVRC_data', 'threshold'))

    labels = sess.run(yi, feed_dict={X: xs})
    
    attribution_methods = [
                           (r'$\tau = 0$', 'rectgradconst'),
                           (r'$q = 0$'   , 0),
                           (r'$q = 10$'  , 10),
                           (r'$q = 20$'  , 20),
                           (r'$q = 80$'  , 80),
                           (r'$q = 90$'  , 90),
                           (r'$q = 95$'  , 95),
                           (r'$q = 99$'  , 99)
                          ]

    attribution_methods = collections.OrderedDict(attribution_methods)
    
    attributions = collections.OrderedDict()
    
    for k, v in attribution_methods.items():
        
        print('Running {} explanation method'.format(k))
        
        if v == 'rectgradconst':
            attribution = de.explain(v, yv, X, xs, tau=0)
        else:
            attribution = de.explain('rectgrad', yv, X, xs, percentile=v)
        
        attributions[k] = preprocess(attribution, 0.5, 99.5)
    
    print('Done!')

    ### Attribution Map Visualization

for i, xi in enumerate(xs):
    
    plt.figure(figsize=(13,6))
    
    xi = (xi - np.min(xi))
    xi /= np.max(xi)
    
    plt.subplot(1, 9, 1)
    plt.imshow(xi)
    plt.xticks([])
    plt.yticks([])
    
    if i == 0:
        plt.title('Image', fontsize=15, pad=10)
    
    for j, (method, attr) in enumerate(attributions.items()):
        
        plt.subplot(1, 9, j + 2)
        v, cmap = pixel_range(attr)
        plt.imshow(attr[i], vmin=v[0], vmax=v[1], cmap=cmap)
        plt.xticks([])
        plt.yticks([])
        
        if i == 0:
            plt.title(method, fontsize=15, pad=10)
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=0.2)

    png_folder = os.path.join(c_dir, 'png_results', '4')
    png_file = os.path.join(png_folder, f'4_Attribution_Map_Visualization_{i}.png')
    if not os.path.exists(png_folder):
        os.mkdir(png_folder)  
    plt.savefig(png_file, bbox_inches='tight', dpi=300)
    plt.show()