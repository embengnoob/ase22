# import os
# import wget
# import tarfile

# from colorama import Fore, Back
# from colorama import init
# init(autoreset=True)
# import colored_traceback
# colored_traceback.add_hook(always=True)

# from tqdm import tqdm

# import matplotlib.pyplot as plt
# import tensorflow as tf
# tf.compat.v1.disable_v2_behavior()
# import numpy as np
import sys
sys.path.append("..")
from models_tf1 import DAVE2_DROPOUT
from trainer_tf1 import Trainer
from utils import *
from heatmap_utils import *
from config import Config, load_config
from sklearn.utils import shuffle

logdir = 'tf_logs/standard/'

os.chdir(os.getcwd().replace('heatmap_methods', ''))
try:
    cfg = Config("config_my.py")
except:
    cfg = load_config("config_my.py")

x_train, x_test, y_train, y_test = load_data(cfg)
x_train, y_train = shuffle(x_train, y_train, random_state=0)
x_test, y_test = shuffle(x_test, y_test, random_state=0)
tf.compat.v1.reset_default_graph()

DNN = DAVE2_DROPOUT(cfg, logdir, 'dave2_dropout')

sess = tf.compat.v1.InteractiveSession()
sess.run(tf.compat.v1.global_variables_initializer())

trainer = Trainer(sess, cfg, DNN, x_train, y_train, is_training=True)
trainer.train(n_epochs=cfg.NUM_EPOCHS_SDC_MODEL, p_epochs=1)

test_acc = Trainer(sess, cfg, DNN, x_test, y_test, is_training=False)
avg_acc = test_acc.evaluate(sess)
print('Test Accuracy : {:.5f}'.format(avg_acc))

sess.close()