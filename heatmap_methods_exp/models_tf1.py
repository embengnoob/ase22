import os, math
from utils import *
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2

class DAVE2_DROPOUT:
    
    def __init__(self, cfg, logdir, name):
        
        if not os.path.exists(logdir):
            
            os.makedirs(logdir)
        
        self.cfg = cfg
        self.logdir = logdir
        self.name = name
        
        self.build_model2()
        self.init_saver()
    
    def save(self, sess):

        self.saver.save(sess, self.logdir + 'model')

    def load(self, sess):

        latest_checkpoint = tf.train.latest_checkpoint(self.logdir)

        if latest_checkpoint:

            self.saver.restore(sess, latest_checkpoint)
    
    def evaluate(self, sess, dataset):

        batch_size = 100
        n_itrs = math.ceil(len(dataset[0]) / batch_size)
        avg_acc = 0

        for itr in range(n_itrs):

            batch_xs, batch_ys = dataset[0][itr * batch_size:(itr + 1) * batch_size], dataset[1][itr * batch_size:(itr + 1) * batch_size]

            feed_dict = {self.X: batch_xs, self.Y: batch_ys}
            acc = sess.run(self.accuracy, feed_dict=feed_dict)
            avg_acc += acc / n_itrs

        return avg_acc
    
    def build_model(self):
        tf.compat.v1.disable_v2_behavior()
        self.X = tf.compat.v1.placeholder(tf.float32, [None, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS], 'X')
        self.Y = tf.compat.v1.placeholder(tf.int64, [None], 'Y')
        Y_hot = tf.one_hot(self.Y, depth=10)
        
        with tf.compat.v1.variable_scope(self.name):
            
            conv1 = tf.compat.v1.layers.conv2d(inputs=self.X, filters=32, kernel_size=[3,3], padding='SAME', activation=tf.nn.relu, use_bias=True, name='conv1')
            conv2 = tf.compat.v1.layers.conv2d(inputs=conv1, filters=32, kernel_size=[3,3], padding='SAME', activation=tf.nn.relu, use_bias=True, name='conv2')
            pool2 = tf.compat.v1.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], strides=[2,2], padding='SAME', name='pool2')
            
            conv3 = tf.compat.v1.layers.conv2d(inputs=pool2, filters=64, kernel_size=[3,3], padding='SAME', activation=tf.nn.relu, use_bias=True, name='conv3')
            conv4 = tf.compat.v1.layers.conv2d(inputs=conv3, filters=64, kernel_size=[3,3], padding='SAME', activation=tf.nn.relu, use_bias=True, name='conv4')
            pool4 = tf.compat.v1.layers.max_pooling2d(inputs=conv4, pool_size=[2,2], strides=[2,2], padding='SAME', name='pool4')
            flat4 = tf.reshape(pool4, [-1, 8 * 8 * 64], name='flat4')
            
            dense5 = tf.compat.v1.layers.dense(inputs=flat4, units=256, activation=tf.nn.relu, use_bias=True, name='dense5')
            self.logits = tf.compat.v1.layers.dense(inputs=dense5, units=10, use_bias=True, name='dense6')
        
        tf.compat.v1.add_to_collection('tensors', self.X)
        tf.compat.v1.add_to_collection('tensors', self.logits)
        
        predictions = tf.argmax(input=self.logits, axis=1)
        correct_predictions = tf.equal(predictions, tf.argmax(input=Y_hot, axis=1))
        self.accuracy = tf.reduce_mean(input_tensor=tf.cast(correct_predictions, tf.float32), name='Accuracy')
        
        self.yi = tf.argmax(input=self.logits, axis=1, name='Prediction')
        self.yx = tf.nn.softmax(self.logits, name='Scores')
        self.yv = tf.reduce_max(input_tensor=self.logits, axis=1, name='MaxScore')
        
        self.loss = tf.reduce_mean(input_tensor=tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=Y_hot))
        self.train = tf.compat.v1.train.AdamOptimizer().minimize(self.loss, var_list=self.vars)

    def build_model2(self):
        RESIZED_IMAGE_HEIGHT, RESIZED_IMAGE_WIDTH = 80, 160
        IMAGE_CHANNELS = 3
        INPUT_SHAPE = (RESIZED_IMAGE_HEIGHT, RESIZED_IMAGE_WIDTH, IMAGE_CHANNELS)

        # Define placeholders
        self.X = tf.compat.v1.placeholder(tf.float32, shape=(None,) + INPUT_SHAPE, name="input")
        self.Y = tf.compat.v1.placeholder(tf.float32, [None], 'Y')
        # Y_hot = tf.one_hot(tf.cast(self.Y, tf.int32), depth=1)

        with tf.compat.v1.variable_scope(self.name):
            # Lambda layer
            lambda_layer = tf.keras.layers.Lambda(lambda x: x / 127.5 - 1.0, name="lambda_layer")(self.X)
            # Convolutional layers
            x = tf.compat.v1.layers.conv2d(inputs=lambda_layer, filters=24, kernel_size=(5, 5), activation=tf.nn.relu, strides=(2, 2), kernel_regularizer=tf.keras.regularizers.l2(1.0e-6))
            x = tf.compat.v1.layers.dropout(x, rate=0.05, training=True)
            x = tf.compat.v1.layers.conv2d(inputs=x, filters=36, kernel_size=(5, 5), activation=tf.nn.relu, strides=(2, 2), kernel_regularizer=tf.keras.regularizers.l2(1.0e-6))
            x = tf.compat.v1.layers.dropout(x, rate=0.05, training=True)
            x = tf.compat.v1.layers.conv2d(inputs=x, filters=48, kernel_size=(5, 5), activation=tf.nn.relu, strides=(2, 2), kernel_regularizer=tf.keras.regularizers.l2(1.0e-6))
            x = tf.compat.v1.layers.dropout(x, rate=0.05, training=True)
            x = tf.compat.v1.layers.conv2d(inputs=x, filters=64, kernel_size=(3, 3), activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(1.0e-6))
            x = tf.compat.v1.layers.dropout(x, rate=0.05, training=True)
            x = tf.compat.v1.layers.conv2d(inputs=x, filters=64, kernel_size=(3, 3), activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(1.0e-6))
            x = tf.compat.v1.layers.dropout(x, rate=0.05, training=True)
            # Flatten layer
            x = tf.compat.v1.layers.flatten(x)
            # Fully connected layers
            x = tf.compat.v1.layers.dense(inputs=x, units=100, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(1.0e-6))
            x = tf.compat.v1.layers.dropout(x, rate=0.05, training=True)
            x = tf.compat.v1.layers.dense(inputs=x, units=50, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(1.0e-6))
            x = tf.compat.v1.layers.dropout(x, rate=0.05, training=True)
            x = tf.compat.v1.layers.dense(inputs=x, units=10, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(1.0e-6))
            x = tf.compat.v1.layers.dropout(x, rate=0.05, training=True)
            # Output layer
            self.logits = tf.compat.v1.layers.dense(inputs=x, units=1)
        
        tf.compat.v1.add_to_collection('tensors', self.X)
        tf.compat.v1.add_to_collection('tensors', self.logits)

        # No need for argmax in regression
        self.yi = tf.identity(self.logits, name='Prediction')
        self.yx = self.logits  # No need for softmax in regression
        self.yv = self.logits  # Use the raw logits as the max score

        # Define target placeholder for regression
        self.target = tf.compat.v1.placeholder(tf.float32, [None], name="Target")

        # Mean Squared Error (MSE) as the loss for regression
        self.loss = tf.reduce_mean(tf.square(self.logits - self.target), name='MSE')

        # Define optimizer and training operation
        self.train = tf.compat.v1.train.AdamOptimizer(learning_rate=self.cfg.LEARNING_RATE).minimize(self.loss, var_list=self.vars)

        # Accuracy for regression is not applicable
        self.accuracy = None

        # tf.compat.v1.add_to_collection('tensors', self.X)
        # tf.compat.v1.add_to_collection('tensors', self.logits)
        
        # predictions = tf.argmax(input=self.logits, axis=1)
        # cprintf(f'{predictions}', 'l_magenta')
        # correct_predictions = tf.equal(predictions, tf.argmax(input=self.Y, axis=1))
        # self.accuracy = tf.reduce_mean(input_tensor=tf.cast(correct_predictions, tf.float32), name='Accuracy')
        
        # self.yi = tf.argmax(input=self.logits, axis=1, name='Prediction')
        # self.yx = tf.nn.softmax(self.logits, name='Scores')
        # self.yv = tf.reduce_max(input_tensor=self.logits, axis=1, name='MaxScore')
        
        # self.loss = tf.reduce_mean(input_tensor=tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
        # self.train = tf.compat.v1.train.AdamOptimizer().minimize(self.loss, var_list=self.vars)


    def init_saver(self):
        
        self.saver = tf.compat.v1.train.Saver()
    
    @property
    def vars(self):
        
        return tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)