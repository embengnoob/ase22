import math
import numpy as np
from utils import *

class Trainer():

    def __init__(self, sess, cfg, model, path_to_pictures, steering_angles, is_training):
        self.sess = sess
        self.cfg = cfg
        self.model = model
        self.batch_size = cfg.BATCH_SIZE
        self.path_to_pictures = path_to_pictures
        self.steering_angles = steering_angles
        self.is_training = is_training
        
    def __getitem__(self, index):
        start_index = index * self.cfg.BATCH_SIZE
        end_index = start_index + self.cfg.BATCH_SIZE
        batch_paths = self.path_to_pictures[start_index:end_index]
        steering_angles = self.steering_angles[start_index:end_index]

        images = np.empty([len(batch_paths), RESIZED_IMAGE_HEIGHT, RESIZED_IMAGE_WIDTH, IMAGE_CHANNELS])
        steers = np.empty([len(batch_paths)])

        for i, paths in enumerate(batch_paths):
            center, left, right = batch_paths[i]
            steering_angle = steering_angles[i]

            # augmentation
            if self.is_training and np.random.rand() < 0.6:
                image, steering_angle = augment(os.path.join(self.cfg.TRAINING_DATA_DIR, self.cfg.TRAINING_SET_DIR),
                                                center, left, right, steering_angle)
            else:
                image = load_image(os.path.join(self.cfg.TRAINING_DATA_DIR, self.cfg.TRAINING_SET_DIR), center)

            # add the image and steering angle to the batch
            images[i] = preprocess(image, False)
            steers[i] = steering_angle

        return images, steers

    def __len__(self):
        return len(self.path_to_pictures) // self.cfg.BATCH_SIZE
    
    # def train(self, n_epochs, p_epochs=10):
        
    #     for epoch in range(n_epochs):
            
    #         train_loss, train_acc = self.train_epoch(epoch)
            
    #         if (epoch + 1) % p_epochs == 0:
                
    #             print('Epoch : {:<3d} | Loss : {:.5f} | Train Accuracy : {:.5f}'.format(epoch + 1, train_loss, train_acc))
        
    #     self.model.save(self.sess)

    def train(self, n_epochs, p_epochs=10):
        for epoch in range(n_epochs):
            train_loss = self.train_epoch(epoch)
            if (epoch + 1) % p_epochs == 0:
                print('Epoch : {:<3d} | Loss : {:.5f}'.format(epoch + 1, train_loss))
        self.model.save(self.sess)
    
    # def train_epoch(self, epoch):

    #     avg_loss = 0
    #     avg_acc = 0
    #     n_itrs = math.ceil(self.__len__())

    #     for itr in range(n_itrs):

    #         loss, acc = self.train_step(itr)
    #         avg_loss += loss / n_itrs
    #         avg_acc += acc / n_itrs
        
    #     return avg_loss, avg_acc
    
    def train_epoch(self, epoch):
        avg_loss = 0
        n_itrs = math.ceil(self.__len__())
        for itr in range(n_itrs):
            loss = self.train_step(itr)
            avg_loss += loss / n_itrs
        return avg_loss

    # def train_step(self, itr):

    #     # batch_xs, batch_ys = self.data_train[0][itr * self.batch_size:(itr + 1) * self.batch_size], self.data_train[1][itr * self.batch_size:(itr + 1) * self.batch_size]

    #     batch_xs, batch_ys = self.__getitem__(itr+1)
    #     cprintf(f'{batch_xs.shape}', 'l_green')
    #     cprintf(f'{batch_ys.shape}', 'l_blue')
    #     feed_dict = {self.model.X: batch_xs, self.model.Y: batch_ys}
    #     _, loss, acc = self.sess.run([self.model.train, self.model.loss, self.model.accuracy], feed_dict=feed_dict)
    #     cprintf(f'{(loss, acc)}', 'l_red')
    #     return loss, acc
    
    def train_step(self, itr):
        batch_xs, batch_ys = self.__getitem__(itr + 1)
        feed_dict = {self.model.X: batch_xs, self.model.target: batch_ys}
        _, loss = self.sess.run([self.model.train, self.model.loss], feed_dict=feed_dict)
        return loss

    
    # def evaluate(self, sess):
    #     n_itrs = math.ceil(self.__len__())
    #     avg_acc = 0
    #     for itr in range(n_itrs):
    #         batch_xs, batch_ys = self.__getitem__(itr+1)
    #         feed_dict = {self.X: batch_xs, self.Y: batch_ys}
    #         acc = sess.run(self.accuracy, feed_dict=feed_dict)
    #         avg_acc += acc / n_itrs
    #     return avg_acc
    
    def evaluate(self):
        n_itrs = math.ceil(self.__len__())
        avg_loss = 0

        for itr in range(n_itrs):
            batch_xs, batch_ys = self.__getitem__(itr + 1)
            feed_dict = {self.model.X: batch_xs, self.model.target: batch_ys}
            loss = self.sess.run(self.model.loss, feed_dict=feed_dict)
            avg_loss += loss / n_itrs

        return avg_loss
