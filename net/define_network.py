import numpy as np
import keras
import tensorflow as tf
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from net import Unet
from net import Resnet

class My_net(object):
    def __init__(self,
                 batch_size,
                 image_height,
                 image_width,
                 image_channel,
                 data_class,
                 unsupervised_loss_weight=None,
                 supervised_loss_weight=None
                 ):

        self.batch_size = batch_size
        self.image_height = image_height
        self.image_width = image_width
        self.image_channel = image_channel
        self.data_class = data_class
        self.unsupervised_loss_weight = unsupervised_loss_weight
        self.supervised_loss_weight = supervised_loss_weight

        self.supervised = np.zeros((batch_size, image_height, image_width, image_channel))
        self.unsupervised = np.zeros((batch_size, image_height, image_width, image_channel))
        self.I = np.zeros((batch_size, image_height, image_width, image_channel))
        self.I0 = np.zeros((batch_size, image_height, image_width, image_channel))
        self.ground_truth = np.zeros((batch_size, image_height, image_width, image_channel))

    def network(self):
        # define placeholder
        self.supervised = Input(
            shape=(self.image_height, self.image_width, self.image_channel),
            dtype='float32',
            name='supervised'
        )
        self.unsupervised = Input(
            shape=(self.image_height, self.image_width, self.image_channel),
            dtype='float32',
            name='unsupervised'
        )
        self.I = Input(
            shape=(self.image_height, self.image_width, self.image_channel),
            dtype='float32',
            name='I'
        )
        self.I0 = Input(
            shape=(self.image_height, self.image_width, self.image_channel),
            dtype='float32',
            name='I0'
        )
        self.ground_truth = Input(
            shape=(self.image_height, self.image_width, self.image_channel),
            dtype='float32',
            name='ground_truth'
        )

        if self.data_class == 'unsupervised':
            y = self.my_net(input=self.unsupervised, img_channel=self.image_channel)
            loss = Lambda(function=self.Lambda_unsupervised_loss)(y)
            self.model = Model(
                inputs=[self.unsupervised, self.I, self.I0],
                outputs=loss
            )
            adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
            self.model.compile(optimizer=adam, loss=self.my_loss)
        elif self.data_class == 'semisupervised':
            unsupervised_y = self.my_net(input=self.unsupervised, img_channel=self.image_channel)
            supervised_y = self.my_net(input=self.supervised, img_channel=self.image_channel)
            unsupervised_loss = Lambda(function=self.Lambda_unsupervised_loss)(unsupervised_y)
            supervised_loss = Lambda(function=self.Lambda_supervised_loss)(supervised_y)
            self.model = Model(
                inputs=[self.unsupervised, self.supervised, self.I, self.I0, self.ground_truth],
                outputs=[unsupervised_loss, supervised_loss]
            )
            adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
            self.model.compile(optimizer=adam,
                               loss=self.my_loss,
                               loss_weights=[self.unsupervised_loss_weight, self.supervised_loss_weight])
        else:
            y = self.my_net(input=self.supervised, img_channel=self.image_channel)
            loss = Lambda(function=self.Lambda_supervised_loss)(y)
            self.model = Model(
                inputs=[self.supervised, self.ground_truth],
                outputs=loss
            )
            adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
            self.model.compile(optimizer=adam, loss=self.my_loss)
    
    def log(self, input, x):
        numerator = K.log(input)
        denominator = K.log(tf.constant(x, dtype=numerator.dtype))
        return numerator / denominator

    def my_loss(self, y_true, y_pred):
        return y_pred

    def Lambda_unsupervised_loss(self, y):
        # 行的diff
        y_r0 = K.concatenate([(y)[:, :, :, :],
                              K.reshape((y)[:, 0, :, :], [self.batch_size, 1, self.image_width, 1])], axis=1)
        y_r1 = K.concatenate([K.reshape((y)[:, self.image_height - 1, :, :], [self.batch_size, 1, self.image_width, 1]),
                              (y)[:, :, :, :]],axis=1)
        y_r = (y_r0 - y_r1)[:, 1:, :, :]
        # 列的diff
        y_l0 = K.concatenate([(y)[:, :, :, :],
                              K.reshape((y)[:, :, 1, :], [self.batch_size, self.image_height, 1, 1])], axis=2)
        y_l1 = K.concatenate([K.reshape((y)[:, :, 1, :], [self.batch_size, self.image_height, 1, 1]),
                              (y)[:, :, :, :]], axis=2)
        y_l = (y_l0 - y_l1)[:, :, 1:, :]

        # 行diff然后行diff
        y_rr0 = K.concatenate([y_r[:, :, :, :],
                               K.reshape(y_r[:, 0, :, :], [self.batch_size, 1, self.image_width, 1])], axis=1)
        y_rr1 = K.concatenate([K.reshape(y_r[:, self.image_height - 1, :, :], [self.batch_size, 1, self.image_width, 1]),
                               y_r[:, :, :, :]],axis=1)
        y_rr = (y_rr0 - y_rr1)[:, 1:, :, :]
        # 行diff然后列diff
        y_rl0 = K.concatenate([y_r[:, :, :, :],
                               K.reshape(y_r[:, :, 1, :], [self.batch_size, self.image_height, 1, 1])], axis=2)
        y_rl1 = K.concatenate([K.reshape(y_r[:, :, 1, :], [self.batch_size, self.image_height, 1, 1]),
                               y_r[:, :, :, :]], axis=2)
        y_rl = (y_rl0 - y_rl1)[:, :, 1:, :]
        # 列diff然后行diff
        y_lr0 = K.concatenate([y_l[:, :, :, :],
                               K.reshape(y_l[:, 0, :, :], [self.batch_size, 1, self.image_width, 1])], axis=1)
        y_lr1 = K.concatenate([K.reshape(y_l[:, self.image_height - 1, :, :], [self.batch_size, 1, self.image_width, 1]),
                               y_l[:, :, :, :]],axis=1)
        y_lr = (y_lr0 - y_lr1)[:, 1:, :, :]
        # 列diff然后列diff
        y_ll0 = K.concatenate([y_l[:, :, :, :],
                               K.reshape(y_l[:, :, 1, :], [self.batch_size, self.image_height, 1, 1])], axis=2)
        y_ll1 = K.concatenate([K.reshape(y_l[:, :, 1, :], [self.batch_size, self.image_height, 1, 1]),
                               y_l[:, :, :, :]], axis=2)
        y_ll = (y_ll0 - y_ll1)[:, :, 1:, :]

        diff_y = K.concatenate(
            [self.log(K.abs(K.reshape(y_rr, [self.batch_size, self.image_height * self.image_width])) + 1e-6, 100) -
             self.log(1e-6, 100),
             (self.log(K.abs(K.reshape(y_rl, [self.batch_size, self.image_height * self.image_width])) + 1e-6, 100) -
              self.log(1e-6, 100)) * 0.5,
             (self.log(K.abs(K.reshape(y_lr, [self.batch_size, self.image_height * self.image_width])) + 1e-6, 100) -
              self.log(1e-6, 100)) * 0.5,
             self.log(K.abs(K.reshape(y_ll, [self.batch_size, self.image_height * self.image_width])) + 1e-6, 100) -
             self.log(1e-6, 100)],
            axis=0)

        unsupervised_loss = K.sum(self.I * (y) + self.I0 * K.exp(-(y))) + 0.008 * K.sum(diff_y)
        return unsupervised_loss

    def Lambda_supervised_loss(self, y):
        supervised_loss = tf.losses.absolute_difference(labels=self.ground_truth, predictions=y)
        return supervised_loss

    def my_net(self, input, img_channel):
        return Unet(input, img_channel)






    
    

