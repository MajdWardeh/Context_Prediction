import os
import random
import numpy as np
from numpy.core.defchararray import center
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from PatchPairGenerator import PatchPairGenerator

class Network:

    def __init__(self, input_shape=(96, 96, 3), base_model='default'):
        self.input_shape = input_shape
        if base_model == 'default':
            self.base_model = self._build_default_base_model()
        elif base_model == 'batchNorm':
            self.base_model = self._build_batchNorm_base_model()
        else:
            raise NotImplementedError
    
        self.model = self.__build()
        self.model.summary()
      
    def _build_default_base_model(self):
        inputLayer = layers.Input((self.input_shape[0], self.input_shape[1], 3, ), name='input') 
        conv1 = layers.Conv2D(96, kernel_size=(11, 11), strides=1, padding='same', name='conv1', activation='relu')(inputLayer)
        pool1 = layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same', name='pool1')(conv1)
        LRN1 = tf.nn.local_response_normalization(pool1, name='LRN1')
        # LRN1 = pool1
        conv2 = layers.Conv2D(384, kernel_size=(5, 5), strides=2, padding='same', name='conv2', activation='relu')(LRN1)
        pool2 = layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same', name='pool2')(conv2)
        LRN2 = tf.nn.local_response_normalization(pool2, name='LRN2')
        # LRN2 = pool2
        conv3 = layers.Conv2D(384, kernel_size=(3, 3), strides=1, name='conv3', activation='relu')(LRN2)
        conv4 = layers.Conv2D(384, kernel_size=(3, 3), strides=1, activation='relu', name='conv4')(conv3)
        conv5 = layers.Conv2D(256, kernel_size=(3, 3), strides=1, activation='relu', name='conv5')(conv4)
        pool5 = layers.MaxPooling2D(pool_size=(3, 3), strides=2, name='pool5')(conv5)
        flatten = layers.Flatten()(pool5)
        fc6 = layers.Dense(4096, activation='relu', name='fc6')(flatten)
        return Model(inputs=inputLayer, outputs=fc6, name='base_model') 

    def _build_batchNorm_base_model(self):
        inputLayer = layers.Input((self.input_shape[0], self.input_shape[1], 3, ), name='input') 
        conv1 = layers.Conv2D(96, kernel_size=(11, 11), strides=1, padding='same', name='conv1', activation='relu')(inputLayer)
        pool1 = layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same', name='pool1')(conv1)
        LRN1 = tf.nn.local_response_normalization(pool1, name='LRN1')
        conv2 = layers.Conv2D(384, kernel_size=(5, 5), strides=2, padding='same', name='conv2', activation='relu')(LRN1)
        pool2 = layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same', name='pool2')(conv2)
        LRN2 = tf.nn.local_response_normalization(pool2, name='LRN2')
        conv3 = layers.Conv2D(384, kernel_size=(3, 3), strides=1, name='conv3', activation='relu')(LRN2)
        conv4 = layers.Conv2D(384, kernel_size=(3, 3), strides=1, activation='relu', name='conv4')(conv3)
        conv5 = layers.Conv2D(256, kernel_size=(3, 3), strides=1, activation='relu', name='conv5')(conv4)
        pool5 = layers.MaxPooling2D(pool_size=(3, 3), strides=2, name='pool5')(conv5)
        # batchNorm1 = layers.BatchNormalization(momentum=0.99, center=False, scale=False)(pool5)
        batchNorm1 = pool5
        flatten = layers.Flatten()(batchNorm1)
        fc6 = layers.Dense(4096, activation='relu', name='fc6')(flatten)
        return Model(inputs=inputLayer, outputs=fc6, name='base_model') 
    

    def __build(self):

      input_a = layers.Input((self.input_shape[0], self.input_shape[1], 3, ), name='input_a')
      input_b = layers.Input((self.input_shape[0], self.input_shape[1], 3, ), name='input_b')

      output_a = self.base_model(input_a)
      output_b = self.base_model(input_b)

      concationation_layer = layers.concatenate([output_a, output_b])
      fc7 = layers.Dense(4096, activation='relu', name='fc7')(concationation_layer)
      batchNorm2 = layers.BatchNormalization(momentum=0.999, center=False, scale=False)(fc7)
      fc8 = layers.Dense(4096, activation='relu', name='fc8')(batchNorm2)
      fc9 = layers.Dense(8, activation='softmax', name='fc9')(fc8)

      return Model(inputs=[input_a, input_b], outputs=[fc9])
    
    def getModel(self):
      return self.model
        
def createGenerators(images_dir, trainRatio=0.8):
    ## set seeds for consistancy
    random.seed = 0
    np.random.seed = 0

    images_path = [os.path.join(images_dir, img) for img in os.listdir(images_dir)]
    # images_path = images_path[:1000]
    random.shuffle(images_path)

    train_images_len = int(len(images_path)*trainRatio)
    train_images = images_path[:train_images_len]
    test_images = images_path[train_images_len:]
    
    train_batch_size = 20*8
    trainGen = PatchPairGenerator(train_images, batch_size=train_batch_size)

    test_batch_size = 20*8
    testGen = PatchPairGenerator(test_images, batch_size=test_batch_size)
    return trainGen, testGen


def main():
    model = Network(base_model='batchNorm').getModel()
    model.compile(
                optimizer= Adam(), 
                loss=SparseCategoricalCrossentropy(), 
                metrics=['sparse_categorical_accuracy']
            )
    images_dir = '/media/majd/Data2/ILSVRC2012_img_val/'
    trainGen, testGen = createGenerators(images_dir)
    # model.load_weights('./weights/first_model.h5')
    history = model.fit(x=trainGen,
                        epochs=100,
                        validation_data=testGen,
                        workers=10
                        # use_multiprocessing=True
                        )
    model.save_weights('./weights/first_model.h5')
    
    
    
    
    

    





if __name__ == "__main__":
    main()