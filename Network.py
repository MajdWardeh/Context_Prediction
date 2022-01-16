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

    def __init__(self, input_shape=(96, 96, 3)):
        self.input_shape = input_shape
        self.base_model = self._build_base_model()
        self.model = self.__build()
        self.model.summary()
      
    def _build_base_model(self):
        inputLayer = layers.Input((self.input_shape[0], self.input_shape[1], 3, ), name='input') 
        conv1 = layers.Conv2D(96, kernel_size=(11, 11), strides=(4, 4), padding='same', activation='relu', name='conv1')(inputLayer)
        batchNorm1 = layers.BatchNormalization(momentum=0.99, center=False, scale=False)(conv1)
        pool1 = layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same', name='pool1')(batchNorm1)
        LRN1 = tf.nn.local_response_normalization(pool1, depth_radius=5, bias=2.0, alpha=1e-4, beta=0.75, name='LRN1')
        conv2 = layers.Conv2D(384, kernel_size=(5, 5), strides=(1, 1), activation='relu', name='conv2')(LRN1)
        batchNorm2 = layers.BatchNormalization(momentum=0.99, center=False, scale=False)(conv2)
        pool2 = layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same', name='pool2')(batchNorm2)
        LRN2 = tf.nn.local_response_normalization(pool2, depth_radius=5, bias=2.0, alpha=1e-4, beta=0.75, name='LRN2')
        conv3 = layers.Conv2D(384, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='conv3')(LRN2)
        batchNorm3 = layers.BatchNormalization(momentum=0.99, center=False, scale=False)(conv3)
        conv4 = layers.Conv2D(384, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='conv4')(batchNorm3)
        batchNorm4 = layers.BatchNormalization(momentum=0.99, center=False, scale=False)(conv4)
        conv5 = layers.Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='conv5')(batchNorm4)
        batchNorm5 = layers.BatchNormalization(momentum=0.99, center=False, scale=False)(conv5)
        pool5 = layers.MaxPooling2D(pool_size=(3, 3), strides=2, name='pool5')(batchNorm5)
        flatten = layers.Flatten()(pool5)
        fc6 = layers.Dense(4096, activation='relu', name='fc6')(flatten)
        model = Model(inputs=inputLayer, outputs=fc6, name='base_model') 
        # model.summary()
        return model

    def __build(self):

      input_a = layers.Input((self.input_shape[0], self.input_shape[1], 3, ), name='input_a')
      input_b = layers.Input((self.input_shape[0], self.input_shape[1], 3, ), name='input_b')

      output_a = self.base_model(input_a)
      output_b = self.base_model(input_b)

      concationation_layer = layers.concatenate([output_a, output_b])
      fc7 = layers.Dense(4096, activation='relu', name='fc7')(concationation_layer)
      fc8 = layers.Dense(4096, activation='relu', name='fc8')(fc7)
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
    model = Network().getModel()
    model.compile(
                optimizer= Adam(), 
                loss=SparseCategoricalCrossentropy(), 
                metrics=['sparse_categorical_accuracy']
            )
    images_dir = '/media/majd/Data2/ILSVRC2012_img_val/'
    trainGen, testGen = createGenerators(images_dir)
    history = model.fit(x=trainGen,
                        epochs=100,
                        validation_data=testGen,
                        workers=10
                        )
    model.save_weights('./weights/first_model.h5')
    
    
    
    
    

    





if __name__ == "__main__":
    main()