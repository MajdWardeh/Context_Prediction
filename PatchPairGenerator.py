from logging import debug
import os
import sys
import math
import random

from numpy.core.defchararray import center
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.utils import Sequence

from Patch import Patch, plotLines, plotAllPatches



class PatchPairGenerator(Sequence):
    def __init__(self, image_set, batch_size, random_state=0, pixelation_rate=0.2, downsamplingSizesNum=5, debugging=False):
        self.img_set = image_set
        assert batch_size % 8 == 0, 'the batch_size must be a multiple of 8'
        self.batch_size = batch_size//8
        self.random_state = random_state
        self.pixelation_rate = pixelation_rate
        self.downsamplingSizesNum = downsamplingSizesNum
        self.debugging = debugging

    def __len__(self):
        return math.ceil(len(self.img_set) / self.batch_size)

    def __getitem__(self, index):
        centerPatch_batch = []
        neighbourPatch_batch = []
        label_batch = []

        for row in range(min(self.batch_size, len(self.img_set)-index*self.batch_size)):
            img = cv2.imread(self.img_set[index*self.batch_size + row])
            if img is None:
                continue
            '''
                resizing:
                    hmin = wmin = (48 + 48 + 96 + 7)*2 = 398 ~= 400
            '''
            hmin = 400
            h, w, _ = img.shape
            ## check if the size of the image is less that 400k pixels or its min dimention is less than hmin (or wmin)
            if h*w//1000 < 400 or min(h, w) < hmin:
                f = hmin*1.5/min(h, w)
                img_scaled = cv2.resize(img, (0, 0), fx=f, fy=f)
            else:
                img_scaled = img
            assert min(img_scaled.shape[0], img_scaled.shape[1]) >= hmin

            img_scaled = img_scaled.astype(np.float32)/255.0

            if self.debugging:
                print('-------------------')
                print('image:', self.img_set[index*self.batch_size + row])
                print('shape:', img.shape, 'resized shape:', img_scaled.shape, 'scaled size:', img_scaled.shape[0]*img_scaled.shape[1]//1000)
                # print('img_scaled type:', img_scaled.dtype)
            

            centerPatch, neighbourPatchs = self.generate_patches(img_scaled)

            if self.debugging:
                img_scaled_copy = img_scaled.copy()
                img_scaled_copy = img_scaled_copy.astype(np.float32)
                plotAllPatches(img_scaled_copy, centerPatch, neighbourPatchs)
                # plotLines(img_scaled_copy, img_scaled.shape[1]//2, img_scaled.shape[0]//2)
                # cv2.imshow('scaled copy', img_scaled_copy)

            ## adding patches to the batches lists
            for i in range(8):
                centerPatch_batch.append(centerPatch.getImage())
                neighbourPatch_batch.append(neighbourPatchs[i].getImage())
                label_batch.append(neighbourPatchs[i].id)
            
        ## shuffling the lists
        centerPatch_batch, neighbourPatch_batch, label_batch = shuffle(centerPatch_batch, neighbourPatch_batch, label_batch)

        centerPatch_batch = np.array(centerPatch_batch)
        neighbourPatch_batch = np.array(neighbourPatch_batch)
        label_batch = np.array(label_batch)

        return ([centerPatch_batch, neighbourPatch_batch], label_batch)
    
    def generate_patches(self, img):
        h, w, _ = img.shape

        patchList = []
        for id in range(8):
            patch = Patch(img, xc=w//2, yc=h//2, id=id)
            if self.debugging:
                patch.plotCenter(img)
            
            patch.applyRandomJitter()

            ## applying Mean Subtraction
            patch.applyMeanSubtraction()

            ## applying color dropping
            # patch.applyColorDropping()

            ## applying pixelation
            # patch.applyPixelation(rate=self.pixelation_rate, downsamplingSizesNum=self.downsamplingSizesNum)

            patchList.append(patch)

        ## create a center patch that we do not jitter
        centerPatch = Patch(img, w//2, h//2, -1, center=True)
        if self.debugging:
            centerPatch.plotCenter(img)
        centerPatch.applyMeanSubtraction()
        # centerPatch.applyColorDropping()
        # centerPatch.applyPixelation(rate=self.pixelation_rate, downsamplingSizesNum=self.downsamplingSizesNum)

        return centerPatch, patchList 



def main():
    images_dir = '/media/majd/Data2/ILSVRC2012_img_val/'
    images_path = [os.path.join(images_dir, img) for img in os.listdir(images_dir)]
    random.shuffle(images_path)
    batch_size = 1*8
    generator = PatchPairGenerator(images_path, batch_size=batch_size, debugging=False)
    finish = False
    for idx in range(generator.__len__()):
        X, y = generator.__getitem__(idx)
        for i in range(batch_size):
            centerPatchs, neighbourPatchs = X
            print('centerPatchs[i].dtype: ', centerPatchs[i].dtype)
            img = np.zeros((96, int(96*2.5), 3), dtype=np.float32)
            img[:, :96] = centerPatchs[i]
            img[:, -96:] = neighbourPatchs[i]
            cv2.imshow('sample', img)
            print('img.dtype: ', img.dtype)
            print('label', y[i])
            key = cv2.waitKey(0)
            if key & 0xff == ord('q'):
                finish = True
                break
        if finish:
            break



if __name__ == '__main__':
    main()