import os
import sys
import math
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np
from typing import List


class Patch:

    def __init__(self, img, xc, yc, id, w=96, h=96, gap_size=48, center=False):
        assert int(xc)==xc, 'xc must be an integer'
        assert int(yc)==yc, 'yc must be an integer'

        self.w = w 
        self.h = h
        self.g = gap_size
        self.id = id
        if center:
            self.id = -1
            self.x = xc
            self.y = yc
        elif id == 0:
            self.x = xc - self.w - self.g
            self.y = yc - self.h - self.g
        elif id == 1:
            self.x = xc
            self.y = yc - self.h - self.g
        elif id == 2:
            self.x = xc + self.w + self.g
            self.y = yc - self.h - self.g
        elif id == 3:
            self.x = xc - self.w - self.g
            self.y = yc 
        elif id == 4:
            self.x = xc + self.w + self.g
            self.y = yc 
        elif id == 5:
            self.x = xc - self.w - self.g
            self.y = yc + self.h + self.g
        elif id == 6:
            self.x = xc
            self.y = yc + self.h + self.g
        elif id == 7:
            self.x = xc + self.w + self.g
            self.y = yc + self.h + self.g
        else:
            raise ValueError('patch id must be in range from 0 to 7')
        self.img = img
        self.img_patch = self.img[self.y-self.h//2 : self.y+self.h//2, self.x-self.w//2 : self.x+self.w//2]
    
    def applyRandomJitter(self, amount=7):
        self.x += np.random.randint(-amount, amount+1)
        self.y += np.random.randint(-amount, amount+1)

    def applyMeanSubtraction(self):
        ## get the patch img from the original image, cast it to float32, and subtract the mean of each color channel.
        self.img_patch = self.img[self.y-self.h//2 : self.y+self.h//2, self.x-self.w//2 : self.x+self.w//2]
        self.img_patch = self.img_patch.astype(np.float32)
        for i in range(3):
            self.img_patch[:, :, i] -= np.mean(self.img_patch[:, :, i])
        return self.img_patch

    def applyColorDropping(self):
        keptColor = np.random.randint(0, 3)
        keptColor_std = np.std(self.img_patch[:, :, keptColor])
        for i in range(3):
            if i == keptColor:
                continue
            self.img_patch[:, :, i] = np.random.normal(loc=0.0, scale=keptColor_std/100, size=(self.h, self.w))
        return self.img_patch
    
    def applyPixelation(self, rate=0.2, downsamplingSizesNum=20):
        if np.random.rand() > rate:
            return self.img_patch

        originalSize = self.h*self.w
        sizes = np.linspace(100, originalSize//2, num=downsamplingSizesNum)
        size = np.random.choice(sizes)
        f = math.sqrt(size/originalSize) 
        img_patch_temp = cv2.resize(self.img_patch, (0, 0), fx=f, fy=f)
        self.img_patch = cv2.resize(img_patch_temp, (self.w, self.h))
        return self.img_patch

    def getImage(self):
        return self.img_patch
    
    ## Debuging functions
    def plotCenter(self, img, color=(255, 0, 0)):
        cv2.circle(img,(self.x, self.y), 5, color, -1)
        cv2.putText(img, str(self.id), (self.x, self.y), cv2.FONT_HERSHEY_SIMPLEX, 1, color)
    
    def plotOnImage(self, img):
        img[self.y-self.h//2 : self.y+self.h//2, self.x-self.w//2 : self.x+self.w//2] = self.img_patch
        

def test_patch_class(img):
    h, w, _ = img.shape
    centerPatch = Patch(img, w//2, h//2, -1, center=True)
    centerPatch.plotCenter(img)
    patchList = []
    for id in range(8):
        patch = Patch(img, xc=w//2, yc=h//2, id=id)

        patch.applyRandomJitter()

        # patch.plotCenter(img)

        patch_img = patch.getImage()
        assert patch_img.shape==(96, 96, 3), 'error in the dimentions of the patch img, found {} expected {}'.format(patch_img.shape, (96, 96, 3))
        cv2.imshow('patch_{}'.format(id), patch_img)

        ## applying Mean Subtraction
        patch_img_zeroMean = patch.applyMeanSubtraction()
        for i in range(3):
            mean = np.mean(patch_img_zeroMean[:, :, i])
            assert np.isclose(mean, 0.0, rtol=0, atol=1e-5), 'the mean of color channel #{} is not close to zero'.format(i)

        cv2.imshow('patch_{}_zeroMean'.format(id), patch_img_zeroMean)

        ## applying color dropping
        patch_img_zeroMean_colorDropped = patch.applyColorDropping()
        stds = [np.std(patch_img_zeroMean_colorDropped[:, :, i]) for i in range(3)]
        stds.sort()
        for i in range(2):
            assert np.isclose(stds[i]*100, stds[2], rtol=0, atol=1), 'large difference between the dropped colors\' stds and the kept color\'s std'

        cv2.imshow('patch_{}_zeroMean_colorDropped'.format(id), patch_img_zeroMean_colorDropped)

        ## applying pixelation
        patch_img_zeroMean_colorDropped_pixeled = patch.applyPixelation(rate=1, downsamplingSizesNum=3)

        cv2.imshow('patch_{}_zeroMean_colorDropped_pixeled'.format(id), patch_img_zeroMean_colorDropped_pixeled)

        patchList.append(patch)
    return patchList 

def plotAllPatches(img, centerPatch: Patch, patchesList: List[Patch]):
    img_out = np.zeros_like(img)
    centerPatch.plotOnImage(img_out)
    for p in patchesList:
        p.plotOnImage(img_out)
    cv2.imshow('allPatches', img_out)
    
def plotLines(img, xc, yc, line_thickness=2):
    '''
        An ugly function used for debugging. It plots the lines of the patchs' layout on an image.
    '''
    h, w, _ = img.shape
    ### ploting vertical lines
    x1 = xc + 48
    cv2.line(img, (x1, 0), (x1, h), (0, 255, 0), thickness=line_thickness)
    x2 = x1 + 48
    cv2.line(img, (x2, 0), (x2, h), (0, 255, 0), thickness=line_thickness)
    x3 = x2 + 96
    cv2.line(img, (x3, 0), (x3, h), (0, 255, 0), thickness=line_thickness)
    x4 = x3 + 7
    cv2.line(img, (x4, 0), (x4, h), (0, 255, 0), thickness=line_thickness)
    x1 = xc - 48
    cv2.line(img, (x1, 0), (x1, h), (0, 255, 0), thickness=line_thickness)
    x2 = x1 - 48
    cv2.line(img, (x2, 0), (x2, h), (0, 255, 0), thickness=line_thickness)
    x3 = x2 - 96
    cv2.line(img, (x3, 0), (x3, h), (0, 255, 0), thickness=line_thickness)
    x4 = x3 - 7
    cv2.line(img, (x4, 0), (x4, h), (0, 255, 0), thickness=line_thickness)
    ### ploting horizontal lines
    y1 = yc + 48
    cv2.line(img, (0, y1), (w, y1), (0, 255, 0), thickness=line_thickness)
    y2 = y1 + 48
    cv2.line(img, (0, y2), (w, y2), (0, 255, 0), thickness=line_thickness)
    y3 = y2 + 96
    cv2.line(img, (0, y3), (w, y3), (0, 255, 0), thickness=line_thickness)
    y4 = y3 + 7
    cv2.line(img, (0, y4), (w, y4), (0, 255, 0), thickness=line_thickness)
    y1 = yc - 48
    cv2.line(img, (0, y1), (w, y1), (0, 255, 0), thickness=line_thickness)
    y2 = y1 - 48
    cv2.line(img, (0, y2), (w, y2), (0, 255, 0), thickness=line_thickness)
    y3 = y2 - 96
    cv2.line(img, (0, y3), (w, y3), (0, 255, 0), thickness=line_thickness)
    y4 = y3 - 7
    cv2.line(img, (0, y4), (w, y4), (0, 255, 0), thickness=line_thickness)

def process_image(img_name):
    img = cv2.imread(img_name)
    '''
        resizing:
            hmin = wmin = (48 + 48 + 96 + 7)*2 = 398 ~= 400
    '''
    hmin = 400
    h, w, _ = img.shape
    if h*w//1000 < 400:
        f = hmin*1.5/min(h, w)
        img_scaled = cv2.resize(img, (0, 0), fx=f, fy=f)
    else:
        img_scaled = img

    patchList = test_patch_class(np.copy(img_scaled))

    plotLines(img_scaled, img_scaled.shape[1]//2, img_scaled.shape[0]//2)

    # cv2.imshow('image', img)
    cv2.imshow('image scaled', img_scaled)
    cv2.waitKey(0)


def main():
    images_dir = './images'
    image_names = [os.path.join(images_dir, I) for I in os.listdir(images_dir)]
    process_image(image_names[1])




    


if __name__ == '__main__':
    main()