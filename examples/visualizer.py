import numpy as np
import os

import cv2


class NpyVisuallizer(object):
    def __init__(self, path) -> None:
        self._path = path
        self._imagesArray = np.load(self._path)
        print(self._imagesArray.shape)
        self._nums = self._imagesArray.shape[0]
        self._channels = self._imagesArray.shape[1]
        self._width = self._imagesArray.shape[2]
        self._height = self._imagesArray.shape[3]
        

    def showSingle(self, index, waitKey = 0) -> None:
        img_array = self._imagesArray[index]
        r = img_array[0]
        g = img_array[1]
        b = img_array[2]
        alpha = img_array[3]
        
        img = np.empty([64,64,4])
        img[:,:,0] = r
        img[:,:,1] = g
        img[:,:,2] = b
        img[:,:,3] = alpha
        cv2.imshow("img",alpha)
        cv2.waitKey(waitKey)
        
    def show(self) -> None:
        for i in range (0, self._nums):
            self.showSingle(i,100)
            
 
        
        
if __name__ == '__main__':
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    npy_path = os.path.join(curr_dir, "../data/spot_source.npy")
    
    nv = NpyVisuallizer(npy_path)
    nv.show()
    cv2.waitKey(0)