import os
import imageio
import numpy as np
from PIL import Image
from scipy import signal
import cv2

class GaussianBlur(object):
    def __init__(self, kernel_size=3, sigma=1.5):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.kernel = self.gaussian_kernel()
 
    def gaussian_kernel(self):
        kernel = np.zeros(shape=(self.kernel_size, self.kernel_size), dtype=np.float32)
        radius = self.kernel_size//2
        for y in range(-radius, radius + 1):  # [-r, r]
            for x in range(-radius, radius + 1):
                #
                v = 1.0 / (2 * np.pi * self.sigma ** 2) * np.exp(-1.0 / (2 * self.sigma ** 2) * (x ** 2 + y ** 2))
                kernel[y + radius, x + radius] = v  #
        kernel2 = kernel / np.sum(kernel)
        return kernel2
 
    def filter(self, img: Image.Image):
        img_arr = np.array(img)
        if len(img_arr.shape) == 2:
            new_arr = signal.convolve2d(img_arr, self.kernel, mode="same", boundary="symm")
        else:
            h, w, c = img_arr.shape
            new_arr = np.zeros(shape=(h, w, c), dtype=np.float32)
            for i in range(c):
                new_arr[..., i] = signal.convolve2d(img_arr[..., i], self.kernel, mode="same", boundary="symm")
        new_arr = np.array(new_arr, dtype=np.uint8)
        return Image.fromarray(new_arr)

def read_dAnything(filename, G_para):
    org = Image.open(filename)
    imgs = org.split()

    if len(imgs) == 3:
        img = org
    elif len(imgs) == 1:
        # label = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        label = imgs[0]
        # img = Image.merge("RGB", (g, g, g))
        sigma1=int(G_para[2])
        kernel_size1 = int(G_para[0])
        out1 = GaussianBlur(kernel_size1, sigma1).filter(label)
        sigma2=int(G_para[3])
        kernel_size2 = int(G_para[1])
        out2 = GaussianBlur(kernel_size2, sigma2).filter(label)
        img = Image.merge("RGB",(label, out1, out2))
        # print(sigma1,sigma2,kernel_size1,kernel_size2)
    else:
        raise Exception("Images must have 3 channels or 1.")

    return img

#
if __name__ =="__main__":
    path = "./WHU=MVS/test/Anything"
    out_path = "./WHU=MVS/test/depthAnything"
    views = int(5)
    for i in range(views):
        name_list = [x for x in os.listdir(os.path.join(path,'{}'.format(i))) if x.endswith('png')]
        out_dir = os.path.join(out_path, '{}'.format(i))
        os.makedirs(out_dir, exist_ok=True)
        for name in name_list:
            deps_path= os.path.join(path, '{}/{}'.format(i, name))
            # label = read_dAnything(label_dir, [3,3,2,4])
            label = read_dAnything(deps_path, [3,25,2,2])
            imageio.imsave(os.path.join(out_dir, name[:-4]+'.png'), label)