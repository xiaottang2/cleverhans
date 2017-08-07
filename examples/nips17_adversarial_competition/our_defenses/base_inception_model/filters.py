import os
import numpy as np
import subprocess
import collections
import re
from PIL import Image
from skimage import data, img_as_float
from skimage.restoration import denoise_bilateral, denoise_nl_means
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import medfilt

class DownUpSamplingFilter(object):
    """ DownUpSampling Filter """
    def __init__(self):
        pass

    def __call__(self, image, ratio=0.5):
        """ image: a PIL Image """
        w, h = image.size
        image = image.resize((int(np.floor(w * ratio)), int(np.floor(h * ratio))),  Image.ANTIALIAS)
        image = image.resize((w, h), Image.ANTIALIAS)
        return Image.fromarray(np.uint8(image))

class GaussianFilter(object):
    """ Median Filter """
    def __init__(self, sigma=2):
        self.sigma = sigma

    def __call__(self, image):
        """ image: a PIL Image """
        image = np.array(image)
        image = gaussian_filter(image, self.sigma)
        return Image.fromarray(np.uint8(image))

class JpegCompressor(object):
    """ Jpeg compressor """
    def __init__(self):
        pass

    def __call__(self, image):
        """ image: a PIL image """
        image.save('tmp.jpg', 'JPEG', quality=10)
        compressed_image = Image.open('tmp.jpg')
        subprocess.call(['rm', 'tmp.jpg'])
        return compressed_image

class MedianFilter(object):
    """ Median Filter """
    def __init__(self, window=None):
        self.window_size = window

    def __call__(self, image):
        """ image: a PIL Image """
        image = np.array(image)
        image = medfilt(image, self.window_size)
        return Image.fromarray(np.uint8(image))

class NonLocalMeanFilter(object):
    """ Non local Filter """
    def __init__(self):
        pass

    def __call__(self, image):
        """ image: a PIL Image """
        image = np.array(image, dtype='float32')
        denoised_image = denoise_nl_means(image, 2, 4, 10, multichannel=True)
        return Image.fromarray(np.uint8(denoised_image))

class BilateralFilter(object):
    """ Bilateral Filter """
    def __init__(self):
        pass

    def __call__(self, image):
        """ image: a PIL Image """
        image = np.array(image, dtype='float32')
        image /= 255
        denoised_image = denoise_bilateral(image, sigma_color=0.05, sigma_spatial=15)
        denoised_image *= 255
        return Image.fromarray(np.uint8(denoised_image))

class LULUFilter(object):
    """ LULU Filter """
    def __init__(self, window=None, padding='edge'):
        """
            window: a 2-item list of horizontal and vertical window size
            padding: 
                'zero' - pad zeros on edge pixels
                'edge' - pad values of edge pixels
        """
        if (isinstance(window, int) and window < 2) or (isinstance(window, collections.Iterable) and min(window) < 2):
            raise ValueError('window size must be larger than 1.')
        if isinstance(window, int):
            self.win_r = window
            self.win_c = window
        elif isinstance(window, collections.Iterable):
            self.win_r = window[0]
            self.win_c = window[1]
        else:
            raise ValueError('\'window\' must be an Iterable.')
        self.padding = padding

    def __call__(self, image, mode='LU'):
        """ image: a PIL Image """
        if re.sub(r'[LU]', '', mode) is not '':
            raise ValueError('\'mode\' has to be a combination of \'L\' and \'U\'.')
        if len(mode) > 4:
            raise ValueError('It is useless to have more than 4 operations.')

        image = np.array(image)
        numrow, numcol, numchannel = image.shape
        if self.padding == 'zero':
            padded_image = np.pad(image, ((self.win_r-1, self.win_r-1), (self.win_c-1, self.win_c-1), (0, 0)), 'constant', constant_values=0)
        elif self.padding == 'edge':
            padded_image = np.pad(image, ((self.win_r-1, self.win_r-1), (self.win_c-1, self.win_c-1), (0, 0)), 'edge')

        for char in mode:
            if char is 'L':
                for i in range(numrow):
                    for j in range(numcol):
                        padded_image[i+self.win_r-1, j+self.win_c-1] = self.opL(i+self.win_r-1, j+self.win_c-1, numchannel, padded_image)
            else:
                for i in range(numrow):
                    for j in range(numcol):
                        padded_image[i+self.win_r-1, j+self.win_c-1] = self.opU(i+self.win_r-1, j+self.win_c-1, numchannel, padded_image)

        result_image = padded_image[self.win_r-1:-(self.win_r-1), self.win_c-1:-(self.win_c-1)]
        return Image.fromarray(np.uint8(result_image))
    
    def opL(self, i, j, numchannel, image):
        outer_seq = []
        max_seq = []
        for r in range(self.win_r):
            for c in range(self.win_c):
                inner_seq = image[i-r:i+self.win_r-r, j-c:j+self.win_c-c]
                min_seq = []
                for channel in range(numchannel):
                    min_seq += [np.min(inner_seq[:, :, channel])]
                outer_seq += [min_seq]
        outer_seq = np.array(outer_seq)
        for channel in range(numchannel):
            max_seq += [np.max(outer_seq[:, channel])]
        return max_seq

    def opU(self, i, j, numchannel, image):
        outer_seq = []
        min_seq = []
        for r in range(self.win_r):
            for c in range(self.win_c):
                inner_seq = image[i-r:i+self.win_r-r, j-c:j+self.win_c-c]
                max_seq = []
                for channel in range(numchannel):
                    max_seq += [np.max(inner_seq[:, :, channel])]
                outer_seq += [max_seq]
        outer_seq = np.array(outer_seq)
        for channel in range(numchannel):
            min_seq += [np.min(outer_seq[:, channel])]
        return min_seq
