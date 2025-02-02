""" 
Artash Nath, Founder, MonitorMyOcean.com

My own functions, plus some functions forked from MBARI (Danelle Cline) .

"""
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.patches import Rectangle
import scipy
import pandas as pd
import cv2
import librosa
from scipy.signal import find_peaks
import sklearn
from scipy.ndimage import convolve
import scipy

IMAGE_SIZE = (224, 224)

# Helper Functions

def psd_1sec(x: np.array([]), sample_rate: int, freq_response_db: float):
    """
    Computes power spectral density (PSD) estimates in 1` second bins on the input signal x
    :param x:  sample array of raw float measurements (to be converted to volts)
    :param sample_rate:  sampling rate of the raw samples
    :param freq_response_db:  frequency response of the hydrophone
    :return: power spectral density, array of sample frequencies
    """

    # convert scaled voltage to volts
    v = x * 3

    # initialize empty spectrogram
    num_seconds = round(len(x) / sample_rate)
    nfreq = int(sample_rate / 2 + 1)
    sg = np.empty((nfreq, num_seconds), float)

    # get window for welch
    w = scipy.signal.get_window('hann', sample_rate)

    # process spectrogram
    spa = 1  # seconds per average
    for x in range(0, num_seconds):
        cstart = x * spa * sample_rate
        cend = cstart + spa * sample_rate
        f, psd = scipy.signal.welch(v[cstart:cend], fs=sample_rate, window=w, nfft=sample_rate)
        psd = 10 * np.log10(psd) + freq_response_db
        sg[:, x] = psd

    return sg, f

####################################################################################

def moving_average(arr, window):
    result = []
    padding = window // 2

    # Pad the array with zeros
    padded_arr = np.pad(arr, (padding, padding), mode='constant')

    # Calculate moving average for each window
    for i in range(padding, len(padded_arr) - padding):
        result.append(np.mean(padded_arr[i - padding:i + padding + 1]))

    return np.array(result)


####################################################################################

def energy_ts(sg:np.array, conf_dict:dict, mean = False):
    freq_min = conf_dict["freq_range"][0][0]
    freq_max = conf_dict["freq_range"][0][1]
    x = sg[freq_min:freq_max]
    if mean:
        x = np.mean(x, axis=0)
    return x

####################################################################################

def find_windows(sg, conf):
    
    
    duration = conf['duration_secs']
    left_w = int(duration/2)
    right_w = duration-left_w
    threshold=0.015
    width = 10
    prom = 5
    if conf['name'] == 'BlueD':
        duration *=0.5
        threshold=0.0075
        width = 2
        prom = 2
    
    high_freq = conf['freq_range'][0][1]
    low_freq = conf['freq_range'][0][0]
    
    a = energy_ts(sg, conf, mean=True)
    am = moving_average(a, 10)
    peaks = find_peaks(am, threshold=threshold, distance=duration, prominence=prom, width=width)[0]
    windows = []
    for p in peaks:
        windows.append([p-left_w, p+right_w, low_freq, high_freq])
        
    return windows



def cut_windows(sg, conf, duration, overlap):
    windows = []
    x = 0
    high_freq = conf['freq_range'][0][1]
    low_freq = conf['freq_range'][0][0]
    while x < sg.shape[1]-duration:
        windows.append([x, x + duration, low_freq, high_freq])
        x+=overlap
    return windows

####################################################################################

def gauss2D(shape=(3, 3), sigma=0.5):
        """
        2D Gaussian mask to emulate MATLAB fspecial('gaussian',[shape],[sigma])
        :param shape (x,y) shape of the mask
        :sigma sigma of the mask
        """
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        mask = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
        mask[mask < np.finfo(mask.dtype).eps * mask.max()] = 0
        mask_sum = mask.sum()
        if mask_sum != 0:
            mask /= mask_sum
        return mask

####################################################################################

def smooth(fft_array, blur_type):
        """
        Smooth fft either in time/frequency, or if empty, both
        :param fft_array: fft array
        :param blur_type:  type of blur either 'time' or 'frequency'
        :return:
        # """
        if blur_type == 'time':
            smoothed_fft_array = convolve(fft_array, weights=gauss2D(shape=(5, 1), sigma=1.0))
        elif blur_type == 'frequency':
            smoothed_fft_array = convolve(fft_array, weights=gauss2D(shape=(1, 5), sigma=1.0))
        else:
            smoothed_fft_array = convolve(fft_array, weights=gauss2D(shape=(2, 2), sigma=1.0))

        return smoothed_fft_array

####################################################################################

cm_data = [[0.2081, 0.1663, 0.5292], [0.2116238095, 0.1897809524, 0.5776761905],
 [0.212252381, 0.2137714286, 0.6269714286], [0.2081, 0.2386, 0.6770857143],
 [0.1959047619, 0.2644571429, 0.7279], [0.1707285714, 0.2919380952,
  0.779247619], [0.1252714286, 0.3242428571, 0.8302714286],
 [0.0591333333, 0.3598333333, 0.8683333333], [0.0116952381, 0.3875095238,
  0.8819571429], [0.0059571429, 0.4086142857, 0.8828428571],
 [0.0165142857, 0.4266, 0.8786333333], [0.032852381, 0.4430428571,
  0.8719571429], [0.0498142857, 0.4585714286, 0.8640571429],
 [0.0629333333, 0.4736904762, 0.8554380952], [0.0722666667, 0.4886666667,
  0.8467], [0.0779428571, 0.5039857143, 0.8383714286],
 [0.079347619, 0.5200238095, 0.8311809524], [0.0749428571, 0.5375428571,
  0.8262714286], [0.0640571429, 0.5569857143, 0.8239571429],
 [0.0487714286, 0.5772238095, 0.8228285714], [0.0343428571, 0.5965809524,
  0.819852381], [0.0265, 0.6137, 0.8135], [0.0238904762, 0.6286619048,
  0.8037619048], [0.0230904762, 0.6417857143, 0.7912666667],
 [0.0227714286, 0.6534857143, 0.7767571429], [0.0266619048, 0.6641952381,
  0.7607190476], [0.0383714286, 0.6742714286, 0.743552381],
 [0.0589714286, 0.6837571429, 0.7253857143],
 [0.0843, 0.6928333333, 0.7061666667], [0.1132952381, 0.7015, 0.6858571429],
 [0.1452714286, 0.7097571429, 0.6646285714], [0.1801333333, 0.7176571429,
  0.6424333333], [0.2178285714, 0.7250428571, 0.6192619048],
 [0.2586428571, 0.7317142857, 0.5954285714], [0.3021714286, 0.7376047619,
  0.5711857143], [0.3481666667, 0.7424333333, 0.5472666667],
 [0.3952571429, 0.7459, 0.5244428571], [0.4420095238, 0.7480809524,
  0.5033142857], [0.4871238095, 0.7490619048, 0.4839761905],
 [0.5300285714, 0.7491142857, 0.4661142857], [0.5708571429, 0.7485190476,
  0.4493904762], [0.609852381, 0.7473142857, 0.4336857143],
 [0.6473, 0.7456, 0.4188], [0.6834190476, 0.7434761905, 0.4044333333],
 [0.7184095238, 0.7411333333, 0.3904761905],
 [0.7524857143, 0.7384, 0.3768142857], [0.7858428571, 0.7355666667,
  0.3632714286], [0.8185047619, 0.7327333333, 0.3497904762],
 [0.8506571429, 0.7299, 0.3360285714], [0.8824333333, 0.7274333333, 0.3217],
 [0.9139333333, 0.7257857143, 0.3062761905], [0.9449571429, 0.7261142857,
  0.2886428571], [0.9738952381, 0.7313952381, 0.266647619],
 [0.9937714286, 0.7454571429, 0.240347619], [0.9990428571, 0.7653142857,
  0.2164142857], [0.9955333333, 0.7860571429, 0.196652381],
 [0.988, 0.8066, 0.1793666667], [0.9788571429, 0.8271428571, 0.1633142857],
 [0.9697, 0.8481380952, 0.147452381], [0.9625857143, 0.8705142857, 0.1309],
 [0.9588714286, 0.8949, 0.1132428571], [0.9598238095, 0.9218333333,
  0.0948380952], [0.9661, 0.9514428571, 0.0755333333],
 [0.9763, 0.9831, 0.0538]]

from matplotlib.colors import LinearSegmentedColormap
parula_map = LinearSegmentedColormap.from_list('parula', cm_data)

def colorizeDenoise(samples):
        """
        Colorize, denoise colored image and save spectrogram
        :param samples:  stft array
        :param plotpath: path to the file to save the output to
        :return:
        """
        # Resize and use linear color map
        stft_resize = cv2.resize(samples, IMAGE_SIZE, cv2.INTER_AREA)
        stft_scaled = np.int16(stft_resize / (stft_resize.max() / 255.0))
        img = parula_map(stft_scaled)
        img_rescale = (img * 255).astype('uint8')
        img_denoise = cv2.fastNlMeansDenoisingColored(img_rescale,None,10,10,7,21)
        #im_rgb = cv2.cvtColor(np.flipud(img_denoise), cv2.COLOR_BGR2RGB)
        
        return img_denoise