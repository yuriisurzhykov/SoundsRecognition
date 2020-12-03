import matplotlib.pyplot as plt
from scipy.io import wavfile as wav
import scipy.io.wavfile as sciread
from scipy.fftpack import fft
import numpy as np
import csv
from PIL import Image

FILENAMES = 'resourses/{}/{}{}.wav'
CLASSIFIED_DATA = 'resourses/results/data'
SOUNDS = {'glass': 'glass',
          'snowball': 'snowball',
          'birds': 'birds'}

def activity_function(list_w, list_x, sigma):
    sum = 0
    for w, x in zip(list_w, list_x):
        sum += (w - x)**2
    result = np.e**(-sum/sigma**2)
    return result


def classify(input_x, X, Y, return_act=True):
    # список сумм
    sums = {}
    # список значений активностей
    activity_fun_list = []
    for y in Y:
        if y not in sums.keys():
            sums.update({y: 0})
    for x, y in zip(X, Y):
        act = activity_function(input_x, x, 0.1)
        activity_fun_list.append(act)
        sums[y] += act
    max_sum = max(sums.values())
    if max_sum < 0.001 and return_act:
        return None, activity_fun_list, sums
    elif max_sum < 0.001:
        return None
    for k, v in sums.items():
        if v == max_sum:
            pred_y = k
    if return_act:
        return pred_y, activity_fun_list, sums
    else:
        return pred_y 


def signal(data, rate):
    fft_signal = np.abs(fft(data))
    N = len(data)
    freq = [i * rate / N for i in range(0, len(fft_signal))]
    freq = [f for f in freq if 8000 >= f >= 200]
    fft_signal = fft_signal[:len(freq)]
    step = len(fft_signal)/10
    attr = []
    f = []
    for i in range(0, int(len(fft_signal)-step), int(step)):
        f.append((freq[i] + freq[int(i+step)])/2)
        attr.append(np.mean(fft_signal[i:int(i+step)]))
    return attr, f
    # return fft_signal, freq

examples = []
y = []
def train(epoches):
    for key in SOUNDS.keys():
        print('Epoche No {}'.format(epoches))
        for i in range(0, 20*epoches):
            rate, data = sciread.read(FILENAMES.format(key, key, i%20+1))
            data = data[:]
            attr, freq = signal(data, rate)
            examples.append(attr)
            y.append(SOUNDS[key])
"""
train(50)
with open("{}_list.csv".format(CLASSIFIED_DATA), "w+", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(examples)
with open("{}_labels.csv".format(CLASSIFIED_DATA), "w+", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(y)

"""