from scipy.io import wavfile as wav
import scipy.io.wavfile as sciread
from scipy.fftpack import fft
import numpy as np
import random
import csv
from PIL import Image
from train_module import classify
from train_module import signal
import matplotlib.pyplot as plt


FILENAMES = 'resourses/{}/{}{}.wav'
PHOTOS = 'resourses/pictures/{}/{}{}.jpg'
CLASSIFIED_DATA = 'resourses/results/data'
TEST_SOUND = 'resourses/test/test10.wav'

SOUNDS = {'glass': 'glass',
          'snowball': 'snowball',
          'birds': 'birds'}

PHRASES = {'glass': ['Вы что, разбили стекло?', 'Упс, похоже, что у вас разбилось что-то стеклянное...', 'Как вы могли разбить стекло, зачем?', 'И что будем делать с разбитым стеклом?'],
           'snowball': ['А можно и мне с вами в снежки?', 'В вас что, прилетел снежок?', 'А на улице уже зима???', 'Эх, снежки, наверное, самая иинтересная игра замой...'],
           'birds': ['Ой, птичка крикнула...']}


with open("{}_list.csv".format(CLASSIFIED_DATA), "r", newline="") as f:
    reader = csv.reader(f)
    examples=list(reader)
with open("{}_labels.csv".format(CLASSIFIED_DATA), "r", newline="") as f:
    reader = csv.reader(f)
    y=list(next(reader))
examples = [np.array([float(i) for i in e]) for e in examples]

A = 0
for w in examples:
    if max(w) > A:
        A = max(w)
examples_2 = []
for w in examples:
    examples_2.append(w/A)

    
fig, axs = plt.subplots(1, 3)
fig.set_figwidth(15)
fig.set_figheight(5)
rate, data = sciread.read(TEST_SOUND)
data = data[:]
axs[0].plot(data)
fftSignal = np.abs(fft(data))
attr, freq = signal(data, rate)
axs[2].set_ylim([0, 1])
axs[2].plot(freq, attr/A, 'b-o')
plt.show()
pred, list_act, sums = classify(attr/A, examples_2, y, return_act=True)
if pred is not None:
    path = PHOTOS.format(pred, pred, random.choice(range(1, 7)))
    im = Image.open(path)
    im.show()
    print(random.choice(PHRASES[pred]))
else:
    print("Какой-то невнятный звук, здесь я безсилен :(")
