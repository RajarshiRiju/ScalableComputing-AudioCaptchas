# feature extractoring and preprocessing data
import librosa
import matplotlib.pyplot as plt
import os
from PIL import Image
import glob
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf

# Plotting the spectrogram
cmap = plt.get_cmap('inferno')

with tf.device('/device:GPU:0'):
    count = 0
    for filename in os.listdir('Audio_predict/'):
        audioname = 'Audio_predict/' + filename
        plt.figure(figsize=(10,6))
        y, sr = librosa.load(audioname, mono=True)
        plt.specgram(y, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap=cmap, sides='default', mode='default', scale='dB');
        plt.axis('off');
        plt.savefig('Audio_to_predict/' + filename[:-3].replace(".", "") + '.png')
        plt.clf()
        count += 1
        print (count)

