import os
import sys
import numpy as np
import math
from scipy.signal import get_window
import matplotlib.pyplot as plt
import stft
import utilFunctions as UF
eps = np.finfo(float).eps

from utilFunctions import wavread
from scipy.fftpack import fft
"""
A4-Part-2: Measuring noise in the reconstructed signal using the STFT model 

A function that measures the amount of noise. Use SNR (signal to noise ratio) in dB to quantify the amount of noise. 

I compute two different SNR values for the following cases:

1) SNR1: Over the entire length of the signal
2) SNR2: For the segment of the signal left after discarding M samples from both the start and the 
end, where M is the analysis window length.

The input arguments to the function are the wav file name including the path (inputFile), window 
type (window), window length (M), FFT size (N), and hop size (H). The function returns a python 
tuple of both the SNR values in decibels: (SNR1, SNR2). Both SNR1 and SNR2 are float values. 
"""
def computeSNR(inputFile, window, M, N, H):
    """
    Input:
            inputFile (string): wav file name including the path 
            window (string): analysis window type (choice of rectangular, triangular, hanning, hamming, 
                    blackman, blackmanharris)
            M (integer): analysis window length (odd positive integer)
            N (integer): fft size (power of two, > M)
            H (integer): hop size for the stft computation
    Output:
            The function returns a python tuple of both the SNR values (SNR1, SNR2)
            SNR1 and SNR2 are floats.
    """
    w = get_window(window, M)
    fs, x = wavread(inputFile)
    mX, pX = stft.stftAnal(x, fs, w, N, H)
    y = stft.stftSynth(mX, pX, M, H)
    y = y[:len(x)]
    noise = y-x
    X = abs(fft(x))
    Y = abs(fft(noise))
    X[X<eps] = eps
    Y[Y<eps] = eps


    Es1 = En1 = 0
    for k in range(0, len(x)):
        Es1 = Es1+X[k]*X[k]
        En1 = En1+Y[k]*Y[k]
    SNR1 = 10*np.log10(float(Es1)/En1)


    x2 = x[M:len(x)-M]
    y2 = y[M:len(y)-M]
    noise2 = y2-x2
    X2 = abs(fft(x2))
    Y2 = abs(fft(noise2))
    X2[X2<eps] = eps
    Y2[Y2<eps] = eps


    Es2 = En2 = 0
    for k in range(0, len(x2)):
        Es2 = Es2+X2[k]*X2[k]
        En2 = En2+Y2[k]*Y2[k]
    SNR2 = 10*np.log10(float(Es2)/En2)


    return(SNR1, SNR2)
