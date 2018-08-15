import numpy as np
import sys
sys.path.append('../../software/models/')
from scipy.fftpack import fft
import matplotlib.pyplot as plt
from scipy.signal import get_window
from dftModel import dftAnal
def zpFFTsizeExpt(x, fs):
    """
    I obtain the positive half of the FFT magnitude spectrum (in dB) for the following cases:
    Case-1: Input signal xseg (256 samples), window w1 (256 samples), and FFT size of 256
    Case-2: Input signal x (512 samples), window w2 (512 samples), and FFT size of 512
    Case-3: Input signal xseg (256 samples), window w1 (256 samples), and FFT size of 512 (Implicitly does a 
            zero-padding of xseg by 256 samples)
    Inputs:
        x (numpy array) = input signal (2*M samples long)
        fs (float) = sampling frequency in Hz
    Output:
        The function should return a tuple (mX1_80, mX2_80, mX3_80)
        mX1_80 (numpy array): The first 80 samples of the magnitude spectrum output of dftAnal for Case-1
        mX2_80 (numpy array): The first 80 samples of the magnitude spectrum output of dftAnal for Case-2
        mX3_80 (numpy array): The first 80 samples of the magnitude spectrum output of dftAnal for Case-3
        
    """
    M = len(x)/2
    xseg = x[:M]
    w1 = get_window('hamming',M)
    w2 = get_window('hamming',2*M)
    mX1, pX1 = dftAnal(xseg, w1, M)
    mX2, pX2 = dftAnal(x, w2, 2*M)
    mX3, pX3 = dftAnal(xseg, w1, 2*M)
    mX1_80 = mX1[:80]
    mX2_80 = mX2[:80]
    mX3_80 = mX3[:80]

    t = np.arange(0, 80, 1)
    plt.plot(t, mX1[:80])
    plt.plot(t, mX2[:80])
    plt.plot(t, mX3[:80])
    plt.show()


    return(mX1_80, mX2_80, mX3_80)

