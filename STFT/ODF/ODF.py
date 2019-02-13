import os
import sys
import numpy as np
from scipy.signal import get_window
import matplotlib.pyplot as plt
import math
import stft as STFT
import utilFunctions as UF

eps = np.finfo(float).eps

import time
from scipy.signal.windows import hamming
from scipy.fftpack import fft

"""
Computing onset detection function

A function computing a simple onset detection function (ODF) using the STFT. I compute two ODFs one 
for each of the frequency bands, low and high. The low frequency band is the set of all the frequencies 
from 0 - 3000 Hz and the high frequency band is the set of all the frequencies from 3000 - 10000 Hz. 

The input arguments to the function are the wav file name including the path (inputFile), window type (window),
window length (M), FFT size (N), and hop size (H). The function returns a numpy array with two columns, 
where the first column is the ODF computed on the low frequency band and the second column is the ODF computed
on the high frequency band.

"""

def computeODF(inputFile='piano.wav', window='hamming', M=1001, N=1024, H=256):
    """
    Inputs:
            inputFile (string): input sound file (monophonic with sampling rate of 44100)
            window (string): analysis window type (choice of rectangular, triangular, hanning, hamming, 
                blackman, blackmanharris)
            M (integer): analysis window size (odd integer value)
            N (integer): fft size (power of two, bigger or equal than than M)
            H (integer): hop size for the STFT computation
    Output:
            The function should return a numpy array with two columns, where the first column is the ODF 
            computed on the low frequency band and the second column is the ODF computed on the high 
            frequency band.
            ODF[:,0]: ODF computed in band 0 < f < 3000 Hz 
            ODF[:,1]: ODF computed in band 3000 < f < 10000 Hz
    """
    
    w = get_window(window, M)
    fs, x = UF.wavread(inputFile)
    mX, pX = STFT.stftAnal(x, fs, w, N, H)
    mX = 10**(mX/20.0)
    El = np.zeros(mX.shape[0])
    Eh = np.zeros(mX.shape[0])
    l = int(np.floor(N*3000.0/fs))
    h = int(np.floor(N*10000.0/fs))
    
    for i in range(0, mX.shape[0]):
        for j in range(1, l+1):
            El[i] += mX[i][j]*mX[i][j]
        for k in range(l+1, h+1):
            Eh[i] += mX[i][k]*mX[i][k]

    El = 10*np.log10(El)
    Eh = 10*np.log10(Eh)
    
    Ol = np.zeros(mX.shape[0])
    Oh = np.zeros(mX.shape[0])
    for i in range(1, mX.shape[0]):
        Ol[i] = El[i]-El[i-1]
        Oh[i] = Eh[i]-Eh[i-1]
    Ol[Ol<0] = 0
    Oh[Oh<0] = 0
    ODF=  np.array([Ol,Oh])
    
    #plot STFT spectrogram
    mX = np.log10(mX)/20.0
    plt.figure(1, figsize=(9.5, 6))

    plt.subplot(211)
    numFrames = int(mX[:,0].size)
    frmTime = H*np.arange(numFrames)/float(fs)                             
    binFreq = np.arange(N/2+1)*float(fs)/N                         
    plt.pcolormesh(frmTime, binFreq, np.transpose(mX))
    plt.title('mX (piano.wav), M=1001, N=1024, H=256')
    plt.autoscale(tight=True)

    plt.subplot(212)
    numFrames = int(ODF[0,:].size)
    frmTime = H*np.arange(numFrames)/float(fs)                                              
    plt.plot(frmTime, ODF[0,:])
    plt.plot(frmTime, ODF[1,:])
    plt.title('ODF (piano.wav), M=1001, N=1024, H=256')
    plt.autoscale(tight=True)
    plt.legend(('low frequency','high frequency'))
    
    plt.tight_layout()
    plt.savefig('spectrogram.png')
    plt.show()
    

    return(int(mX[:,0].size), int(ODF[0,:].size))
    
