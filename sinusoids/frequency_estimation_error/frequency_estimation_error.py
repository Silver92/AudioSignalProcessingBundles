import numpy as np
from scipy.signal import get_window
import math
import os
import sys
import utilFunctions as UF
import dftModel as DFT
""" 
Minimizing the frequency estimation error of a sinusoid

A function that estimates the frequency of a sinusoidal signal at a given time instant. The 
function returns the estimated frequency in Hz, together with the window size and the FFT 
size used in the analysis.  

The input arguments to the function are the wav file name including the path (inputFile) containing 
the sinusoidal signal, and the frequency of the sinusoid in Hz (f). The frequency of the input sinusoid  
can range between 100Hz and 2000Hz. The function returns a three element tuple of the estimated 
frequency of the sinusoid (fEst), the window size (M) and the FFT size (N) used.

The window size is the minimum positive integer of the form 100*k + 1 (where k is a 
positive integer) for which the frequency estimation error is < 0.05 Hz. For a window size M,
FFT size (N) is the smallest power of 2 larger than M. 


"""
def minFreqEstErr(inputFile='sine-440.wav', f=440):
    """
    Inputs:
            inputFile (string) = wav file including the path
            f (float) = frequency of the sinusoid present in the input audio signal (Hz)
    Output:
            fEst (float) = Estimated frequency of the sinusoid (Hz)
            M (int) = Window size
            N (int) = FFT size
    """

    window = 'blackman'
    t = -40
    (fs, x) = UF.wavread(inputFile)
    k = 11
    N = 2

    while True:
        M = 100*k+1
        while N<M:
            N = N*2
        w = get_window(window, M)
        x1 = x[.5*fs:.5*fs+M]
        mX, pX = DFT.dftAnal(x1, w, N)
        ploc = UF.peakDetection(mX, t)
        iploc, ipmag, ipphase = UF.peakInterp(mX, pX, ploc)
        fEst = fs*iploc[0]/float(N)
        if abs(fEst-f) < 0.05:
            break
        else:
            k += 1

    return(fEst, M, N)












